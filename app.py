"""
Lost & Found Item Matcher — Streamlit Frontend
AI-powered matching using Bi-LSTM
"""

import re
import json
import torch
import torch.nn as nn
import streamlit as st


# ──────────────────────────────────────────────
# 1. Model Definition (must match training)
# ──────────────────────────────────────────────
class BiLSTMMatchingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super(BiLSTMMatchingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        classifier_input_dim = 3 * (2 * hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def encode(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)

    def forward(self, lost_input, found_input):
        lost_repr = self.encode(lost_input)
        found_repr = self.encode(found_input)
        diff = torch.abs(lost_repr - found_repr)
        combined = torch.cat([lost_repr, found_repr, diff], dim=1)
        return self.classifier(combined).squeeze(1)


# ──────────────────────────────────────────────
# 2. Text Processing Utilities
# ──────────────────────────────────────────────
TEMPLATE_KEYWORDS = {'item', 'color', 'brand', 'location'}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_indices(text, vocab):
    return [vocab.get(word, vocab.get('<UNK>', 1)) for word in text.split()]


def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))


def analyze_token_coverage(text, vocab):
    cleaned = clean_text(text)
    words = cleaned.split()
    content_words = [w for w in words if w not in TEMPLATE_KEYWORDS]
    known_words = [w for w in content_words if w in vocab]
    unknown_words = [w for w in content_words if w not in vocab]
    coverage = len(known_words) / len(content_words) if content_words else 0.0
    return {
        'total_content': len(content_words),
        'known_count': len(known_words),
        'unknown_count': len(unknown_words),
        'coverage': coverage,
        'known_words': known_words,
        'unknown_words': unknown_words
    }


# ──────────────────────────────────────────────
# 3. Load Model & Vocabulary (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model_and_vocab():
    device = torch.device('cpu')
    with open('vocabulary.json', 'r') as f:
        vocab = json.load(f)
    checkpoint = torch.load('lost_found_model_complete.pth', map_location=device)
    model = BiLSTMMatchingModel(
        checkpoint['vocab_size'], checkpoint['embed_dim'],
        checkpoint['hidden_dim'], checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, vocab, checkpoint['max_len'], device


# ──────────────────────────────────────────────
# 4. Prediction Function
# ──────────────────────────────────────────────
def predict_match(lost_text, found_text, model, vocab, max_len, device):
    lost_clean = clean_text(lost_text)
    found_clean = clean_text(found_text)
    lost_ids = pad_sequence(text_to_indices(lost_clean, vocab), max_len)
    found_ids = pad_sequence(text_to_indices(found_clean, vocab), max_len)
    lost_tensor = torch.tensor([lost_ids], dtype=torch.long).to(device)
    found_tensor = torch.tensor([found_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = model(lost_tensor, found_tensor).item()
    return prob, lost_ids, found_ids


# ──────────────────────────────────────────────
# 5. Field Comparison Helper
# ──────────────────────────────────────────────
def compare_attribute(lost_val, found_val):
    """
    Compares an attribute between lost and found items.
    Returns: (symbol, css_class, display_text)
    """
    l = lost_val.strip()
    f = found_val.strip()

    if not l and not f:
        return ("—", "status-empty", "Not specified")

    if l and f:
        l_low = l.lower()
        f_low = f.lower()
        if l_low == f_low or l_low in f_low or f_low in l_low:
            # Matching
            display_val = l if len(l) >= len(f) else f
            return ("✓", "status-match", display_val)
        else:
            # Conflict / Mismatch
            return ("✗", "status-mismatch", f"{l} (Lost) vs {f} (Found)")

    # One specified, one missing / uncertain
    specified = l if l else f
    return ("?", "status-uncertain", specified)


# ──────────────────────────────────────────────
# 6. Streamlit UI
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Lost & Found Item Matcher",
        page_icon="🔍",
        layout="centered"
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

        /* Force light theme */
        .stApp { background-color: #f8fafc !important; }

        /* Hide Streamlit UI chrome */
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }

        /* Global font */
        html, body, .stApp, .stMarkdown, label, input, textarea, button, p, span, div {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* ── Header ── */
        .page-header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #ffffff;
            padding: 1.8rem 2.2rem;
            border-radius: 12px;
            margin-bottom: 1.8rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
            border: 1px solid #334155;
        }
        .page-header h1 {
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0;
            color: #ffffff;
            letter-spacing: -0.02em;
        }
        .page-header p {
            font-size: 0.92rem;
            color: #94a3b8;
            margin: 0.35rem 0 0 0;
            font-weight: 500;
        }

        /* ── Section labels ── */
        .section-label {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding-bottom: 0.45rem;
            margin-bottom: 0.8rem;
            border-bottom: 2px solid #e2e8f0;
        }
        .section-lost  { color: #d97706; border-bottom-color: #f59e0b; }
        .section-found { color: #059669; border-bottom-color: #10b981; }

        /* ── Form Inputs ── */
        .stTextInput label, .stTextArea label {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: #475569 !important;
            margin-bottom: 0.2rem !important;
        }
        .stTextInput input, .stTextArea textarea {
            background: #ffffff !important;
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 8px !important;
            font-size: 0.9rem !important;
            color: #1e293b !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        }

        /* ── Match Button ── */
        .stButton > button {
            background: #1e293b !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 2rem !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: #334155 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.2) !important;
        }

        /* ── Result Card ── */
        .result-container {
            margin-top: 1.8rem;
            display: flex;
            justify-content: center;
        }
        .result-box {
            width: 100%;
            max-width: 480px;
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.8rem 2rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            text-align: center;
        }
        .result-box.match {
            border-color: #22c55e;
            background: #ffffff;
        }
        .result-box.nomatch {
            border-color: #ef4444;
            background: #ffffff;
        }

        .result-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }
        .result-box.match .result-title {
            color: #16a34a;
        }
        .result-box.nomatch .result-title {
            color: #dc2626;
        }

        .prob-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            margin-bottom: 0.2rem;
        }
        .prob-value {
            font-size: 2.8rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 1.5rem;
            letter-spacing: -0.03em;
        }
        .result-box.match .prob-value {
            color: #15803d;
        }
        .result-box.nomatch .prob-value {
            color: #b91c1c;
        }

        .attributes-table {
            width: 100%;
            border-top: 1px dashed #cbd5e1;
            padding-top: 1.2rem;
            text-align: left;
            font-size: 0.9rem;
        }
        .attr-row {
            display: flex;
            align-items: center;
            padding: 0.45rem 0;
            border-bottom: 1px solid #f1f5f9;
        }
        .attr-symbol {
            width: 24px;
            font-weight: 700;
            font-size: 1.05rem;
            display: inline-block;
        }
        .status-match { color: #16a34a; }
        .status-mismatch { color: #dc2626; }
        .status-uncertain { color: #d97706; }
        .status-empty { color: #94a3b8; }

        .attr-name {
            width: 85px;
            font-weight: 600;
            color: #334155;
        }
        .attr-val {
            flex: 1;
            color: #1e293b;
            font-weight: 500;
            font-size: 0.88rem;
        }

        .guidance-box {
            margin-top: 1.2rem;
            font-size: 0.88rem;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            text-align: left;
            line-height: 1.45;
        }
        .guidance-match {
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        .guidance-nomatch {
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }

        .alert-callout {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            margin-top: 0.9rem;
            text-align: left;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── 1. Header ──
    st.markdown("""
    <div class="page-header">
        <h1>Lost & Found Item Matcher</h1>
        <p>AI-powered matching using Bi-LSTM</p>
    </div>
    """, unsafe_allow_html=True)

    model, vocab, max_len, device = load_model_and_vocab()

    # ── 2 & 3. Lost Item and Found Item Sections ──
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-label section-lost">Lost Item</div>', unsafe_allow_html=True)
        lost_desc = st.text_area("Description", placeholder="e.g., Left my black wireless headphones on the desk while studying", key="ld", height=85)
        l1, l2 = st.columns(2)
        with l1:
            lost_category = st.text_input("Category", placeholder="e.g., Headphones", key="lc_cat")
            lost_color = st.text_input("Color", placeholder="e.g., Black", key="lc_col")
        with l2:
            lost_brand = st.text_input("Brand", placeholder="e.g., JBL", key="lb_brand")
            lost_location = st.text_input("Location", placeholder="e.g., Library", key="ll_loc")

    with col2:
        st.markdown('<div class="section-label section-found">Found Item</div>', unsafe_allow_html=True)
        found_desc = st.text_area("Description", placeholder="e.g., Found an over-ear black headphone near study table 4", key="fd", height=85)
        f1, f2 = st.columns(2)
        with f1:
            found_category = st.text_input("Category", placeholder="e.g., Headphones", key="fc_cat")
            found_color = st.text_input("Color", placeholder="e.g., Black", key="fc_col")
        with f2:
            found_brand = st.text_input("Brand", placeholder="e.g., JBL", key="fb_brand")
            found_location = st.text_input("Location", placeholder="e.g., Library", key="fl_loc")

    st.write("")

    # ── 4. Match Button ──
    if st.button("🔍 Find Match", use_container_width=True, type="primary"):
        if not lost_desc.strip() or not found_desc.strip():
            st.warning("Please enter a description for both the Lost Item and Found Item.")
            return

        lost_combined = f"{lost_desc} item {lost_category} color {lost_color} brand {lost_brand} location {lost_location}"
        found_combined = f"{found_desc} item {found_category} color {found_color} brand {found_brand} location {found_location}"

        # Analyze token coverage
        lost_cov = analyze_token_coverage(lost_combined, vocab)
        found_cov = analyze_token_coverage(found_combined, vocab)

        with st.spinner("Finding match..."):
            raw_prob, lost_ids, found_ids = predict_match(lost_combined, found_combined, model, vocab, max_len, device)

        # Attribute comparison
        cat_sym, cat_class, cat_txt = compare_attribute(lost_category, found_category)
        col_sym, col_class, col_txt = compare_attribute(lost_color, found_color)
        loc_sym, loc_class, loc_txt = compare_attribute(lost_location, found_location)
        brd_sym, brd_class, brd_txt = compare_attribute(lost_brand, found_brand)

        # Count mismatches
        mismatches = [sym for sym in (cat_sym, col_sym, loc_sym, brd_sym) if sym == "✗"]
        matches = [sym for sym in (cat_sym, col_sym, loc_sym, brd_sym) if sym == "✓"]

        # Detect <UNK> collision:
        # If words are different, but token IDs are identical because all words mapped to <UNK>
        is_unk_collision = (
            lost_ids == found_ids and
            clean_text(lost_combined) != clean_text(found_combined) and
            (lost_cov['unknown_count'] > 0 or found_cov['unknown_count'] > 0)
        )

        has_no_known_words = (lost_cov['known_count'] == 0 or found_cov['known_count'] == 0)

        # Validation Logic:
        # 1. If Category explicitly conflicts (e.g., Watch vs Bottle or gibberish vs gibberish), cannot be a match.
        # 2. If <UNK> collision occurred (all unknown words collapsed to identical <UNK> vectors), the 100% prob is false.
        # 3. If major attributes conflict (>=2 mismatches), override model hallucination.
        effective_prob = raw_prob
        override_reason = None

        if is_unk_collision or (has_no_known_words and len(mismatches) > 0):
            effective_prob = 0.0
            override_reason = "Out-Of-Vocabulary Collision: The entered words are not in the vocabulary and collapsed to unknown tokens (<UNK>), producing a false match."
        elif cat_sym == "✗":
            effective_prob = min(effective_prob, 0.15)
            override_reason = "Category Mismatch: The item categories are completely different."
        elif len(mismatches) >= 2:
            effective_prob = min(effective_prob, 0.20)
            override_reason = f"Multiple Attribute Conflicts: {len(mismatches)} attributes do not match."

        is_match = effective_prob >= 0.5
        box_class = "match" if is_match else "nomatch"
        title_text = "POSSIBLE MATCH" if is_match else "NO MATCH"
        pct_display = f"{effective_prob * 100:.0f}%"

        # ── 5. Result Section ──
        claim_location = found_location.strip() or lost_location.strip() or "the campus reception"
        guidance = (
            f"<strong>Match confirmed!</strong> You can claim this item at <strong>{claim_location}</strong>."
            if is_match else
            "<strong>Items do not match.</strong> Check back for new reports as items are turned in daily."
        )

        st.markdown(f"""
        <div class="result-container">
            <div class="result-box {box_class}">
                <div class="result-title">{title_text}</div>
                <div class="prob-label">Match Probability</div>
                <div class="prob-value">{pct_display}</div>
                <div class="attributes-table">
                    <div class="attr-row">
                        <span class="attr-symbol {cat_class}">{cat_sym}</span>
                        <span class="attr-name">Category</span>
                        <span class="attr-val">{cat_txt}</span>
                    </div>
                    <div class="attr-row">
                        <span class="attr-symbol {col_class}">{col_sym}</span>
                        <span class="attr-name">Color</span>
                        <span class="attr-val">{col_txt}</span>
                    </div>
                    <div class="attr-row">
                        <span class="attr-symbol {loc_class}">{loc_sym}</span>
                        <span class="attr-name">Location</span>
                        <span class="attr-val">{loc_txt}</span>
                    </div>
                    <div class="attr-row">
                        <span class="attr-symbol {brd_class}">{brd_sym}</span>
                        <span class="attr-name">Brand</span>
                        <span class="attr-val">{brd_txt}</span>
                    </div>
                </div>
                <div class="guidance-box {'guidance-match' if is_match else 'guidance-nomatch'}">
                    {guidance}
                </div>
                {f'<div class="alert-callout">⚠️ <strong>Notice:</strong> {override_reason}</div>' if override_reason else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Model Input Tokens (Debug Details)"):
            st.markdown(f"""
            **Token Analysis:**
            - **Lost Input:** `{clean_text(lost_combined)}`
              - Recognized words ({lost_cov['known_count']}): `{', '.join(lost_cov['known_words']) or 'None'}`
              - Unknown `<UNK>` words ({lost_cov['unknown_count']}): `{', '.join(lost_cov['unknown_words']) or 'None'}`
            - **Found Input:** `{clean_text(found_combined)}`
              - Recognized words ({found_cov['known_count']}): `{', '.join(found_cov['known_words']) or 'None'}`
              - Unknown `<UNK>` words ({found_cov['unknown_count']}): `{', '.join(found_cov['unknown_words']) or 'None'}`
            - **Token IDs Comparison:**
              - Lost IDs  : `{lost_ids[:15]}...`
              - Found IDs : `{found_ids[:15]}...`
              - Identical Token ID Streams? `{'Yes (<UNK> collision detected!)' if lost_ids == found_ids else 'No'}`
            - **Raw Bi-LSTM Output:** `{raw_prob:.4f}` ({raw_prob * 100:.1f}%)
            - **Effective Calibrated Score:** `{effective_prob:.4f}` ({effective_prob * 100:.1f}%)
            """)


if __name__ == "__main__":
    main()
