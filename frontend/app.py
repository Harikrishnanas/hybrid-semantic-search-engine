"""
Hybrid Semantic Search Engine — Streamlit Frontend
===================================================
Professional search engine UI (not a chatbot).
"""

import os
import time
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────── #
#  Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="Hybrid Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ─────────────────────────────────────────────────────────────────────────── #
#  Custom CSS — dark professional search engine theme
# ─────────────────────────────────────────────────────────────────────────── #

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #0f1117;
    --bg-card:       #1a1d27;
    --bg-card2:      #1e2130;
    --accent:        #4f8ef7;
    --accent2:       #7c5cfc;
    --accent-green:  #22c55e;
    --accent-orange: #f59e0b;
    --accent-red:    #ef4444;
    --text-primary:  #f0f4ff;
    --text-muted:    #8892a4;
    --border:        #2a2f42;
    --gradient:      linear-gradient(135deg, #4f8ef7 0%, #7c5cfc 100%);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: var(--bg-primary) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, .stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── App title banner ── */
.app-title {
    text-align: center;
    padding: 2rem 0 0.5rem;
}
.app-title h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-title p {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0.3rem 0 0;
}

/* ── Search bar styling ── */
.stTextInput > div > div > input {
    background: var(--bg-card2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 1.1rem !important;
    padding: 0.85rem 1.2rem !important;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.2s ease, transform 0.1s ease !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Cards ── */
.search-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.search-card h3 {
    color: var(--text-primary);
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.8rem;
    letter-spacing: 0.3px;
}

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e1a35 100%);
    border: 1px solid #2d3260;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
}
.answer-card p {
    color: var(--text-primary);
    line-height: 1.75;
    margin: 0;
}

/* ── Source badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.badge-doc   { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-ai    { background: rgba(124,92,252,0.15); color: #a78bfa; border: 1px solid rgba(124,92,252,0.3); }
.badge-hybrid { background: rgba(79,142,247,0.15); color: #93c5fd; border: 1px solid rgba(79,142,247,0.3); }
.badge-fallback { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }

/* ── Query type badge ── */
.qtype-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}
.qtype-doc    { background: rgba(34,197,94,0.1); color: #4ade80; }
.qtype-general { background: rgba(124,92,252,0.1); color: #a78bfa; }
.qtype-mixed  { background: rgba(79,142,247,0.1); color: #93c5fd; }

/* ── Progress bars ── */
.metric-bar-wrap {
    margin-bottom: 1rem;
}
.metric-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
    font-size: 0.85rem;
    color: var(--text-muted);
}
.metric-label span:last-child {
    font-weight: 600;
    color: var(--text-primary);
}
.metric-bar-bg {
    background: var(--border);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
.bar-strong  { background: linear-gradient(90deg, #22c55e, #4ade80); }
.bar-medium  { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
.bar-weak    { background: linear-gradient(90deg, #ef4444, #f87171); }
.bar-accent  { background: var(--gradient); }

/* ── Analytics grid ── */
.analytics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.7rem;
    margin-top: 0.5rem;
}
.analytics-item {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    text-align: center;
}
.analytics-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.analytics-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* ── Concept chips ── */
.concept-chip {
    display: inline-block;
    background: rgba(79,142,247,0.1);
    border: 1px solid rgba(79,142,247,0.25);
    color: #93c5fd;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem 0.2rem 0 0;
}

/* ── Related questions ── */
.rq-item {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.4rem;
    color: var(--accent);
    font-size: 0.87rem;
    cursor: pointer;
    transition: background 0.15s ease;
}
.rq-item:hover { background: rgba(79,142,247,0.1); }

/* ── Evidence chunks ── */
.evidence-item {
    background: var(--bg-card2);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}
.evidence-meta {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.evidence-tag {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}
.tag-page    { background: rgba(34,197,94,0.15); color: #4ade80; }
.tag-chunk   { background: rgba(245,158,11,0.15); color: #fcd34d; }
.tag-bm25    { background: rgba(79,142,247,0.15); color: #93c5fd; }
.tag-faiss   { background: rgba(124,92,252,0.15); color: #a78bfa; }
.tag-rerank  { background: rgba(239,68,68,0.15); color: #fca5a5; }
.evidence-text {
    color: var(--text-muted);
    font-size: 0.87rem;
    line-height: 1.6;
}

/* ── Topic heatmap ── */
.heatmap-item {
    margin-bottom: 0.6rem;
}
.heatmap-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    margin-bottom: 0.2rem;
}
.heatmap-label span:first-child { color: var(--text-primary); }
.heatmap-label span:last-child  { color: var(--text-muted); font-weight: 600; }

/* ── Section headers ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.9rem;
}

/* ── Doc info card ── */
.doc-info-card {
    background: linear-gradient(135deg, rgba(79,142,247,0.08) 0%, rgba(124,92,252,0.08) 100%);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.doc-info-name {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-info-stats {
    font-size: 0.78rem;
    color: var(--text-muted);
}

/* ── Spinner text ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    color: var(--accent) !important;
    font-weight: 600 !important;
    background: var(--bg-card2) !important;
    border-radius: 8px !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Sidebar file uploader ── */
.stFileUploader {
    background: var(--bg-card2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── divider ── */
hr { border-color: var(--border) !important; }

/* ── Metric numbers in sidebar ── */
.stat-row { display: flex; justify-content: space-between; margin: 0.25rem 0; }
.stat-label { font-size: 0.8rem; color: var(--text-muted); }
.stat-value { font-size: 0.8rem; font-weight: 700; color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────── #
#  Session state
# ─────────────────────────────────────────────────────────────────────────── #

for key, default in {
    "active_document": None,
    "uploader_key": 0,
    "last_result": None,
    "last_query": "",
    "doc_intelligence": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────── #
#  Helper: render a progress bar
# ─────────────────────────────────────────────────────────────────────────── #

def render_metric_bar(label: str, value: float, bar_class: str = "bar-accent", custom_strength: str = None):
    width = min(max(value, 0), 100)
    
    if custom_strength:
        strength = custom_strength
        if width >= 75:
            bar_class = "bar-strong"
        elif width >= 50:
            bar_class = "bar-medium"
        else:
            bar_class = "bar-weak"
    else:
        if value >= 80:
            strength = "Strong Match"
            bar_class = "bar-strong"
        elif value >= 40:
            strength = "Moderate Match"
            bar_class = "bar-medium"
        else:
            strength = "Weak Match"
            bar_class = "bar-weak"

    st.markdown(f"""
    <div class="metric-bar-wrap">
      <div class="metric-label">
        <span>{label}</span>
        <span>{value:.0f}% &nbsp;·&nbsp; {strength}</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill {bar_class}" style="width:{width}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_heatmap_bar(label: str, pct: float):
    width = min(max(pct, 0), 100)
    if pct >= 70:
        cls = "bar-strong"
    elif pct >= 35:
        cls = "bar-medium"
    else:
        cls = "bar-weak"
    st.markdown(f"""
    <div class="heatmap-item">
      <div class="heatmap-label">
        <span>{label}</span><span>{pct:.0f}%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill {cls}" style="width:{width}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────── #
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────── #

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
      <div style="font-size:1.3rem;font-weight:700;color:#f0f4ff;">🔍 Search Engine</div>
      <div style="font-size:0.78rem;color:#8892a4;margin-top:0.2rem;">Knowledge Discovery Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📄 Document Upload</div>', unsafe_allow_html=True)

    if st.session_state.active_document is None:
        uploaded_file = st.file_uploader(
            "Upload PDF (Optional)",
            type=["pdf"],
            key=f"uploader_{st.session_state.uploader_key}",
            help="Upload a PDF to enable document-grounded search",
        )

        if uploaded_file is not None:
            with st.spinner("🔄 Processing document..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    upload_res = requests.post(
                        f"{API_URL}/upload-pdf", files=files, timeout=60
                    )

                    if upload_res.status_code == 201:
                        file_info = upload_res.json()
                        file_path = file_info["file_path"]

                        index_res = requests.post(
                            f"{API_URL}/create-index",
                            json={"file_path": file_path},
                            timeout=300,
                        )

                        if index_res.status_code == 200:
                            index_info = index_res.json()
                            st.session_state.active_document = {
                                "filename": file_info["filename"],
                                "index_id": index_info["index_id"],
                                "num_pages": index_info["num_pages"],
                                "num_chunks": index_info["num_chunks"],
                            }

                            # Fetch document intelligence
                            di_res = requests.get(
                                f"{API_URL}/document-intelligence/{index_info['index_id']}",
                                params={"filename": file_info["filename"]},
                                timeout=30,
                            )
                            if di_res.status_code == 200:
                                st.session_state.doc_intelligence = di_res.json()

                            st.success("✅ Document indexed!")
                            st.rerun()
                        else:
                            st.error(f"Indexing failed: {index_res.text}")
                    else:
                        st.error(f"Upload failed: {upload_res.text}")

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        doc = st.session_state.active_document
        st.markdown(f"""
        <div class="doc-info-card">
          <div class="doc-info-name">📄 {doc['filename']}</div>
          <div class="doc-info-stats">
            <div class="stat-row">
              <span class="stat-label">Pages</span>
              <span class="stat-value">{doc['num_pages']}</span>
            </div>
            <div class="stat-row">
              <span class="stat-label">Chunks</span>
              <span class="stat-value">{doc['num_chunks']}</span>
            </div>
            <div class="stat-row">
              <span class="stat-label">Index ID</span>
              <span class="stat-value">{doc['index_id'][:12]}…</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Document Intelligence
        di = st.session_state.doc_intelligence
        if di:
            st.markdown('<div class="section-header">🧠 Document Intelligence</div>', unsafe_allow_html=True)

            if di.get("top_topics"):
                st.markdown("**Top Topics**")
                for topic in di["top_topics"]:
                    st.markdown(f"• {topic}")

            if di.get("coverage_heatmap"):
                st.markdown("**Knowledge Coverage**")
                for item in di["coverage_heatmap"]:
                    render_heatmap_bar(item["topic"], item["percentage"])

        if st.button("🗑️ Remove Document", use_container_width=True):
            st.session_state.active_document = None
            st.session_state.doc_intelligence = None
            st.session_state.uploader_key += 1
            st.session_state.last_result = None
            st.rerun()

    # ── Mode indicator ──
    st.markdown("---")
    if st.session_state.active_document:
        st.markdown("""
        <div style="background:rgba(79,142,247,0.08);border:1px solid rgba(79,142,247,0.2);
             border-radius:8px;padding:0.6rem 0.8rem;font-size:0.78rem;color:#93c5fd;">
        ⚡ <strong>Hybrid Mode Active</strong><br>
        BM25 + Dense Retrieval + CrossEncoder Reranking
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(124,92,252,0.08);border:1px solid rgba(124,92,252,0.2);
             border-radius:8px;padding:0.6rem 0.8rem;font-size:0.78rem;color:#a78bfa;">
        🧠 <strong>AI Knowledge Mode</strong><br>
        Upload a PDF to enable document search
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────── #
#  Main area
# ─────────────────────────────────────────────────────────────────────────── #

# ── App header ──
st.markdown("""
<div class="app-title">
  <h1>🔍 Hybrid Semantic Search Engine</h1>
  <p>AI-Powered Knowledge Retrieval and Discovery Platform</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")  # spacer

# ── Search bar ──
col_search, col_btn = st.columns([5, 1])
with col_search:
    query_input = st.text_input(
        label="search_query",
        placeholder="Enter your search query…  (e.g. 'What is deep learning?')",
        value=st.session_state.last_query,
        label_visibility="collapsed",
        key="search_input",
    )
with col_btn:
    search_clicked = st.button("Search 🔍", use_container_width=True)

st.markdown("")  # spacer

# ── Execute search ──
if search_clicked and query_input.strip():
    st.session_state.last_query = query_input.strip()

    index_id = (
        st.session_state.active_document["index_id"]
        if st.session_state.active_document
        else None
    )

    with st.spinner("⚡ Running hybrid retrieval pipeline…"):
        try:
            payload = {
                "query": query_input.strip(),
                "top_k": 5,
                "index_id": index_id,
            }
            res = requests.post(
                f"{API_URL}/query", json=payload, timeout=120
            )

            if res.status_code == 200:
                st.session_state.last_result = res.json()
            else:
                st.error(f"Search failed: {res.text}")

        except Exception as e:
            st.error(f"Connection error: {e}")

# ─────────────────────────────────────────────────────────────────────────── #
#  Results display
# ─────────────────────────────────────────────────────────────────────────── #

result = st.session_state.last_result

if result:
    source_type = result.get("source_type", "ai")
    query_type  = result.get("query_type", "general")
    answer      = result.get("answer", "")
    source_label = result.get("source_label", "")
    coverage    = result.get("document_coverage", 0.0)
    confidence  = result.get("semantic_confidence", 0.0)
    confidence_label = result.get("confidence_label", "Very Low")
    evidence_strength = result.get("evidence_strength", "Very Weak")
    evidence_found = result.get("evidence_found", False)
    questions   = result.get("related_questions", [])
    analytics   = result.get("analytics")
    results     = result.get("results", [])

    # ── Query classification ──
    qtype_map = {
        "document": ("qtype-doc",     "📄 Document Query"),
        "general":  ("qtype-general", "🧠 General Knowledge Query"),
        "mixed":    ("qtype-mixed",   "📄🧠 Mixed Query"),
    }
    qtype_cls, qtype_label = qtype_map.get(query_type, ("qtype-general", "🧠 General Knowledge Query"))

    # ── Source badge ──
    badge_map = {
        "doc":      ("badge-doc",     source_label),
        "ai":       ("badge-ai",      source_label),
        "hybrid":   ("badge-hybrid",  source_label),
        "fallback": ("badge-fallback", source_label),
    }
    badge_cls, badge_text = badge_map.get(source_type, ("badge-ai", source_label))

    st.markdown(f"""
    <div style="margin-bottom:0.5rem;">
      <span class="qtype-badge {qtype_cls}">{qtype_label}</span>
    </div>
    <div class="badge {badge_cls}">{badge_text}</div>
    """, unsafe_allow_html=True)

    # ── Two column layout: Answer + Analytics ──
    col_left, col_right = st.columns([3, 2], gap="medium")

    with col_left:
        # Answer card
        st.markdown(f"""
        <div class="answer-card">
          <div class="section-header" style="margin-bottom:0.7rem;">💬 Answer</div>
          <p>{answer.replace(chr(10), '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Coverage + Confidence bars
        if st.session_state.active_document:
            st.markdown('<div class="search-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📊 Semantic Analysis</div>', unsafe_allow_html=True)
            
            # Coverage bar logic
            if evidence_found:
                render_metric_bar("Document Coverage", coverage)
            else:
                render_metric_bar("Document Coverage", 0.0, custom_strength="No Evidence")
                
            render_metric_bar("Semantic Confidence", confidence, custom_strength=confidence_label)
            
            st.markdown(f"""
            <div class="metric-label" style="margin-top: 0.5rem;">
              <span>Evidence Strength</span>
              <span>{evidence_strength}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Related Questions
        if questions:
            st.markdown('<div class="search-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">❓ Related Questions</div>', unsafe_allow_html=True)
            for q in questions:
                st.markdown(f'<div class="rq-item">→ {q}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Retrieval Analytics
        if analytics:
            st.markdown('<div class="search-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚙️ Retrieval Analytics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="analytics-grid">
              <div class="analytics-item">
                <div class="analytics-value">{analytics['total_chunks']}</div>
                <div class="analytics-label">Total Chunks</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics['bm25_matches']}</div>
                <div class="analytics-label">BM25 Matches</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics['semantic_matches']}</div>
                <div class="analytics-label">Dense Matches</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics['merged_candidates']}</div>
                <div class="analytics-label">Merged</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics['final_results']}</div>
                <div class="analytics-label">Reranked</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value" style="font-size:1.1rem;">{analytics['response_time_ms']:.0f}ms</div>
                <div class="analytics-label">Response Time</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Pipeline visualization
        if st.session_state.active_document:
            st.markdown('<div class="search-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔄 Retrieval Pipeline</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.8rem;color:#8892a4;line-height:2;">
              <span style="color:#93c5fd;">Query</span><br>
              &nbsp;&nbsp;↓<br>
              <span style="color:#4ade80;">BM25 Sparse Retrieval</span><br>
              &nbsp;&nbsp;↓<br>
              <span style="color:#a78bfa;">Dense FAISS Retrieval</span><br>
              &nbsp;&nbsp;↓<br>
              <span style="color:#fcd34d;">Merge + Deduplicate</span><br>
              &nbsp;&nbsp;↓<br>
              <span style="color:#fca5a5;">CrossEncoder Reranking</span><br>
              &nbsp;&nbsp;↓<br>
              <span style="color:#93c5fd;">Final Results</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Evidence Explorer ──
    if results:
        st.markdown("")
        with st.expander("▼ View Evidence  —  Retrieval Transparency", expanded=False):
            st.markdown(
                "<div style='font-size:0.8rem;color:#8892a4;margin-bottom:1rem;'>"
                "Showing retrieved document chunks with their individual BM25, Semantic, and Rerank scores."
                "</div>",
                unsafe_allow_html=True,
            )
            for i, r in enumerate(results):
                bm25_str  = f"{r.get('bm25_score', 0):.3f}"
                faiss_str = f"{r.get('score', 0):.4f}"
                rerank_str = f"{r.get('rerank_score', 0):.3f}"
                text_preview = r["text"][:400] + ("…" if len(r["text"]) > 400 else "")

                st.markdown(f"""
                <div class="evidence-item">
                  <div class="evidence-meta">
                    <span class="evidence-tag tag-page">📄 Page {r['page_number']}</span>
                    <span class="evidence-tag tag-chunk">Chunk #{r['chunk_index']}</span>
                    <span class="evidence-tag tag-bm25">BM25: {bm25_str}</span>
                    <span class="evidence-tag tag-faiss">Semantic: {faiss_str}</span>
                    <span class="evidence-tag tag-rerank">Rerank: {rerank_str}</span>
                  </div>
                  <div class="evidence-text">{text_preview}</div>
                </div>
                """, unsafe_allow_html=True)

# ── Empty state ──
elif not result:
    if st.session_state.active_document:
        mode_text = "Hybrid Search Mode — BM25 + Dense Retrieval + CrossEncoder Reranking active"
        mode_color = "#4f8ef7"
    else:
        mode_text = "AI Knowledge Mode — Upload a PDF to enable document-grounded search"
        mode_color = "#7c5cfc"

    st.markdown(f"""
    <div style="text-align:center;padding:4rem 2rem;">
      <div style="font-size:3.5rem;margin-bottom:1rem;">🔍</div>
      <div style="font-size:1.3rem;font-weight:600;color:#f0f4ff;margin-bottom:0.5rem;">
        Start Searching
      </div>
      <div style="font-size:0.9rem;color:#8892a4;max-width:500px;margin:0 auto 1.5rem;">
        Enter a query above to discover insights from your documents or AI knowledge base.
      </div>
      <div style="display:inline-block;background:rgba(79,142,247,0.08);
           border:1px solid rgba(79,142,247,0.2);border-radius:8px;
           padding:0.5rem 1rem;font-size:0.8rem;color:{mode_color};">
        ⚡ {mode_text}
      </div>
    </div>
    """, unsafe_allow_html=True)