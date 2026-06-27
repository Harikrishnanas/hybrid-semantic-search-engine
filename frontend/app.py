"""
Multi-Document Hybrid Semantic Search Engine — Streamlit Frontend
=================================================================
Professional semantic search engine UI (not a chatbot).
Supports PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON, HTML, MD
"""

import os
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────── #
#  Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="Multi-Document Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ─────────────────────────────────────────────────────────────────────────── #
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────── #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:    #0a0e1a;
    --bg-card:       #111828;
    --bg-card2:      #161d2e;
    --accent:        #4f8ef7;
    --accent2:       #7c5cfc;
    --accent-green:  #22c55e;
    --accent-orange: #f59e0b;
    --accent-red:    #ef4444;
    --accent-teal:   #14b8a6;
    --text-primary:  #e8eeff;
    --text-muted:    #7a8499;
    --border:        #1e2640;
    --gradient:      linear-gradient(135deg, #4f8ef7 0%, #7c5cfc 100%);
    --gradient2:     linear-gradient(135deg, #14b8a6 0%, #4f8ef7 100%);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background-color: var(--bg-primary) !important;
}
#MainMenu, footer, .stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Title banner ── */
.app-title { text-align: center; padding: 1.5rem 0 0.5rem; }
.app-title h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-title p { color: var(--text-muted); font-size: 0.92rem; margin: 0.3rem 0 0; }

/* ── Search bar ── */
.stTextInput > div > div > input {
    background: var(--bg-card2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 1.05rem !important;
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
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── Cards ── */
.search-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, #111828 0%, #141228 100%);
    border: 1px solid #1e2640;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
}
.answer-card p { color: var(--text-primary); line-height: 1.8; margin: 0; }

/* ── Source badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.9rem; border-radius: 20px; font-size: 0.82rem;
    font-weight: 600; margin-bottom: 0.8rem;
}
.badge-doc    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-ai     { background: rgba(124,92,252,0.15); color: #a78bfa; border: 1px solid rgba(124,92,252,0.3); }
.badge-hybrid { background: rgba(79,142,247,0.15); color: #93c5fd; border: 1px solid rgba(79,142,247,0.3); }

/* ── Query type badge ── */
.qtype-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.25rem 0.75rem; border-radius: 6px;
    font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;
}
.qtype-doc     { background: rgba(34,197,94,0.1);  color: #4ade80; }
.qtype-general { background: rgba(124,92,252,0.1); color: #a78bfa; }
.qtype-mixed   { background: rgba(79,142,247,0.1); color: #93c5fd; }

/* ── Progress bars ── */
.metric-bar-wrap { margin-bottom: 1rem; }
.metric-label {
    display: flex; justify-content: space-between;
    margin-bottom: 0.3rem; font-size: 0.85rem; color: var(--text-muted);
}
.metric-label span:last-child { font-weight: 600; color: var(--text-primary); }
.metric-bar-bg { background: var(--border); border-radius: 6px; height: 8px; overflow: hidden; }
.metric-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; }
.bar-strong  { background: linear-gradient(90deg, #22c55e, #4ade80); }
.bar-medium  { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
.bar-weak    { background: linear-gradient(90deg, #ef4444, #f87171); }
.bar-accent  { background: var(--gradient); }
.bar-teal    { background: var(--gradient2); }

/* ── Analytics grid ── */
.analytics-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 0.7rem; margin-top: 0.5rem;
}
.analytics-item {
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.7rem 0.9rem; text-align: center;
}
.analytics-value { font-size: 1.35rem; font-weight: 700; color: var(--accent); line-height: 1; }
.analytics-label { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.4px; }

/* ── Evidence items ── */
.evidence-item {
    background: var(--bg-card2);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}
.evidence-meta { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
.evidence-tag { font-size: 0.72rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 4px; }
.tag-source  { background: rgba(20,184,166,0.15); color: #5eead4; }
.tag-type    { background: rgba(245,158,11,0.15);  color: #fcd34d; }
.tag-page    { background: rgba(34,197,94,0.15);   color: #4ade80; }
.tag-chunk   { background: rgba(245,158,11,0.15);  color: #fcd34d; }
.tag-bm25    { background: rgba(79,142,247,0.15);  color: #93c5fd; }
.tag-faiss   { background: rgba(124,92,252,0.15);  color: #a78bfa; }
.tag-rerank  { background: rgba(239,68,68,0.15);   color: #fca5a5; }
.evidence-text { color: var(--text-muted); font-size: 0.87rem; line-height: 1.65; }

/* ── Source attribution ── */
.source-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(79,142,247,0.1); border: 1px solid rgba(79,142,247,0.25);
    color: #93c5fd; border-radius: 20px;
    padding: 0.25rem 0.75rem; font-size: 0.8rem; font-weight: 500;
    margin: 0.2rem 0.2rem 0 0;
}

/* ── Heatmap ── */
.heatmap-item { margin-bottom: 0.6rem; }
.heatmap-label { display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.2rem; }
.heatmap-label span:first-child { color: var(--text-primary); }
.heatmap-label span:last-child  { color: var(--text-muted); font-weight: 600; }

/* ── Section headers ── */
.section-header {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem; margin-bottom: 0.9rem;
}

/* ── Doc info card ── */
.doc-stat-row { display: flex; justify-content: space-between; margin: 0.3rem 0; }
.doc-stat-label { font-size: 0.78rem; color: var(--text-muted); }
.doc-stat-value { font-size: 0.78rem; font-weight: 700; color: var(--text-primary); }

/* ── File uploader ── */
.stFileUploader {
    background: var(--bg-card2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── Progress spinner ── */
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

hr { border-color: var(--border) !important; }

.no-evidence-box {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.18);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #fca5a5;
    margin-top: 0.8rem;
    line-height: 1.5;
}

.strength-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 14px;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────── #
#  Session state
# ─────────────────────────────────────────────────────────────────────────── #

for key, default in {
    "active_index": None,       # { index_id, filenames, total_files, total_chunks, total_pages, document_types }
    "uploader_key": 0,
    "last_result": None,
    "last_query": "",
    "doc_intelligence": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────── #
#  Helpers
# ─────────────────────────────────────────────────────────────────────────── #

ALLOWED_TYPES = ["pdf", "docx", "txt", "csv", "xlsx", "xls", "pptx", "json", "html", "htm", "md"]

def render_metric_bar(label: str, value: float, bar_class: str = "bar-accent", strength_label: str = ""):
    width = min(max(value, 0), 100)
    display = strength_label if strength_label else f"{value:.0f}%"
    st.markdown(f"""
    <div class="metric-bar-wrap">
      <div class="metric-label">
        <span>{label}</span>
        <span>{value:.0f}% &nbsp;·&nbsp; {display}</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill {bar_class}" style="width:{width}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_heatmap_bar(label: str, pct: float):
    # 20 blocks max => 1 block per 5%
    blocks = int(pct / 5)
    blocks = min(max(blocks, 0), 20)
    bar_str = "█" * blocks
    
    st.markdown(f"""
    <div class="heatmap-item" style="margin-bottom:0.7rem;">
      <div style="font-size:0.85rem;color:#e8eeff;margin-bottom:0.1rem;">{label}</div>
      <div style="font-size:0.75rem;color:#93c5fd;letter-spacing:1px;font-family:monospace;line-height:1;">
        {bar_str} <span style="color:#b0b8cc;letter-spacing:0;font-family:'Inter',sans-serif;margin-left:4px;">{pct:.0f}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_source_dist_bar(source: str, pct: float):
    width = min(max(pct, 0), 100)
    cls = "bar-teal" if pct >= 40 else ("bar-accent" if pct >= 20 else "bar-medium")
    st.markdown(f"""
    <div class="heatmap-item">
      <div class="heatmap-label">
        <span style="font-size:0.8rem;">📄 {source}</span><span>{pct:.0f}%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill {cls}" style="width:{width}%;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def confidence_bar_class(label: str) -> str:
    return {
        "Very High": "bar-strong",
        "High":      "bar-strong",
        "Moderate":  "bar-medium",
        "Low":       "bar-weak",
        "Very Low":  "bar-weak",
    }.get(label, "bar-accent")


def coverage_bar_class(strength: str) -> str:
    return {
        "Strong Evidence":   "bar-strong",
        "Moderate Evidence": "bar-medium",
        "Weak Evidence":     "bar-weak",
        "No Evidence":       "bar-weak",
    }.get(strength, "bar-accent")


def strength_color(strength: str) -> str:
    return {
        "Strong Evidence":   "#4ade80",
        "Moderate Evidence": "#fcd34d",
        "Weak Evidence":     "#fb923c",
        "No Evidence":       "#ef4444",
    }.get(strength, "#8892a4")


# ─────────────────────────────────────────────────────────────────────────── #
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────── #

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
      <div style="font-size:1.25rem;font-weight:700;color:#e8eeff;">🔍 Semantic Search</div>
      <div style="font-size:0.75rem;color:#7a8499;margin-top:0.2rem;">Multi-Document Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📂 Document Upload</div>', unsafe_allow_html=True)

    if st.session_state.active_index is None:
        # ── File Upload ──
        st.markdown(f"""
        <div style="font-size:0.78rem;color:#7a8499;margin-bottom:0.5rem;">
        Supported: PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON, HTML, MD
        · Max {20} files
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            help="Upload up to 20 documents to enable document-grounded search",
            label_visibility="collapsed",
        )

        if uploaded_files:
            if len(uploaded_files) > 20:
                st.error(f"❌ Too many files. Maximum 20 allowed, you selected {len(uploaded_files)}.")
            else:
                st.markdown(f"<div style='font-size:0.8rem;color:#7a8499;margin-bottom:0.5rem;'>{len(uploaded_files)} file(s) selected</div>", unsafe_allow_html=True)
                if st.button("⚡ Process Documents", use_container_width=True):
                    with st.spinner("🔄 Uploading and indexing documents…"):
                        try:
                            # Build multipart files
                            files_payload = []
                            for uf in uploaded_files:
                                files_payload.append(
                                    ("files", (uf.name, uf.getvalue(), uf.type or "application/octet-stream"))
                                )
                            upload_res = requests.post(
                                f"{API_URL}/upload-files", files=files_payload, timeout=120
                            )

                            if upload_res.status_code == 201:
                                file_info = upload_res.json()
                                index_res = requests.post(
                                    f"{API_URL}/create-index",
                                    json={"file_paths": file_info["file_paths"]},
                                    timeout=600,
                                )

                                if index_res.status_code == 200:
                                    idx_info = index_res.json()
                                    st.session_state.active_index = {
                                        "index_id":      idx_info["index_id"],
                                        "filenames":     file_info["filenames"],
                                        "total_files":   idx_info["total_files"],
                                        "total_chunks":  idx_info["total_chunks"],
                                        "total_pages":   idx_info["total_pages"],
                                        "document_types": [],
                                    }

                                    # Fetch document intelligence
                                    di_res = requests.get(
                                        f"{API_URL}/document-intelligence/{idx_info['index_id']}",
                                        timeout=120,
                                    )
                                    if di_res.status_code == 200:
                                        di_data = di_res.json()
                                        st.session_state.doc_intelligence = di_data
                                        st.session_state.active_index["document_types"] = di_data.get("document_types", [])

                                    st.success(f"✅ {idx_info['total_files']} document(s) indexed!")
                                    st.rerun()
                                else:
                                    st.error(f"Indexing failed: {index_res.text}")
                            else:
                                st.error(f"Upload failed: {upload_res.text}")

                        except Exception as e:
                            st.error(f"Error: {e}")
    else:
        # ── Active Index info ──
        idx = st.session_state.active_index
        di = st.session_state.doc_intelligence

        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(79,142,247,0.07),rgba(124,92,252,0.07));
             border:1px solid rgba(79,142,247,0.18);border-radius:12px;padding:0.9rem 1rem;margin-bottom:1rem;">
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-weight:700;font-size:0.9rem;color:#e8eeff;margin-bottom:0.6rem;">
          📚 {idx['total_files']} Document(s) Active
        </div>
        <div class="doc-stat-row"><span class="doc-stat-label">Files</span><span class="doc-stat-value">{idx['total_files']}</span></div>
        <div class="doc-stat-row"><span class="doc-stat-label">Chunks</span><span class="doc-stat-value">{idx['total_chunks']}</span></div>
        <div class="doc-stat-row"><span class="doc-stat-label">Pages</span><span class="doc-stat-value">{idx['total_pages'] or '—'}</span></div>
        """, unsafe_allow_html=True)

        if idx.get("document_types"):
            types_str = " · ".join(idx["document_types"])
            st.markdown(f'<div class="doc-stat-row"><span class="doc-stat-label">Types</span><span class="doc-stat-value">{types_str}</span></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Files list ──
        if idx.get("filenames"):
            st.markdown('<div class="section-header">📄 Files Uploaded</div>', unsafe_allow_html=True)
            for fn in idx["filenames"]:
                ext = fn.rsplit(".", 1)[-1].upper() if "." in fn else "?"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;">
                  <span style="background:rgba(79,142,247,0.1);border:1px solid rgba(79,142,247,0.2);
                        color:#93c5fd;border-radius:4px;padding:0.1rem 0.4rem;
                        font-size:0.68rem;font-weight:700;">{ext}</span>
                  <span style="font-size:0.8rem;color:#b0b8cc;overflow:hidden;
                        white-space:nowrap;text-overflow:ellipsis;max-width:180px;">{fn}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Document Intelligence (Per Document) ──
        if idx.get("filenames"):
            st.markdown("---")
            st.markdown('<div class="section-header">🧠 Document Intelligence</div>', unsafe_allow_html=True)
            
            if di and di.get("documents"):
                for doc in di["documents"]:
                    doc_name = doc.get("filename", "Unknown")
                    ext = doc_name.rsplit(".", 1)[-1].upper() if "." in doc_name else "?"
                    st.markdown(f"""
                    <div style="margin-top:1rem;padding:0.8rem;background:rgba(23,30,50,0.5);border:1px solid rgba(255,255,255,0.05);border-radius:6px;">
                      <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
                        <span style="font-size:1.1rem;">📄</span>
                        <span style="font-weight:600;font-size:0.85rem;color:#e8eeff;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{doc_name}</span>
                      </div>
                    """, unsafe_allow_html=True)
                
                    if doc.get("summary"):
                        st.markdown("<div style='font-size:0.75rem;font-weight:600;color:#93c5fd;margin-bottom:0.3rem;'>Summary</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:0.8rem;color:#b0b8cc;line-height:1.5;margin-bottom:0.8rem;'>{doc.get('summary')}</div>", unsafe_allow_html=True)
                    
                    if doc.get("top_topics"):
                        st.markdown("<div style='font-size:0.75rem;font-weight:600;color:#93c5fd;margin-bottom:0.3rem;'>Main Topics</div>", unsafe_allow_html=True)
                        topics_html = "".join([f"<li style='font-size:0.8rem;color:#b0b8cc;'>{t}</li>" for t in doc.get("top_topics")])
                        st.markdown(f"<ul style='margin-bottom:0.8rem;padding-left:1.2rem;line-height:1.4;'>{topics_html}</ul>", unsafe_allow_html=True)
                        
                    if doc.get("coverage_heatmap"):
                        st.markdown("<div style='font-size:0.75rem;font-weight:600;color:#93c5fd;margin-bottom:0.3rem;'>Topic Presence</div>", unsafe_allow_html=True)
                        for item in doc.get("coverage_heatmap", []):
                            render_heatmap_bar(item["topic"], item["percentage"])
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#7a8499;font-size:0.85rem;padding:1rem;text-align:center;'>Generating document intelligence...</div>", unsafe_allow_html=True)

        # ── Remove button ──
        st.markdown("---")
        if st.button("🗑️ Remove All Documents", use_container_width=True):
            st.session_state.active_index = None
            st.session_state.doc_intelligence = None
            st.session_state.uploader_key += 1
            st.session_state.last_result = None
            st.rerun()

    # ── Mode indicator ──
    st.markdown("---")
    if st.session_state.active_index:
        st.markdown("""
        <div style="background:rgba(79,142,247,0.07);border:1px solid rgba(79,142,247,0.18);
             border-radius:8px;padding:0.6rem 0.8rem;font-size:0.77rem;color:#93c5fd;">
        ⚡ <strong>Hybrid Mode Active</strong><br>
        BM25 + Dense Retrieval + CrossEncoder
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(124,92,252,0.07);border:1px solid rgba(124,92,252,0.18);
             border-radius:8px;padding:0.6rem 0.8rem;font-size:0.77rem;color:#a78bfa;">
        🧠 <strong>AI Knowledge Mode</strong><br>
        Upload documents to enable document search
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────── #
#  Main area
# ─────────────────────────────────────────────────────────────────────────── #

st.markdown("""
<div class="app-title">
  <h1>🔍 Multi-Document Semantic Search</h1>
  <p>AI-Powered Knowledge Retrieval · BM25 + Dense + CrossEncoder · Multi-Format Documents</p>
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

st.markdown("")

# ── Execute search ──
if search_clicked and query_input.strip():
    st.session_state.last_query = query_input.strip()

    index_id = (
        st.session_state.active_index["index_id"]
        if st.session_state.active_index
        else None
    )

    with st.spinner("⚡ Running hybrid retrieval pipeline…"):
        try:
            payload = {
                "query": query_input.strip(),
                "top_k": 5,
                "index_id": index_id,
            }
            res = requests.post(f"{API_URL}/query", json=payload, timeout=180)

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
    source_type      = result.get("source_type", "ai")
    query_type       = result.get("query_type", "general")
    answer           = result.get("answer", "")
    source_label     = result.get("source_label", "")
    coverage         = result.get("document_coverage", 0.0)
    confidence       = result.get("semantic_confidence", 0.0)
    confidence_label = result.get("confidence_label", "Very Low")
    evidence_strength = result.get("evidence_strength", "No Evidence")
    evidence_found   = result.get("evidence_found", False)
    analytics        = result.get("analytics")
    results          = result.get("results", [])
    best_bm25        = result.get("best_bm25_score", 0.0)
    best_faiss       = result.get("best_faiss_score", 0.0)
    best_rerank      = result.get("best_rerank_score", 0.0)
    sources_used     = result.get("sources_used", [])
    evidence_dist    = result.get("evidence_distribution", {})

    # ── Query type badge ──
    qtype_map = {
        "document": ("qtype-doc",     "📄 Document Query"),
        "general":  ("qtype-general", "🧠 General Knowledge"),
        "mixed":    ("qtype-mixed",   "📄🧠 Mixed Query"),
    }
    qtype_cls, qtype_label = qtype_map.get(query_type, ("qtype-general", "🧠 General Knowledge"))

    # ── Source badge ──
    badge_map = {
        "doc":    ("badge-doc",    source_label),
        "ai":     ("badge-ai",     source_label),
        "hybrid": ("badge-hybrid", source_label),
    }
    badge_cls, badge_text = badge_map.get(source_type, ("badge-ai", source_label))

    st.markdown(f"""
    <div style="margin-bottom:0.5rem;">
      <span class="qtype-badge {qtype_cls}">{qtype_label}</span>
    </div>
    <div class="badge {badge_cls}">{badge_text}</div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ──
    col_left, col_right = st.columns([3, 2], gap="medium")

    with col_left:
        # ── Answer ──
        st.markdown(f"""
        <div class="answer-card">
          <div class="section-header" style="margin-bottom:0.7rem;">💬 Answer</div>
          <p>{answer.replace(chr(10), '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)

        # (Sources Used and Evidence Distribution sections removed per requirements)

        # ── Semantic Analysis ──
        if st.session_state.active_index:
            st.markdown('<div class="search-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📊 Semantic Analysis</div>', unsafe_allow_html=True)

            if evidence_found:
                render_metric_bar(
                    "Document Coverage", coverage,
                    bar_class=coverage_bar_class(evidence_strength),
                    strength_label=evidence_strength,
                )
                render_metric_bar(
                    "Semantic Confidence", confidence,
                    bar_class=confidence_bar_class(confidence_label),
                    strength_label=confidence_label,
                )
            else:
                render_metric_bar("Document Coverage", 0.0, bar_class="bar-weak", strength_label="No Evidence")
                render_metric_bar("Semantic Confidence", min(confidence, 10.0), bar_class="bar-weak", strength_label="Very Low")

            # Evidence strength row
            disp_strength = evidence_strength if evidence_found else "No Evidence"
            s_color = strength_color(disp_strength)
            st.markdown(f"""
            <div class="metric-label" style="margin-top:0.5rem;">
              <span>Evidence Strength</span>
              <span style="color:{s_color};font-weight:700;">{disp_strength}</span>
            </div>
            """, unsafe_allow_html=True)

            if not evidence_found and st.session_state.active_index:
                st.markdown("""
                <div class="no-evidence-box">
                  ⚠️ No relevant supporting evidence found in uploaded documents.<br>
                  Document Coverage: 0% — Answer generated using AI knowledge.
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # ── Retrieval Transparency ──
        if st.session_state.active_index:
            st.markdown('<div class="search-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔍 Retrieval Transparency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.85rem;line-height:2.1;">
              <div class="metric-label">
                <span style="color:#7a8499;">Best BM25 Score</span>
                <span style="color:#e8eeff;font-weight:700;font-family:monospace;">{best_bm25:.3f}</span>
              </div>
              <div class="metric-label">
                <span style="color:#7a8499;">Best Dense Similarity</span>
                <span style="color:#e8eeff;font-weight:700;font-family:monospace;">{best_faiss:.4f}</span>
              </div>
              <div class="metric-label">
                <span style="color:#7a8499;">Best Rerank Score</span>
                <span style="color:#e8eeff;font-weight:700;font-family:monospace;">{best_rerank:.4f}</span>
              </div>
              <div class="metric-label" style="margin-top:0.4rem;padding-top:0.4rem;border-top:1px solid #1e2640;">
                <span style="color:#7a8499;">Confidence</span>
                <span style="color:#93c5fd;font-size:0.78rem;">{confidence_label}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Retrieval Analytics ──
        if analytics and st.session_state.active_index:
            st.markdown('<div class="search-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚙️ Retrieval Analytics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="analytics-grid">
              <div class="analytics-item">
                <div class="analytics-value">{analytics.get('total_chunks', 0)}</div>
                <div class="analytics-label">Total Chunks</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics.get('bm25_matches', 0)}</div>
                <div class="analytics-label">BM25 Matches</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics.get('semantic_matches', 0)}</div>
                <div class="analytics-label">Dense Matches</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics.get('merged_candidates', 0)}</div>
                <div class="analytics-label">Merged</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics.get('final_results', 0)}</div>
                <div class="analytics-label">Reranked</div>
              </div>
              <div class="analytics-item">
                <div class="analytics-value">{analytics.get('response_time_ms', 0):.0f}<span style="font-size:0.65rem;">ms</span></div>
                <div class="analytics-label">Response Time</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Total Documents stat
            if st.session_state.active_index:
                total_docs = st.session_state.active_index.get("total_files", 1)
                st.markdown(f"""
                <div class="metric-label" style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid #1e2640;">
                  <span style="color:#7a8499;">Total Documents</span>
                  <span style="color:#e8eeff;font-weight:700;">{total_docs}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ── Evidence Chunks (only when real evidence exists) ──
    if results and evidence_found:
        st.markdown("<div style='margin-top:0.5rem;'>", unsafe_allow_html=True)
        with st.expander(f"📄 Supporting Evidence ({len(results)} chunk{'s' if len(results) != 1 else ''})", expanded=False):
            for i, r in enumerate(results):
                rerank = r.get("rerank_score", 0.0)
                bm25   = r.get("bm25_score", 0.0)
                faiss  = r.get("score", 0.0)
                src    = r.get("source_file", "Unknown")
                dtype  = r.get("document_type", "")
                page   = r.get("page_number")
                chunk  = r.get("chunk_index", i)
                text   = r.get("text", "")

                page_tag = f'<span class="evidence-tag tag-page">Page {page}</span>' if page else ""

                st.markdown(f"""
                <div class="evidence-item">
                  <div class="evidence-meta">
                    <span class="evidence-tag tag-source">📄 {src}</span>
                    {f'<span class="evidence-tag tag-type">{dtype}</span>' if dtype else ""}
                    {page_tag}
                    <span class="evidence-tag tag-chunk">Chunk #{chunk}</span>
                    <span class="evidence-tag tag-rerank">Rerank {rerank:.3f}</span>
                    <span class="evidence-tag tag-faiss">Dense {faiss:.3f}</span>
                    <span class="evidence-tag tag-bm25">BM25 {bm25:.2f}</span>
                  </div>
                  <div class="evidence-text">{text[:600]}{'…' if len(text) > 600 else ''}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif not result and not st.session_state.active_index:
    # Welcome state — no documents, no query
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;">
      <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
      <div style="font-size:1.1rem;color:#7a8499;font-weight:500;">
        Upload documents from the sidebar to get started,<br>
        or ask any question using AI knowledge.
      </div>
      <div style="margin-top:1.5rem;display:flex;justify-content:center;gap:0.5rem;flex-wrap:wrap;">
        <span class="source-chip">PDF</span>
        <span class="source-chip">DOCX</span>
        <span class="source-chip">TXT</span>
        <span class="source-chip">CSV</span>
        <span class="source-chip">XLSX</span>
        <span class="source-chip">PPTX</span>
        <span class="source-chip">JSON</span>
        <span class="source-chip">HTML</span>
        <span class="source-chip">MD</span>
      </div>
    </div>
    """, unsafe_allow_html=True)