import os
import sys
import json
import base64
import tempfile
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Financial Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── STYLING ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; color: #e2e8f0; }

    /* Fixed right panel */
    .pdf-panel {
        position: sticky;
        top: 0;
        height: 100vh;
        overflow: hidden;
    }

    /* Scrollable left panel */
    .chat-panel {
        height: 100vh;
        overflow-y: auto;
    }

    /* Chat bubbles */
    .user-bubble {
        background: #1e3a5f;
        border-left: 3px solid #2563eb;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0;
        color: #e2e8f0;
    }
    .bot-bubble {
        background: #1a1a2e;
        border-left: 3px solid #10b981;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0;
        color: #e2e8f0;
    }

    /* Source pills */
    .source-pill {
        display: inline-block;
        background: #111827;
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        color: #9ca3af;
        margin: 3px 2px;
    }

    /* Input styling */
    .stTextInput input {
        background: #1a1a2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    /* Button */
    .stButton button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stButton button:hover {
        background: #1d4ed8 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        color: #e2e8f0 !important;
    }

    /* Selectbox */
    .stSelectbox div {
        background: #1a1a2e;
        color: #e2e8f0;
        border-color: #374151;
    }

    /* Metrics */
    .stMetric {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #374151;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0e0e0e; }
    ::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── IMPORTS ───────────────────────────────────────────────────────────────────

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import extract_text_from_pdf, chunk_pages, embed_chunks
from retriever import (
    load_vector_store, embed_question,
    find_relevant_chunks, build_prompt, get_answer
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────

defaults = {
    "chat_history": [],
    "pdf_bytes": None,
    "pdf_name": None,
    "temp_index": None,
    "temp_chunks": None,
    "active_pdf_bytes": None,
    "active_pdf_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CACHE VECTOR STORE ────────────────────────────────────────────────────────
# @st.cache_resource loads once and reuses
# Fixes the "loading 4 times" problem

@st.cache_resource
def get_permanent_index():
    return load_vector_store()

# ── HELPERS ───────────────────────────────────────────────────────────────────

def pdf_to_base64_url(pdf_bytes):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def process_pdf_upload(uploaded_file):
    """Embed uploaded PDF into temporary in-memory index."""
    import faiss

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("📖 Reading document..."):
        pages = extract_text_from_pdf(tmp_path)
        chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    os.unlink(tmp_path)

    with st.spinner(
        f"🔮 Embedding {len(chunks)} chunks via Bedrock..."
    ):
        embeddings = embed_chunks(chunks)

    dimension = len(embeddings[0])
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index, chunks


def run_rag(question, use_preloaded, use_uploaded):
    """Full RAG pipeline — returns answer and sources."""
    q_vec = embed_question(question)
    all_chunks = []

    if use_preloaded:
        perm_index, perm_chunks = get_permanent_index()
        if perm_index:
            all_chunks += find_relevant_chunks(
                q_vec, perm_index, perm_chunks, top_k=3
            )

    if use_uploaded and st.session_state.temp_index:
        all_chunks += find_relevant_chunks(
            q_vec,
            st.session_state.temp_index,
            st.session_state.temp_chunks,
            top_k=3
        )

    if not all_chunks:
        return "No documents loaded. Please upload a PDF or run ingest.py first.", []

    all_chunks.sort(key=lambda x: x["distance"])
    top_chunks = all_chunks[:3]
    prompt = build_prompt(question, top_chunks)
    answer = get_answer(prompt)
    return answer, top_chunks


# ── LAYOUT ────────────────────────────────────────────────────────────────────

# App title
st.markdown("""
<div style="
    background: linear-gradient(90deg,#1e3a5f,#111827);
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 16px;
    border: 1px solid #1f2937;
">
    <span style="font-size:20px; font-weight:700; color:#e2e8f0;">
        📄 Financial Document Assistant
    </span>
    <span style="
        font-size:12px;
        color:#6b7280;
        margin-left:12px;
    ">
        AWS Bedrock · RAG · Claude
    </span>
</div>
""", unsafe_allow_html=True)

# Two columns — left chat, right PDF viewer
left, right = st.columns([1, 1.1], gap="large")

# ════════════════════════════════════════
# LEFT — CHAT
# ════════════════════════════════════════

with left:

    # ── Document scope ──
    scope = st.radio(
        "Search in:",
        ["All documents", "Preloaded only", "Uploaded only"],
        horizontal=True,
        key="scope"
    )

    st.markdown("---")

    # ── Chat history ──
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="
            text-align:center;
            color:#374151;
            padding:48px 16px;
            border:1px dashed #1f2937;
            border-radius:12px;
        ">
            <div style="font-size:36px">💬</div>
            <div style="
                font-size:14px;
                margin-top:10px;
                color:#6b7280;
            ">
                Start by asking a question<br>about your documents
            </div>
            <div style="
                font-size:12px;
                margin-top:16px;
                color:#374151;
                line-height:1.8;
            ">
                "What is this document about?"<br>
                "What are the main risk factors?"<br>
                "What is the Digital Upside Return?"<br>
                "What are the tax benefits?"
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    <div style="
                        font-size:11px;
                        color:#60a5fa;
                        margin-bottom:4px;
                        font-weight:600;
                    ">YOU</div>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-bubble">
                    <div style="
                        font-size:11px;
                        color:#34d399;
                        margin-bottom:4px;
                        font-weight:600;
                    ">ASSISTANT</div>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)

                # Sources
                if msg.get("sources"):
                    src_html = ""
                    for s in msg["sources"]:
                        score = s["distance"]
                        relevance = (
                            "🟢 High" if score < 0.8
                            else "🟡 Medium" if score < 1.2
                            else "🔴 Low"
                        )
                        name = s["source"][:35] + "..." \
                            if len(s["source"]) > 35 \
                            else s["source"]
                        src_html += f"""
                        <span class="source-pill">
                            {relevance} · {name} · p{s['page']}
                        </span>"""
                    st.markdown(
                        f"<div style='margin:6px 0'>"
                        f"{src_html}</div>",
                        unsafe_allow_html=True
                    )

    st.markdown("---")

    # ── Input area ──
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What are the main risk factors?",
        key="q_input",
        label_visibility="collapsed"
    )

    col_ask, col_clear = st.columns([3, 1])

    with col_ask:
        ask = st.button("🔍 Ask", key="ask_btn")

    with col_clear:
        clear = st.button("🗑️", key="clear_btn")

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if ask and question.strip():
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        use_pre = scope in ["All documents", "Preloaded only"]
        use_up = scope in ["All documents", "Uploaded only"]

        with st.spinner("Thinking..."):
            try:
                answer, sources = run_rag(
                    question, use_pre, use_up
                )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}",
                    "sources": []
                })
        st.rerun()

    # ── Status bar ──
    st.markdown("---")
    perm_index, perm_chunks = get_permanent_index()
    doc_count = len(set(
        c["source"] for c in perm_chunks
    )) if perm_chunks else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("📚 Docs", doc_count)
    m2.metric("🧩 Chunks", len(perm_chunks) if perm_chunks else 0)
    m3.metric(
        "⬆️ Uploaded",
        "✅" if st.session_state.temp_index else "—"
    )

# ════════════════════════════════════════
# RIGHT — PDF VIEWER (sticky)
# ════════════════════════════════════════

with right:
    st.markdown("""
    <div style="
        font-size:15px;
        font-weight:700;
        color:#e2e8f0;
        margin-bottom:12px;
    ">
        📂 Document Viewer
    </div>
    """, unsafe_allow_html=True)

    tab_up, tab_pre = st.tabs([
        "⬆️ Upload", "📚 Preloaded"
    ])

    # ── Upload tab ──
    with tab_up:
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="uploader",
            label_visibility="collapsed"
        )

        if uploaded:
            if uploaded.name != st.session_state.pdf_name:
                st.session_state.pdf_name = uploaded.name
                st.session_state.pdf_bytes = uploaded.getvalue()
                idx, cks = process_pdf_upload(uploaded)
                st.session_state.temp_index = idx
                st.session_state.temp_chunks = cks
                st.session_state.active_pdf_bytes = \
                    st.session_state.pdf_bytes
                st.session_state.active_pdf_name = uploaded.name
                st.success(
                    f"✅ {len(cks)} chunks ready — "
                    f"switch to 'Uploaded only' to query this doc"
                )

            if st.session_state.pdf_bytes:
                try:
                    pdf_url = pdf_to_base64_url(
                        st.session_state.pdf_bytes
                    )
                    st.markdown(
                        f'<iframe src="{pdf_url}" '
                        f'width="100%" height="680" '
                        f'style="border:1px solid #1f2937;'
                        f'border-radius:8px;"></iframe>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.info(
                        "📄 PDF uploaded and indexed. "
                        "Preview not available for large files. "
                        "You can now ask questions about it."
                    )

    # ── Preloaded tab ──
    with tab_pre:
        docs = []
        if os.path.exists("data/"):
            docs = [
                f for f in os.listdir("data/")
                if f.endswith(".pdf")
            ]

        if docs:
            selected = st.selectbox(
                "Select document",
                docs,
                key="doc_select",
                label_visibility="collapsed"
            )

            if selected:
                path = os.path.join("data/", selected)
                with open(path, "rb") as f:
                    b = f.read()

                try:
                    url = pdf_to_base64_url(b)
                    st.markdown(
                        f'<iframe src="{url}" '
                        f'width="100%" height="680" '
                        f'style="border:1px solid #1f2937;'
                        f'border-radius:8px;"></iframe>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.info(
                        f"📄 {selected} is loaded and ready "
                        f"for questions. Preview unavailable "
                        f"for large files."
                    )
        else:
            st.info(
                "No preloaded documents. "
                "Add PDFs to data/ and run ingest.py."
            )
