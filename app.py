import os
import sys
import base64
import tempfile
import numpy as np
import streamlit as st
from pypdf import PdfReader
import io

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── STYLING ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; color: #e2e8f0; }

    .user-bubble {
        background: #1e3a5f;
        border-left: 3px solid #2563eb;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .bot-bubble {
        background: #1a1a2e;
        border-left: 3px solid #10b981;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .source-pill {
        display: inline-block;
        background: #111827;
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px;
        color: #9ca3af;
        margin: 2px;
    }
    .status-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 8px 0;
        font-size: 13px;
    }
    .stTextInput input {
        background: #1a1a2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
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
    #MainMenu, footer, header { visibility: hidden; }
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
    embed_question, find_relevant_chunks,
    build_prompt, get_answer
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────

defaults = {
    "chat_history": [],
    "pdf_bytes": None,
    "pdf_name": None,
    "doc_index": None,
    "doc_chunks": None,
    "doc_ready": False,
    "chunk_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── HELPERS ───────────────────────────────────────────────────────────────────

def pdf_to_base64_url(pdf_bytes):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def get_page_count(pdf_bytes):
    """Get page count without crashing if it fails."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception:
        return 0


def process_pdf(uploaded_file):
    """
    Extract → Chunk → Embed → Store in memory.
    Nothing written to disk permanently.
    Everything lives in session state only.
    Disappears when session ends.
    """
    import faiss

    # Write to temp file so PyPDF can read it
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Step 1 — Extract text
    with st.spinner("📖 Extracting text from PDF..."):
        pages = extract_text_from_pdf(tmp_path)
    os.unlink(tmp_path)

    if not pages:
        st.error(
            "Could not extract text from this PDF. "
            "Make sure it's not a scanned image PDF."
        )
        return False

    # Step 2 — Chunk
    with st.spinner("✂️ Chunking document..."):
        chunks = chunk_pages(
            pages,
            chunk_size=500,
            overlap=50
        )

    # Step 3 — Embed via Bedrock Titan
    with st.spinner(
        f"🔮 Embedding {len(chunks)} chunks via "
        f"AWS Bedrock (~{max(1, len(chunks)//10)} sec)..."
    ):
        embeddings = embed_chunks(chunks)

    # Step 4 — Build FAISS index in memory
    with st.spinner("🗄️ Building search index..."):
        dimension = len(embeddings[0])
        vectors = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

    # Persist in session state
    st.session_state.doc_index = index
    st.session_state.doc_chunks = chunks
    st.session_state.doc_ready = True
    st.session_state.chunk_count = len(chunks)

    return True


def run_rag(question):
    """
    Full RAG pipeline:
    Embed question → Search FAISS → Build prompt → Claude answer
    """
    if not st.session_state.doc_ready:
        return (
            "Please upload and process a document first.",
            []
        )

    q_vec = embed_question(question)

    chunks = find_relevant_chunks(
        q_vec,
        st.session_state.doc_index,
        st.session_state.doc_chunks,
        top_k=3
    )

    if not chunks:
        return "No relevant content found.", []

    prompt = build_prompt(question, chunks)
    answer = get_answer(prompt)

    return answer, chunks


# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="
    background: linear-gradient(90deg, #1e3a5f, #111827);
    padding: 14px 22px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid #1f2937;
">
    <span style="font-size:20px; font-weight:700; color:#e2e8f0;">
        📄 Document Assistant
    </span>
    <span style="font-size:12px; color:#6b7280; margin-left:12px;">
        Upload any PDF · Ask anything · AWS Bedrock + RAG
    </span>
</div>
""", unsafe_allow_html=True)

# ── TWO COLUMN LAYOUT ─────────────────────────────────────────────────────────

left, right = st.columns([1, 1.1], gap="large")

# ════════════════════════════════════════
# LEFT — CHAT
# ════════════════════════════════════════

with left:

    # ── Document status bar ──
    if st.session_state.doc_ready:
        st.markdown(f"""
        <div class="status-box">
            <span style="color:#10b981; font-weight:600;">
                ✅ Document ready
            </span>
            <span style="color:#6b7280; margin-left:8px;
                font-size:12px;">
                {st.session_state.pdf_name}
            </span>
            <br>
            <span style="color:#4b5563; font-size:11px;">
                {st.session_state.chunk_count} chunks indexed
                · Ask anything below
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-box">
            <span style="color:#f59e0b; font-weight:600;">
                ⏳ No document loaded
            </span>
            <br>
            <span style="color:#4b5563; font-size:12px;">
                Upload a PDF on the right and click Process
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Chat messages ──
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 48px 16px;
            border: 1px dashed #1f2937;
            border-radius: 12px;
            color: #374151;
        ">
            <div style="font-size:40px;">💬</div>
            <div style="
                font-size:14px;
                color:#6b7280;
                margin-top:12px;
            ">
                Upload a document and start asking questions
            </div>
            <div style="
                font-size:12px;
                color:#374151;
                margin-top:16px;
                line-height:2;
            ">
                Works with any PDF —<br>
                Financial docs · Scriptures ·
                Research papers · Books · Contracts
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    <div style="
                        font-size:10px;
                        color:#60a5fa;
                        font-weight:700;
                        margin-bottom:5px;
                        letter-spacing:1px;
                    ">YOU</div>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-bubble">
                    <div style="
                        font-size:10px;
                        color:#34d399;
                        font-weight:700;
                        margin-bottom:5px;
                        letter-spacing:1px;
                    ">ASSISTANT</div>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)

                # Source pills below each answer
                if msg.get("sources"):
                    pills = ""
                    for s in msg["sources"]:
                        score = s["distance"]
                        rel = (
                            "🟢" if score < 0.8
                            else "🟡" if score < 1.2
                            else "🔴"
                        )
                        name = (
                            s["source"][:25] + "..."
                            if len(s["source"]) > 25
                            else s["source"]
                        )
                        pills += f"""
                        <span class="source-pill">
                            {rel} {name} · p{s['page']}
                        </span>"""
                    st.markdown(
                        f"<div style='margin:4px 0 12px 0'>"
                        f"{pills}</div>",
                        unsafe_allow_html=True
                    )

    st.markdown("---")

    # ── Question input ──
    question = st.text_input(
        "question",
        placeholder="Ask anything about your document...",
        key="q_input",
        label_visibility="collapsed",
        disabled=not st.session_state.doc_ready
    )

    col_ask, col_clear = st.columns([4, 1])

    with col_ask:
        ask_clicked = st.button(
            "🔍 Ask",
            key="ask_btn",
            disabled=not st.session_state.doc_ready
        )
    with col_clear:
        if st.button("🗑️", key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Handle ask ──
    if ask_clicked and question.strip():
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Thinking..."):
            try:
                answer, sources = run_rag(question)
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

# ════════════════════════════════════════
# RIGHT — PDF UPLOAD + VIEWER
# ════════════════════════════════════════

with right:
    st.markdown("""
    <div style="
        font-size:15px;
        font-weight:700;
        color:#e2e8f0;
        margin-bottom:12px;
    ">
        📂 Upload Document
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ──
    uploaded = st.file_uploader(
        "Upload any PDF",
        type=["pdf"],
        key="uploader",
        label_visibility="collapsed"
    )

    if uploaded:
        # Reset state when new file is uploaded
        if uploaded.name != st.session_state.pdf_name:
            st.session_state.pdf_name = uploaded.name
            st.session_state.pdf_bytes = uploaded.getvalue()
            st.session_state.doc_ready = False
            st.session_state.doc_index = None
            st.session_state.doc_chunks = None
            st.session_state.chat_history = []

        # ── Process button ──
        if not st.session_state.doc_ready:
            process_clicked = st.button(
                "⚡ Process Document",
                key="process_btn",
                use_container_width=True
            )
            if process_clicked:
                success = process_pdf(uploaded)
                if success:
                    st.success(
                        f"✅ Ready — "
                        f"{st.session_state.chunk_count} "
                        f"chunks indexed. Ask questions on the left."
                    )
                    st.rerun()
        else:
            st.success(
                f"✅ {st.session_state.chunk_count} chunks ready"
            )

        # ── PDF Preview ──
        if st.session_state.pdf_bytes:
            st.markdown(
                "<div style='margin-top:12px;'></div>",
                unsafe_allow_html=True
            )

            file_size_mb = len(
                st.session_state.pdf_bytes
            ) / (1024 * 1024)

            # Small files — inline preview
            if file_size_mb < 2:
                try:
                    pdf_url = pdf_to_base64_url(
                        st.session_state.pdf_bytes
                    )
                    st.markdown(
                        f'<iframe src="{pdf_url}" '
                        f'width="100%" height="650" '
                        f'style="border:1px solid #1f2937;'
                        f'border-radius:8px;'
                        f'background:#111827;">'
                        f'</iframe>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.info("Preview unavailable.")

            # Large files — info card
            else:
                page_count = get_page_count(
                    st.session_state.pdf_bytes
                )
                chunks_text = (
                    f"<div style='margin-top:12px;"
                    f"font-size:12px; color:#10b981;'>"
                    f"✅ {st.session_state.chunk_count} "
                    f"chunks indexed and searchable</div>"
                    if st.session_state.doc_ready else ""
                )

                st.markdown(f"""
                <div style="
                    background: #111827;
                    border: 1px solid #1f2937;
                    border-radius: 12px;
                    padding: 40px 24px;
                    text-align: center;
                    margin-top: 8px;
                ">
                    <div style="font-size:52px;">📄</div>
                    <div style="
                        font-size:15px;
                        font-weight:700;
                        color:#e2e8f0;
                        margin-top:14px;
                        word-break: break-all;
                    ">
                        {st.session_state.pdf_name}
                    </div>
                    <div style="
                        font-size:13px;
                        color:#6b7280;
                        margin-top:8px;
                    ">
                        {file_size_mb:.1f} MB
                        {" · " + str(page_count) +
                        " pages" if page_count else ""}
                    </div>
                    <div style="
                        background: #1f2937;
                        border-radius: 8px;
                        padding: 10px 16px;
                        margin-top: 16px;
                        font-size: 12px;
                        color: #9ca3af;
                        line-height: 1.6;
                    ">
                        File too large for browser preview.<br>
                        Document is fully indexed
                        and ready for questions.
                    </div>
                    {chunks_text}
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label="⬇️ Download to view locally",
                    data=st.session_state.pdf_bytes,
                    file_name=st.session_state.pdf_name,
                    mime="application/pdf",
                    use_container_width=True
                )

    else:
        # Empty state — no file uploaded yet
        st.markdown("""
        <div style="
            text-align: center;
            padding: 80px 20px;
            border: 2px dashed #1f2937;
            border-radius: 12px;
            color: #374151;
            margin-top: 8px;
        ">
            <div style="font-size:52px;">📄</div>
            <div style="
                font-size:15px;
                color:#6b7280;
                margin-top:14px;
                font-weight:600;
            ">
                Drop any PDF here
            </div>
            <div style="
                font-size:12px;
                color:#374151;
                margin-top:10px;
                line-height:2;
            ">
                Financial documents · Scriptures<br>
                Research papers · Books · Contracts<br>
                Any PDF works
            </div>
        </div>
        """, unsafe_allow_html=True)