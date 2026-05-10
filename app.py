import os
import sys
import json
import base64
import hashlib
import tempfile
import pickle
import numpy as np
import streamlit as st
from classifier import classify_document
from extractor import extract_document_data

st.set_page_config(page_title="Financial Document Assistant", page_icon="📄", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; color: #e2e8f0; }
    .user-bubble { background: #1e3a5f; border-left: 3px solid #2563eb; border-radius: 10px; padding: 12px 16px; margin: 8px 0; }
    .bot-bubble { background: #1a1a2e; border-left: 3px solid #10b981; border-radius: 10px; padding: 12px 16px; margin: 8px 0; }
    .source-pill { display: inline-block; background: #111827; border: 1px solid #374151; border-radius: 20px; padding: 3px 10px; font-size: 11px; color: #9ca3af; margin: 2px; }
    .stButton button { background: #2563eb !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; width: 100% !important; }
    .stTextInput input { background: #1a1a2e !important; color: #e2e8f0 !important; border: 1px solid #374151 !important; border-radius: 8px !important; }
    #MainMenu, footer, header { visibility: hidden; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0e0e0e; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever import find_relevant_chunks

DOCUMENT_TYPES = {
    "structured_note": {
        "icon": "📄", "label": "Structured Note / Prospectus", "color": "#60a5fa",
        "system_prompt": "You are an expert analyst for structured notes. TERMINOLOGY: 'Underlying'=Reference Asset, 'Protection'=Buffer%, 'Return'=Digital Upside Return. JSON requests: return ONLY valid JSON, no markdown fences.",
        "suggested_questions": ["Extract underlying, protection % and return as table", "What are the main risk factors?", "What is the payment at maturity formula?"]
    },
    "form_10k": {
        "icon": "📊", "label": "10-K Annual Report", "color": "#34d399",
        "system_prompt": "You are an expert analyst for SEC 10-K annual reports. TERMINOLOGY: 'Revenue'=Net Revenue, 'Profit'=Net Income, 'EPS'=Earnings Per Share. Always include fiscal year. JSON: clean JSON only.",
        "suggested_questions": ["What was total revenue and net income?", "Extract key financial metrics as a table", "What are the main business risks?"]
    },
    "form_10q": {
        "icon": "📋", "label": "10-Q Quarterly Report", "color": "#a78bfa",
        "system_prompt": "You are an expert analyst for SEC 10-Q quarterly reports. TERMINOLOGY: 'Revenue'=Net Revenue, 'Profit'=Net Income, 'YoY'=Year over year. Always specify quarter and year. JSON: clean JSON only.",
        "suggested_questions": ["What was revenue and profit this quarter?", "How did this quarter compare to last year?", "Extract quarterly financials as a table"]
    },
    "form_13f": {
        "icon": "📑", "label": "Form 13F — Holdings Report", "color": "#fb923c",
        "system_prompt": "You are an expert analyst for SEC Form 13F. TERMINOLOGY: 'Holdings'=Securities held, 'Market value'=value in thousands. Sort holdings by market value. JSON: use keys ticker, shares, market_value.",
        "suggested_questions": ["What are the top 10 holdings by value?", "Extract all holdings as a table", "What new positions were added?"]
    },
    "municipal_bond": {
        "icon": "🏛️", "label": "Municipal Bond / Official Statement", "color": "#f472b6",
        "system_prompt": "You are an expert analyst for municipal bonds. TERMINOLOGY: 'Yield'=Interest Rate/Coupon, 'Rating'=Credit Rating, 'Tax status'=Tax-exempt/Taxable. Always note tax status and include rating agency name.",
        "suggested_questions": ["What is the bond type, rate, and maturity?", "Extract key bond terms as a table", "What is the credit rating and tax status?"]
    },
    "other": {
        "icon": "📁", "label": "Financial Document", "color": "#94a3b8",
        "system_prompt": "You are an expert financial document analyst. Map user language to document terminology. Extract exact values. JSON requests: clean JSON only.",
        "suggested_questions": ["What is this document about?", "Extract the key financial terms as a table", "What are the main risks mentioned?"]
    },
}

defaults = {
    "chat_history": [], "pdf_bytes": None, "pdf_name": None,
    "doc_index": None, "doc_chunks": None, "doc_ready": False,
    "chunk_count": 0, "doc_type": None, "extracted_data": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_cache_path(pdf_bytes):
    return f"cache/{hashlib.md5(pdf_bytes).hexdigest()}.pkl"

def load_from_cache(pdf_bytes):
    path = get_cache_path(pdf_bytes)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_to_cache(pdf_bytes, data):
    os.makedirs("cache", exist_ok=True)
    with open(get_cache_path(pdf_bytes), "wb") as f:
        pickle.dump(data, f)

def pdf_to_base64_url(pdf_bytes):
    return f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode('utf-8')}"

def extract_text(pdf_path):
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(pdf_path), "markdown"
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"text": text, "page": i + 1})
        return pages, "pages"

def make_chunks(result, result_type, source_name, chunk_size=800, overlap=100):
    chunks = []
    if result_type == "markdown":
        lines = result.split('\n')
        current_chunk = ""
        current_page = 1
        for line in lines:
            if len(current_chunk) + len(line) > chunk_size:
                if current_chunk.strip():
                    chunks.append({"text": current_chunk.strip(), "page": current_page, "source": source_name, "chunk_id": len(chunks)})
                    current_chunk = current_chunk[-overlap:] + "\n" + line
                else:
                    current_chunk += "\n" + line
            else:
                current_chunk += "\n" + line
            if "<!-- Page" in line:
                try:
                    current_page = int(line.split("<!-- Page")[1].split("-->")[0].strip())
                except Exception:
                    pass
        if current_chunk.strip():
            chunks.append({"text": current_chunk.strip(), "page": current_page, "source": source_name, "chunk_id": len(chunks)})
    else:
        for page in result:
            text = page["text"]
            start = 0
            while start < len(text):
                chunk_text = text[start:start + chunk_size]
                if chunk_text.strip():
                    chunks.append({"text": chunk_text, "page": page["page"], "source": source_name, "chunk_id": len(chunks)})
                start += (chunk_size - overlap)
    return chunks

def embed_chunks(chunks):
    import boto3, time
    from concurrent.futures import ThreadPoolExecutor
    from dotenv import load_dotenv
    load_dotenv()
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    def embed_one(text, retries=3):
        for attempt in range(retries):
            try:
                r = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=json.dumps({"inputText": text}), contentType="application/json", accept="application/json")
                return json.loads(r["body"].read())["embedding"]
            except Exception as e:
                if "ThrottlingException" in str(e) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    embeddings = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(embed_one, c["text"]): i for i, c in enumerate(chunks)}
        for f in futures:
            embeddings[futures[f]] = f.result()
    return embeddings

def embed_question(question):
    import boto3
    from dotenv import load_dotenv
    load_dotenv()
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    r = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=json.dumps({"inputText": question}), contentType="application/json", accept="application/json")
    return np.array([json.loads(r["body"].read())["embedding"]]).astype("float32")

def process_pdf(uploaded_file, progress_bar):
    import faiss
    pdf_bytes = uploaded_file.getvalue()

    progress_bar.progress(0.05, text="Checking cache...")
    cached = load_from_cache(pdf_bytes)
    if cached:
        chunks = cached["chunks"]
        st.session_state.doc_index = cached["index"]
        st.session_state.doc_chunks = chunks
        st.session_state.doc_ready = True
        st.session_state.chunk_count = len(chunks)
        progress_bar.progress(0.7, text="🧠 Classifying...")
        st.session_state.doc_type = classify_document(chunks)
        progress_bar.progress(0.9, text="📊 Extracting data...")
        st.session_state.extracted_data = extract_document_data(chunks, st.session_state.doc_type)
        progress_bar.progress(1.0, text="✅ Done!")
        return True

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    progress_bar.progress(0.15, text="📖 Extracting text...")
    result, result_type = extract_text(tmp_path)
    os.unlink(tmp_path)

    progress_bar.progress(0.3, text="✂️ Chunking...")
    chunks = make_chunks(result, result_type, uploaded_file.name)

    if not chunks:
        st.error("Could not extract text from this PDF.")
        return False

    progress_bar.progress(0.5, text=f"🔮 Embedding {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    progress_bar.progress(0.85, text="🗄️ Building index...")
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(vectors)

    progress_bar.progress(0.9, text="💾 Caching...")
    save_to_cache(pdf_bytes, {"index": index, "chunks": chunks})

    st.session_state.doc_index = index
    st.session_state.doc_chunks = chunks
    st.session_state.doc_ready = True
    st.session_state.chunk_count = len(chunks)

    progress_bar.progress(0.94, text="🧠 Classifying...")
    st.session_state.doc_type = classify_document(chunks)

    progress_bar.progress(0.97, text="📊 Extracting data...")
    st.session_state.extracted_data = extract_document_data(chunks, st.session_state.doc_type)

    progress_bar.progress(1.0, text="✅ Ready!")
    return True

def run_rag(question):
    if not st.session_state.doc_ready:
        return "Please upload and process a document first.", []

    doc_type = st.session_state.doc_type or "other"
    if doc_type not in DOCUMENT_TYPES:
        doc_type = "other"
    doc_config = DOCUMENT_TYPES[doc_type]

    q_vec = embed_question(question)
    chunks = find_relevant_chunks(q_vec, st.session_state.doc_index, st.session_state.doc_chunks, top_k=5)
    if not chunks:
        return "No relevant content found.", []

    context = "\n".join(f"--- Page {c['page']} ---\n{c['text']}" for c in chunks)
    q_lower = question.lower()
    fmt = "Return ONLY a clean JSON object. No markdown fences. No explanation. Use null for missing values." if any(w in q_lower for w in ["json", "table", "extract", "list", "fields", "values", "format", "show"]) else "Answer concisely. Give the direct answer first. Keep it under 150 words."

    import boto3
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = bedrock.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 800, "system": doc_config["system_prompt"], "messages": [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\n{fmt}"}]}),
        contentType="application/json", accept="application/json"
    )
    return json.loads(response["body"].read())["content"][0]["text"], chunks

def render_table(data_dict):
    rows = ""
    for k, v in data_dict.items():
        if v is None or (isinstance(v, list) and not v):
            continue
        dk = str(k).replace("_", " ").title()
        dv = "<br>".join(str(i) for i in v[:5]) if isinstance(v, list) else "<br>".join(f"{sk}: {sv}" for sk, sv in v.items() if sv) if isinstance(v, dict) else str(v)
        rows += f'<tr><td style="padding:8px 12px;color:#94a3b8;font-size:11px;font-weight:600;border-bottom:1px solid #1f2937;white-space:nowrap;">{dk}</td><td style="padding:8px 12px;color:#e2e8f0;font-size:12px;border-bottom:1px solid #1f2937;">{dv}</td></tr>'
    return f'<table style="width:100%;border-collapse:collapse;background:#111827;border-radius:8px;border:1px solid #1f2937;"><thead><tr style="background:#1f2937;"><th style="padding:8px 12px;text-align:left;color:#60a5fa;font-size:10px;letter-spacing:1px;">FIELD</th><th style="padding:8px 12px;text-align:left;color:#60a5fa;font-size:10px;letter-spacing:1px;">VALUE</th></tr></thead><tbody>{rows}</tbody></table>'


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div style="background:linear-gradient(90deg,#1e3a5f,#111827);padding:14px 22px;border-radius:10px;margin-bottom:16px;border:1px solid #1f2937;"><span style="font-size:20px;font-weight:700;color:#e2e8f0;">📄 Financial Document Assistant</span><span style="font-size:12px;color:#6b7280;margin-left:12px;">Upload any financial document · Auto-detect · Ask anything · AWS Bedrock</span></div>', unsafe_allow_html=True)

col_chat, col_viewer = st.columns([1.2, 1], gap="large")

# ── LEFT: CHAT ────────────────────────────────────────────────────────────────
with col_chat:
    doc_type = st.session_state.doc_type or "other"
    config = DOCUMENT_TYPES.get(doc_type, DOCUMENT_TYPES["other"])

    # Status
    if st.session_state.doc_ready:
        st.markdown(f'<div style="background:#111827;border:1px solid {config["color"]}44;border-radius:10px;padding:10px 14px;margin-bottom:12px;"><span style="color:{config["color"]};font-weight:700;font-size:12px;">✅ {config["icon"]} {config["label"]} detected</span><br><span style="color:#6b7280;font-size:11px;">{st.session_state.pdf_name} · {st.session_state.chunk_count} chunks indexed</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:10px 14px;margin-bottom:12px;"><span style="color:#f59e0b;font-weight:700;font-size:12px;">⏳ No document loaded</span><br><span style="color:#4b5563;font-size:11px;">Upload a PDF on the right and click Process →</span></div>', unsafe_allow_html=True)

    # Dashboard
    if st.session_state.extracted_data:
        with st.expander("📊 Document Dashboard", expanded=True):
            st.markdown(render_table(st.session_state.extracted_data), unsafe_allow_html=True)

    st.markdown("---")

    # Chat
    if not st.session_state.chat_history:
        st.markdown('<div style="text-align:center;padding:32px 16px;border:1px dashed #1f2937;border-radius:12px;"><div style="font-size:32px;">💬</div><div style="font-size:13px;color:#6b7280;margin-top:8px;">Upload a document and start asking questions</div></div>', unsafe_allow_html=True)
        if st.session_state.doc_ready:
            st.markdown('<div style="margin-top:12px;font-size:11px;color:#6b7280;font-weight:700;letter-spacing:1px;">SUGGESTED QUESTIONS</div>', unsafe_allow_html=True)
            for sq in config["suggested_questions"]:
                if st.button(f"→ {sq}", key=f"sq_{sq[:25]}", use_container_width=True):
                    st.session_state["prefill_q"] = sq
                    st.rerun()
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble"><div style="font-size:10px;color:#60a5fa;font-weight:700;margin-bottom:5px;letter-spacing:1px;">YOU</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                content = msg["content"]
                clean = content.strip()
                # Extract JSON even if mixed with text
                if "```json" in clean:
                    try:
                        json_part = clean.split("```json")[1].split("```")[0].strip()
                        clean = json_part
                    except Exception:
                        pass
                elif clean.startswith("```"):
                    clean = "\n".join(clean.split("\n")[1:-1]).strip()

                is_json, parsed = False, None
                try:
                    parsed = json.loads(clean)
                    if isinstance(parsed, dict):
                        is_json = True
                except Exception:
                    pass
                table_html = render_table(parsed) if is_json else content
                st.markdown(f'<div class="bot-bubble"><div style="font-size:10px;color:#34d399;font-weight:700;margin-bottom:5px;letter-spacing:1px;">ASSISTANT</div>{table_html}</div>', unsafe_allow_html=True)
                if msg.get("sources"):
                    pills = "".join(f'<span class="source-pill">{"🟢" if s["distance"]<0.5 else "🟡" if s["distance"]<1.0 else "🔴"} p{s["page"]}</span>' for s in msg["sources"])
                    st.markdown(f"<div style='margin:4px 0 8px 0'>{pills}</div>", unsafe_allow_html=True)

    st.markdown("---")
    prefill = st.session_state.pop("prefill_q", "")
question = st.text_input("question", value=prefill, placeholder="Ask anything about your document...", key="q_input", label_visibility="collapsed", disabled=not st.session_state.doc_ready)
col_ask, col_clear = st.columns([4, 1])
with col_ask:
    ask_clicked = st.button("🔍 Ask", disabled=not st.session_state.doc_ready)
with col_clear:
    if st.button("🗑️"):
        st.session_state.chat_history = []
        st.rerun()

# Auto-submit when suggested question was clicked
if prefill and prefill.strip():
    ask_clicked = True
    question = prefill

    if ask_clicked and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer, sources = run_rag(question)
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error: {str(e)}", "sources": []})
        st.rerun()

# ── RIGHT: UPLOAD ─────────────────────────────────────────────────────────────
with col_viewer:
    st.markdown('<div style="font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:12px;">📂 Upload Document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="uploader", label_visibility="collapsed")

    if uploaded:
        if uploaded.name != st.session_state.pdf_name:
            st.session_state.update({"pdf_name": uploaded.name, "pdf_bytes": uploaded.getvalue(), "doc_ready": False, "doc_index": None, "doc_chunks": None, "chat_history": [], "extracted_data": None, "doc_type": None})

        if not st.session_state.doc_ready:
            if st.button("⚡ Process Document", use_container_width=True):
                if process_pdf(uploaded, st.progress(0, text="Starting...")):
                    st.rerun()
        else:
            st.success(f"✅ {st.session_state.chunk_count} chunks ready")

        if st.session_state.pdf_bytes:
            mb = len(st.session_state.pdf_bytes) / (1024 * 1024)
            # if mb < 2:
            #     try:
            #         st.markdown(f'<iframe src="{pdf_to_base64_url(st.session_state.pdf_bytes)}" width="100%" height="580" style="border:1px solid #1f2937;border-radius:8px;"></iframe>', unsafe_allow_html=True)
            #     except Exception:
            #         st.info("Preview unavailable.")
            # else:
            #     st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:24px;text-align:center;margin-top:8px;"><div style="font-size:36px;">📄</div><div style="font-size:13px;color:#e2e8f0;margin-top:8px;font-weight:600;">{st.session_state.pdf_name}</div><div style="font-size:11px;color:#6b7280;margin-top:4px;">{mb:.1f} MB · Too large for preview</div></div>', unsafe_allow_html=True)
            #     st.download_button("⬇️ Download to view locally", data=st.session_state.pdf_bytes, file_name=st.session_state.pdf_name, mime="application/pdf", use_container_width=True)
            st.info("📄 Document processed and ready. Ask questions on the left.")
    else:
        st.markdown('<div style="text-align:center;padding:80px 20px;border:2px dashed #1f2937;border-radius:12px;color:#374151;margin-top:8px;"><div style="font-size:48px;">📄</div><div style="font-size:14px;color:#6b7280;margin-top:12px;font-weight:600;">Drop any financial PDF here</div><div style="font-size:12px;color:#374151;margin-top:8px;line-height:2;">10-K · 10-Q · 13F · Municipal Bonds<br>Structured Notes · Any financial document</div></div>', unsafe_allow_html=True)