import os
import sys
import io
import json
import time
import base64
import hashlib
import tempfile
import pickle
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
    .doc-section {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 6px 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    .doc-section:hover {
        border-color: #2563eb;
    }
    .doc-section-active {
        background: #1e3a5f;
        border: 1px solid #2563eb;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 6px 0;
    }
    .stButton button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    .stTextInput input {
        background: #1a1a2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0e0e0e; }
    ::-webkit-scrollbar-thumb {
        background: #374151; border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── IMPORTS ───────────────────────────────────────────────────────────────────

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever import embed_question, find_relevant_chunks, get_answer

# ── DOCUMENT TYPE DEFINITIONS ─────────────────────────────────────────────────
# Each document type has:
# - icon and label for the UI
# - key fields the FA needs to extract
# - a targeted system prompt for Claude
# - suggested questions to guide the analyst

DOCUMENT_TYPES = {
    "structured_note": {
        "icon": "📄",
        "label": "Structured Note / Prospectus",
        "color": "#60a5fa",
        "key_fields": [
            "Underlying / Reference Asset",
            "Buffer / Protection Percentage",
            "Digital Return / Coupon",
            "Maturity Date",
            "Issuer",
            "CUSIP / ISIN",
            "Principal Amount"
        ],
        "system_prompt": """You are an expert analyst specializing 
in structured notes and derivative securities.

TERMINOLOGY GUIDE FOR THIS DOCUMENT TYPE:
- "Underlying" or "asset" = Reference Asset, Index, SPX, etc.
- "Protection" or "buffer" = Buffer Percentage, Downside Protection
- "Return" or "upside" = Digital Upside Return, Coupon Rate
- "Maturity" = Maturity Date, Final Valuation Date
- "Risk" = Risk Factors section

EXTRACTION RULES:
- For JSON requests: return ONLY valid JSON, no markdown fences
- Extract exact values including % signs and dates
- If asking for table data, structure it cleanly
- Map user terminology to document terminology automatically""",
        "suggested_questions": [
            "Extract underlying, protection % and return as table",
            "What are the main risk factors?",
            "What is the payment at maturity formula?",
            "Extract all key terms as JSON"
        ]
    },

    "form_10k": {
        "icon": "📊",
        "label": "10-K Annual Report",
        "color": "#34d399",
        "key_fields": [
            "Total Revenue",
            "Net Income / Loss",
            "Earnings Per Share (EPS)",
            "Total Assets",
            "Total Liabilities",
            "Operating Cash Flow",
            "Fiscal Year End"
        ],
        "system_prompt": """You are an expert financial analyst 
specializing in SEC 10-K annual reports.

TERMINOLOGY GUIDE:
- "Revenue" or "sales" = Net Revenue, Total Revenue, Net Sales
- "Profit" or "earnings" = Net Income, Operating Income
- "EPS" = Earnings Per Share, Basic/Diluted EPS
- "Cash" = Operating Cash Flow, Free Cash Flow
- "Debt" = Long-term Debt, Total Liabilities
- "Assets" = Total Assets, Net Assets
- "Growth" = Year-over-year change in revenue/income

EXTRACTION RULES:
- Always include the fiscal year and currency
- For financial figures include units (millions, billions)
- For JSON: return clean JSON only, no explanation
- Compare current year vs prior year when available""",
        "suggested_questions": [
            "What was total revenue and net income this year?",
            "Extract key financial metrics as a table",
            "What are the main business risks?",
            "How did revenue change year over year?"
        ]
    },

    "form_10q": {
        "icon": "📋",
        "label": "10-Q Quarterly Report",
        "color": "#a78bfa",
        "key_fields": [
            "Quarterly Revenue",
            "Quarterly Net Income",
            "EPS This Quarter",
            "Quarter End Date",
            "YoY Revenue Change",
            "Operating Expenses",
            "Guidance (if provided)"
        ],
        "system_prompt": """You are an expert analyst specializing 
in SEC 10-Q quarterly reports.

TERMINOLOGY GUIDE:
- "Quarter" = Three months ended [date]
- "Revenue" = Net Revenue, Quarterly Revenue, Net Sales
- "Profit" = Net Income, Operating Income for the quarter
- "YoY" = Year over year comparison
- "QoQ" = Quarter over quarter comparison
- "Guidance" = Forward looking statements, outlook

EXTRACTION RULES:
- Always specify which quarter (Q1/Q2/Q3) and year
- Compare to same quarter prior year when available
- For JSON: return clean JSON with quarter specified
- Flag any guidance or forward-looking statements""",
        "suggested_questions": [
            "What was revenue and profit this quarter?",
            "How did this quarter compare to last year?",
            "Extract quarterly financials as a table",
            "What guidance did management provide?"
        ]
    },

    "form_13f": {
        "icon": "📑",
        "label": "Form 13F — Holdings Report",
        "color": "#fb923c",
        "key_fields": [
            "Institution Name",
            "Report Date / Quarter",
            "Total Portfolio Value",
            "Top 10 Holdings",
            "New Positions",
            "Sold Positions",
            "Largest Position %"
        ],
        "system_prompt": """You are an expert analyst specializing 
in SEC Form 13F institutional holdings reports.

TERMINOLOGY GUIDE:
- "Holdings" or "positions" = Securities held, Portfolio
- "Market value" = Current value of position in thousands
- "Shares" = Number of shares held
- "CUSIP" = Security identifier number
- "New position" = Not held in prior quarter
- "Increased" = Added shares vs prior quarter
- "Reduced" = Sold some shares vs prior quarter
- "Investment manager" = The institution filing

EXTRACTION RULES:
- Market values in 13F are reported in thousands of dollars
- Sort holdings by market value descending when listing
- For JSON: use clear keys like ticker, shares, market_value
- Always include the reporting period""",
        "suggested_questions": [
            "What are the top 10 holdings by value?",
            "Extract all holdings as a table",
            "What new positions were added this quarter?",
            "What is the total portfolio value?"
        ]
    },

    "municipal_bond": {
        "icon": "🏛️",
        "label": "Municipal Bond / Official Statement",
        "color": "#f472b6",
        "key_fields": [
            "Issuer / Municipality",
            "Bond Type (GO / Revenue)",
            "Principal Amount",
            "Interest Rate / Yield",
            "Maturity Date",
            "Credit Rating",
            "Tax Status",
            "Use of Proceeds"
        ],
        "system_prompt": """You are an expert analyst specializing 
in municipal bonds and public finance.

TERMINOLOGY GUIDE:
- "Issuer" = The municipality, city, county, or authority
- "Bond type" = General Obligation (GO) or Revenue Bond
- "Yield" or "rate" = Interest Rate, Coupon Rate, Yield to Maturity
- "Rating" = Moody's, S&P, Fitch credit rating
- "Tax status" = Tax-exempt, AMT, Taxable
- "Maturity" = Final maturity date, serial maturities
- "Security" = What backs the bond (tax revenues, project revenue)
- "Call" = Call provisions, first call date

EXTRACTION RULES:
- Always note if interest is tax-exempt or taxable
- Include credit rating agency name with rating
- For JSON: include currency and units for dollar amounts
- Note any special redemption provisions""",
        "suggested_questions": [
            "What is the bond type, rate, and maturity?",
            "Extract key bond terms as a table",
            "What is the credit rating and tax status?",
            "What are the use of proceeds?"
        ]
    },

    "other": {
        "icon": "📁",
        "label": "Other Financial Document",
        "color": "#94a3b8",
        "key_fields": [
            "Document Type",
            "Key Parties",
            "Main Subject",
            "Important Dates",
            "Key Financial Figures",
            "Risk Factors"
        ],
        "system_prompt": """You are an expert financial document analyst.

GENERAL TERMINOLOGY GUIDE:
- Map user's natural language to document terminology
- "underlying" / "asset" → Reference Asset, Security, Index
- "protection" / "buffer" → Buffer %, Protection Level
- "return" / "yield" → Coupon, Rate, Return, Yield
- "issuer" / "company" → Issuer, Registrant, Obligor
- "maturity" / "expiry" → Maturity Date, Expiration
- "holdings" → Portfolio, Securities, Investments
- "revenue" → Net Revenue, Total Revenue, Net Sales
- "profit" → Net Income, Operating Income
- "risk" → Risk Factors, Key Risks, Material Risks

EXTRACTION RULES:
- Extract exact values when found
- For JSON: return clean JSON only
- If term not found look for synonyms first
- Be specific about units and dates""",
        "suggested_questions": [
            "What is this document about?",
            "Extract the key financial terms as a table",
            "What are the main risks mentioned?",
            "Summarize the key points"
        ]
    }
}

# ── SESSION STATE ─────────────────────────────────────────────────────────────

defaults = {
    "chat_history": [],
    "pdf_bytes": None,
    "pdf_name": None,
    "doc_index": None,
    "doc_chunks": None,
    "doc_ready": False,
    "chunk_count": 0,
    "selected_doc_type": None,
    "page_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_doc_hash(pdf_bytes):
    """MD5 fingerprint of PDF — same file = same hash."""
    return hashlib.md5(pdf_bytes).hexdigest()


def get_cache_path(pdf_bytes):
    return f"cache/{get_doc_hash(pdf_bytes)}.pkl"


def load_from_cache(pdf_bytes):
    """Load embeddings from disk if already processed."""
    path = get_cache_path(pdf_bytes)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_to_cache(pdf_bytes, data):
    """Save embeddings to disk for future reuse."""
    os.makedirs("cache", exist_ok=True)
    path = get_cache_path(pdf_bytes)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def pdf_to_base64_url(pdf_bytes):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def extract_markdown(pdf_path):
    """
    Convert PDF to markdown preserving tables and structure.
    Falls back to plain text if pymupdf4llm not available.
    """
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(pdf_path)
    except ImportError:
        # Fallback to pypdf
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text,
                    "page": i + 1,
                    "source": os.path.basename(pdf_path)
                })
        return pages


def chunk_markdown(markdown_text, chunk_size=800, overlap=100):
    """
    Chunk markdown text intelligently —
    tries to split at headers and paragraph boundaries
    rather than mid-sentence.

    Why larger chunks (800 vs 500)?
    Markdown preserves structure so each chunk has more
    meaningful context. Larger chunks = fewer API calls
    = faster processing.

    Why larger overlap (100 vs 50)?
    Markdown sections often reference each other.
    More overlap = less chance of losing cross-references.
    """
    chunks = []
    lines = markdown_text.split('\n')
    current_chunk = ""
    current_start = 0

    for line in lines:
        # If adding this line exceeds chunk size
        # and we're at a good break point
        if len(current_chunk) + len(line) > chunk_size:
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": max(1, current_start),
                    "source": "document",
                    "chunk_id": len(chunks)
                })
                # Keep last overlap chars for next chunk
                current_chunk = current_chunk[-overlap:] + "\n" + line
            else:
                current_chunk += "\n" + line
        else:
            current_chunk += "\n" + line

        # Track approximate page number
        if "<!-- Page" in line:
            try:
                current_start = int(
                    line.split("<!-- Page")[1].split("-->")[0].strip()
                )
            except Exception:
                pass

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "page": max(1, current_start),
            "source": "document",
            "chunk_id": len(chunks)
        })

    return chunks


def embed_locally(chunks):
    """
    Embed using Bedrock Titan with parallel calls.
    
    Instead of 500 sequential calls (2 minutes)
    we run 10 calls at a time simultaneously.
    500 chunks / 10 parallel = 50 batches
    50 batches × 0.5 seconds = ~25 seconds total
    
    8x faster than sequential, no local model needed.
    """
    import boto3
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dotenv import load_dotenv
    load_dotenv()

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1"
    )

    def embed_single(chunk_text):
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": chunk_text}),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["embedding"]

    embeddings = [None] * len(chunks)

    # Run 10 embedding calls at the same time
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {
            executor.submit(embed_single, chunk["text"]): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            embeddings[idx] = future.result()

    return embeddings


def process_pdf(uploaded_file, progress_bar):
    """
    Full pipeline: PDF → Markdown → Chunks → Embeddings → FAISS
    With caching so same doc is never processed twice.
    """
    import faiss

    pdf_bytes = uploaded_file.getvalue()

    # ── Check cache first ──
    progress_bar.progress(0.05, text="Checking cache...")
    cached = load_from_cache(pdf_bytes)

    if cached:
        # Instant load — already processed before
        st.session_state.doc_index = cached["index"]
        st.session_state.doc_chunks = cached["chunks"]
        st.session_state.doc_ready = True
        st.session_state.chunk_count = len(cached["chunks"])
        progress_bar.progress(1.0, text="✅ Loaded from cache instantly!")
        return True

    # ── Write to temp file ──
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # ── Step 1: Extract as Markdown ──
    progress_bar.progress(0.15, text="📖 Converting PDF to markdown...")
    result = extract_markdown(tmp_path)
    os.unlink(tmp_path)

    # Handle both markdown string and pages list
    if isinstance(result, str):
        # Got markdown string — chunk it
        progress_bar.progress(
            0.3, text="✂️ Chunking markdown..."
        )
        chunks = chunk_markdown(result, chunk_size=800, overlap=100)
        # Fix source name
        for c in chunks:
            c["source"] = uploaded_file.name
    else:
        # Got pages list from fallback
        progress_bar.progress(0.3, text="✂️ Chunking text...")
        chunks = []
        for page in result:
            text = page["text"]
            start = 0
            while start < len(text):
                end = start + 800
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "page": page["page"],
                        "source": uploaded_file.name,
                        "chunk_id": len(chunks)
                    })
                start += (800 - 100)

    if not chunks:
        st.error("Could not extract text from this PDF.")
        return False

    # ── Step 2: Embed locally ──
    progress_bar.progress(
        0.5,
        text=f"🔮 Embedding {len(chunks)} chunks locally (fast)..."
    )
    embeddings = embed_locally(chunks)

    # ── Step 3: Build FAISS index ──
    progress_bar.progress(0.85, text="🗄️ Building search index...")
    dimension = len(embeddings[0])
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # ── Step 4: Cache to disk ──
    progress_bar.progress(0.95, text="💾 Saving to cache...")
    save_to_cache(pdf_bytes, {
        "index": index,
        "chunks": chunks
    })

    # ── Store in session ──
    st.session_state.doc_index = index
    st.session_state.doc_chunks = chunks
    st.session_state.doc_ready = True
    st.session_state.chunk_count = len(chunks)
    progress_bar.progress(1.0, text="✅ Ready!")

    return True


def embed_question_locally(question):
    """Embed question using Bedrock Titan."""
    import boto3
    import json
    from dotenv import load_dotenv
    load_dotenv()

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1"
    )
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": question}),
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return np.array([result["embedding"]]).astype("float32")


def run_rag(question, doc_type_key):
    """Full RAG pipeline with document-type-aware prompting."""
    if not st.session_state.doc_ready:
        return "Please upload and process a document first.", []

    doc_config = DOCUMENT_TYPES[doc_type_key]

    # Embed question locally
    q_vec = embed_question_locally(question)

    # Find relevant chunks
    chunks = find_relevant_chunks(
        q_vec,
        st.session_state.doc_index,
        st.session_state.doc_chunks,
        top_k=5
    )

    if not chunks:
        return "No relevant content found.", []

    # Build context
    context = ""
    for chunk in chunks:
        context += f"\n--- Page {chunk['page']} ---\n"
        context += chunk["text"]
        context += "\n"

    # Format instruction based on question intent
    q_lower = question.lower()
    if any(w in q_lower for w in [
        "json", "table", "extract", "list",
        "fields", "values", "format", "show"
    ]):
        format_instruction = (
            "Return ONLY a clean JSON object. "
            "No markdown fences. No explanation. "
            "Use null for missing values."
        )
    else:
        format_instruction = (
            "Answer concisely. Give the direct answer first, "
            "then context. Keep it under 150 words."
        )

    # Full prompt with doc-type system prompt
    import boto3
    import json as json_lib

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1"
    )

    response = bedrock.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=json_lib.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 800,
            "system": doc_config["system_prompt"],
            "messages": [{
                "role": "user",
                "content": f"""Context from document:
{context}

Question: {question}

{format_instruction}"""
            }]
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json_lib.loads(response["body"].read())
    answer = result["content"][0]["text"]

    return answer, chunks


# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="
    background: linear-gradient(90deg, #1e3a5f, #111827);
    padding: 14px 22px;
    border-radius: 10px;
    margin-bottom: 16px;
    border: 1px solid #1f2937;
">
    <span style="font-size:20px; font-weight:700; color:#e2e8f0;">
        📄 Financial Document Assistant
    </span>
    <span style="font-size:12px; color:#6b7280; margin-left:12px;">
        Select document type · Upload · Ask anything
    </span>
</div>
""", unsafe_allow_html=True)

# ── THREE COLUMN LAYOUT ───────────────────────────────────────────────────────
# Left: document type selector
# Middle: chat
# Right: PDF viewer

col_types, col_chat, col_viewer = st.columns(
    [0.8, 1.2, 1],
    gap="medium"
)

# ════════════════════════════════════════
# LEFT — DOCUMENT TYPE SELECTOR
# ════════════════════════════════════════

with col_types:
    st.markdown("""
    <div style="
        font-size:12px;
        font-weight:700;
        color:#6b7280;
        letter-spacing:2px;
        margin-bottom:12px;
    ">SELECT DOCUMENT TYPE</div>
    """, unsafe_allow_html=True)

    for type_key, config in DOCUMENT_TYPES.items():
        is_active = st.session_state.selected_doc_type == type_key
        border_color = config["color"] if is_active else "#1f2937"
        bg_color = "#1a1a2e" if is_active else "#111827"

        st.markdown(f"""
        <div style="
            background: {bg_color};
            border: 1px solid {border_color};
            border-radius: 10px;
            padding: 10px 12px;
            margin: 4px 0;
        ">
            <span style="font-size:16px;">{config['icon']}</span>
            <span style="
                font-size:12px;
                font-weight:{'700' if is_active else '400'};
                color:{'#e2e8f0' if is_active else '#6b7280'};
                margin-left:8px;
            ">{config['label']}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Select" if not is_active else "✓ Selected",
            key=f"select_{type_key}",
            use_container_width=True
        ):
            if st.session_state.selected_doc_type != type_key:
                st.session_state.selected_doc_type = type_key
                st.session_state.doc_ready = False
                st.session_state.doc_index = None
                st.session_state.doc_chunks = None
                st.session_state.chat_history = []
                st.session_state.pdf_bytes = None
                st.session_state.pdf_name = None
            st.rerun()

    # ── Key fields for selected type ──
    if st.session_state.selected_doc_type:
        config = DOCUMENT_TYPES[
            st.session_state.selected_doc_type
        ]
        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:11px; color:#6b7280;
            font-weight:700; letter-spacing:1px;
            margin-bottom:8px;">
            KEY FIELDS
        </div>
        """, unsafe_allow_html=True)
        for field in config["key_fields"]:
            st.markdown(f"""
            <div style="
                font-size:11px;
                color:#94a3b8;
                padding: 3px 0;
                border-bottom: 1px solid #1f2937;
            ">→ {field}</div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════
# MIDDLE — CHAT
# ════════════════════════════════════════

with col_chat:

    if not st.session_state.selected_doc_type:
        # No type selected yet
        st.markdown("""
        <div style="
            text-align:center;
            padding:60px 16px;
            border:1px dashed #1f2937;
            border-radius:12px;
            color:#374151;
            margin-top:20px;
        ">
            <div style="font-size:40px;">←</div>
            <div style="
                font-size:14px;
                color:#6b7280;
                margin-top:12px;
            ">
                Select a document type to get started
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        config = DOCUMENT_TYPES[
            st.session_state.selected_doc_type
        ]

        # ── Status ──
        if st.session_state.doc_ready:
            st.markdown(f"""
            <div style="
                background:#111827;
                border:1px solid {config['color']}44;
                border-radius:10px;
                padding:10px 14px;
                margin-bottom:12px;
            ">
                <span style="
                    color:{config['color']};
                    font-weight:700;
                    font-size:12px;
                ">✅ {config['label']} ready</span>
                <br>
                <span style="color:#6b7280; font-size:11px;">
                    {st.session_state.pdf_name} ·
                    {st.session_state.chunk_count} chunks
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background:#111827;
                border:1px solid #1f2937;
                border-radius:10px;
                padding:10px 14px;
                margin-bottom:12px;
            ">
                <span style="color:#f59e0b; font-weight:700;
                    font-size:12px;">
                    ⏳ Upload a {config['label']}
                </span>
                <br>
                <span style="color:#4b5563; font-size:11px;">
                    Use the panel on the right →
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Chat history ──
        if not st.session_state.chat_history:
            # Show suggested questions
            st.markdown(f"""
            <div style="
                padding:16px;
                border:1px dashed #1f2937;
                border-radius:12px;
            ">
                <div style="
                    font-size:11px;
                    color:#6b7280;
                    font-weight:700;
                    letter-spacing:1px;
                    margin-bottom:10px;
                ">SUGGESTED QUESTIONS</div>
            """, unsafe_allow_html=True)

            for sq in config["suggested_questions"]:
                if st.button(
                    f"→ {sq}",
                    key=f"sq_{sq[:20]}",
                    use_container_width=True,
                    disabled=not st.session_state.doc_ready
                ):
                    # Auto-fill as question
                    st.session_state["prefill_q"] = sq
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Render chat
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="user-bubble">
                        <div style="font-size:10px; color:#60a5fa;
                            font-weight:700; margin-bottom:5px;
                            letter-spacing:1px;">YOU</div>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    content = msg["content"]

                    # Auto-detect JSON and render as table
                    clean = content.strip()
                    if clean.startswith("```"):
                        lines = clean.split("\n")
                        clean = "\n".join(lines[1:-1])
                    clean = clean.strip()

                    is_json = False
                    parsed = None
                    try:
                        import json as jl
                        parsed = jl.loads(clean)
                        if isinstance(parsed, dict):
                            is_json = True
                    except Exception:
                        pass

                    if is_json:
                        st.markdown("""
                        <div class="bot-bubble">
                            <div style="font-size:10px;
                                color:#34d399; font-weight:700;
                                margin-bottom:8px;
                                letter-spacing:1px;">
                                ASSISTANT
                            </div>
                        """, unsafe_allow_html=True)

                        rows = ""
                        for k, v in parsed.items():
                            if isinstance(v, dict):
                                for sk, sv in v.items():
                                    dk = f"{k} — {sk}".replace("_", " ")
                                    rows += f"""
                                    <tr>
                                    <td style="padding:8px 12px;
                                        color:#94a3b8; font-size:11px;
                                        font-weight:600;
                                        border-bottom:1px solid #1f2937;
                                        white-space:nowrap;">
                                        {dk}</td>
                                    <td style="padding:8px 12px;
                                        color:#e2e8f0; font-size:12px;
                                        border-bottom:1px solid #1f2937;">
                                        {sv}</td>
                                    </tr>"""
                            else:
                                dk = str(k).replace("_", " ")
                                rows += f"""
                                <tr>
                                <td style="padding:8px 12px;
                                    color:#94a3b8; font-size:11px;
                                    font-weight:600;
                                    border-bottom:1px solid #1f2937;
                                    white-space:nowrap;">
                                    {dk}</td>
                                <td style="padding:8px 12px;
                                    color:#e2e8f0; font-size:12px;
                                    border-bottom:1px solid #1f2937;">
                                    {v}</td>
                                </tr>"""

                        st.markdown(f"""
                        <table style="width:100%;
                            border-collapse:collapse;
                            background:#111827;
                            border-radius:8px;
                            overflow:hidden;
                            border:1px solid #1f2937;">
                            <thead>
                                <tr style="background:#1f2937;">
                                    <th style="padding:8px 12px;
                                        text-align:left;
                                        color:#60a5fa;
                                        font-size:10px;
                                        letter-spacing:1px;">
                                        FIELD</th>
                                    <th style="padding:8px 12px;
                                        text-align:left;
                                        color:#60a5fa;
                                        font-size:10px;
                                        letter-spacing:1px;">
                                        VALUE</th>
                                </tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.markdown(f"""
                        <div class="bot-bubble">
                            <div style="font-size:10px;
                                color:#34d399; font-weight:700;
                                margin-bottom:5px;
                                letter-spacing:1px;">
                                ASSISTANT</div>
                            {content}
                        </div>
                        """, unsafe_allow_html=True)

                    # Source pills
                    if msg.get("sources"):
                        pills = ""
                        for s in msg["sources"]:
                            rel = (
                                "🟢" if s["distance"] < 0.5
                                else "🟡" if s["distance"] < 1.0
                                else "🔴"
                            )
                            pills += f"""
                            <span class="source-pill">
                                {rel} p{s['page']}
                            </span>"""
                        st.markdown(
                            f"<div style='margin:4px 0 8px 0'>"
                            f"{pills}</div>",
                            unsafe_allow_html=True
                        )

        st.markdown("---")

        # ── Input ──
        prefill = st.session_state.pop("prefill_q", "")
        question = st.text_input(
            "question",
            value=prefill,
            placeholder="Ask anything about your document...",
            key="q_input",
            label_visibility="collapsed",
            disabled=not st.session_state.doc_ready
        )

        col_ask, col_clear = st.columns([4, 1])
        with col_ask:
            ask_clicked = st.button(
                "🔍 Ask",
                disabled=not st.session_state.doc_ready
            )
        with col_clear:
            if st.button("🗑️"):
                st.session_state.chat_history = []
                st.rerun()

        if ask_clicked and question.strip():
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            with st.spinner("Thinking..."):
                try:
                    answer, sources = run_rag(
                        question,
                        st.session_state.selected_doc_type
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

# ════════════════════════════════════════
# RIGHT — PDF UPLOAD + VIEWER
# ════════════════════════════════════════

with col_viewer:
    if not st.session_state.selected_doc_type:
        st.markdown("""
        <div style="
            text-align:center;
            padding:60px 16px;
            border:2px dashed #1f2937;
            border-radius:12px;
            color:#374151;
            margin-top:20px;
        ">
            <div style="font-size:40px;">📄</div>
            <div style="font-size:13px; color:#6b7280;
                margin-top:12px;">
                Select a document type first
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        config = DOCUMENT_TYPES[
            st.session_state.selected_doc_type
        ]
        st.markdown(f"""
        <div style="font-size:13px; font-weight:700;
            color:{config['color']}; margin-bottom:10px;">
            {config['icon']} Upload {config['label']}
        </div>
        """, unsafe_allow_html=True)

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
                st.session_state.doc_ready = False
                st.session_state.doc_index = None
                st.session_state.doc_chunks = None
                st.session_state.chat_history = []

            if not st.session_state.doc_ready:
                if st.button(
                    "⚡ Process Document",
                    use_container_width=True
                ):
                    progress_bar = st.progress(
                        0, text="Starting..."
                    )
                    success = process_pdf(
                        uploaded, progress_bar
                    )
                    if success:
                        st.rerun()
            else:
                st.success(
                    f"✅ {st.session_state.chunk_count} chunks ready"
                )

            # PDF Preview
            if st.session_state.pdf_bytes:
                file_size_mb = len(
                    st.session_state.pdf_bytes
                ) / (1024 * 1024)

                if file_size_mb < 2:
                    try:
                        url = pdf_to_base64_url(
                            st.session_state.pdf_bytes
                        )
                        st.markdown(
                            f'<iframe src="{url}" width="100%"'
                            f' height="580" style="border:1px solid'
                            f' #1f2937; border-radius:8px;"></iframe>',
                            unsafe_allow_html=True
                        )
                    except Exception:
                        st.info("Preview unavailable.")
                else:
                    st.markdown(f"""
                    <div style="
                        background:#111827;
                        border:1px solid #1f2937;
                        border-radius:10px;
                        padding:24px;
                        text-align:center;
                        margin-top:8px;
                    ">
                        <div style="font-size:36px;">📄</div>
                        <div style="font-size:13px;
                            color:#e2e8f0; margin-top:8px;
                            font-weight:600;">
                            {st.session_state.pdf_name}
                        </div>
                        <div style="font-size:11px;
                            color:#6b7280; margin-top:4px;">
                            {file_size_mb:.1f} MB ·
                            Too large for preview
                        </div>
                        {"<div style='font-size:11px; color:#10b981; margin-top:8px;'>✅ " + str(st.session_state.chunk_count) + " chunks indexed</div>" if st.session_state.doc_ready else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    st.download_button(
                        "⬇️ Download to view locally",
                        data=st.session_state.pdf_bytes,
                        file_name=st.session_state.pdf_name,
                        mime="application/pdf",
                        use_container_width=True
                    )
        else:
            st.markdown(f"""
            <div style="
                text-align:center;
                padding:60px 16px;
                border:2px dashed #1f2937;
                border-radius:12px;
                color:#374151;
                margin-top:8px;
            ">
                <div style="font-size:40px;">{config['icon']}</div>
                <div style="font-size:13px; color:#6b7280;
                    margin-top:12px; font-weight:600;">
                    Drop your {config['label']} here
                </div>
                <div style="font-size:11px; color:#374151;
                    margin-top:8px;">
                    PDF format only
                </div>
            </div>
            """, unsafe_allow_html=True)