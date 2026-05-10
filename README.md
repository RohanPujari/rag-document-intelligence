# Financial Document Intelligence Platform
### AWS Bedrock + FAISS + Claude + Streamlit

A production-deployed RAG pipeline that transforms dense financial documents into instant, queryable intelligence. Built for financial analysts who need answers fast — not another PDF to read.

🔗 **[Live Demo](https://your-app.streamlit.app)** · **[GitHub](https://github.com/RohanPujari/rag-document-intelligence)**

---

## What It Does

Upload any financial document and get:

- **Auto-detected document type** — 10-K, 10-Q, 13F, Municipal Bond, Structured Note
- **Instant dashboard** — key fields extracted automatically (revenue, net income, EPS, holdings, yield, etc.)
- **Chat interface** — ask anything in plain English, get answers grounded in the document
- **Source attribution** — every answer shows exactly which page it came from
- **JSON → Table rendering** — structured data renders as clean tables automatically

---

## Supported Document Types

| Type | Auto-Extracted Fields |
|------|----------------------|
| 📊 10-K Annual Report | Revenue, Net Income, EPS, Total Assets, Key Risks |
| 📋 10-Q Quarterly Report | Quarterly Revenue, Net Income, YoY Change, Guidance |
| 📑 Form 13F | Top Holdings, Market Value, New/Closed Positions |
| 🏛️ Municipal Bond | Yield, Maturity, Credit Rating, Tax Status |
| 📄 Structured Note | Underlying, Buffer %, Digital Return, Maturity |

---

## Architecture

```
User uploads PDF
        ↓
PDF → Markdown conversion (pymupdf4llm)
        ↓
Intelligent chunking (800 chars, 100 overlap)
        ↓
Parallel embedding via Amazon Titan Embeddings V2
(3 concurrent Bedrock calls with retry logic)
        ↓
FAISS vector index built in memory
MD5 hash cached to disk (same doc = instant reload)
        ↓
Auto-classification (Claude detects doc type)
        ↓
Schema extraction (targeted fields per doc type)
        ↓
User asks question
        ↓
Question embedded → FAISS similarity search → Top 5 chunks
        ↓
Claude (via Bedrock) generates grounded answer
        ↓
JSON responses auto-rendered as tables
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | AWS Bedrock — Claude Haiku 4.5 | Latest model, inference profile routing |
| Embeddings | Amazon Titan Embeddings V2 | AWS-native, no external API keys, 1536-dim |
| Vector Store | FAISS (in-memory) | Sub-millisecond search, zero infrastructure cost |
| PDF Processing | pymupdf4llm + pypdf | Markdown conversion preserves tables and structure |
| Classification | Claude via Bedrock | Auto-detects doc type from first 5 chunks |
| Extraction | Claude via Bedrock | Schema-driven extraction per document type |
| Caching | MD5 hash + pickle | Same document reloads instantly from disk |
| UI | Streamlit | Deployed on Streamlit Cloud |
| AWS SDK | boto3 | Parallel embedding calls with throttle retry |

---

## Project Structure

```
rag-document-intelligence/
├── app.py              # Main Streamlit app — upload, chat, dashboard
├── classifier.py       # Auto-detects document type using Claude
├── extractor.py        # Schema-based field extraction per doc type
├── retriever.py        # FAISS search + Claude answer generation
├── ingest.py           # Standalone ingestion pipeline (legacy)
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit server config
└── .env                # AWS credentials (not tracked in git)
```

---

## Key Technical Decisions

**Markdown conversion before chunking**
Raw PDF extraction mangles tables — numbers lose context. Converting to markdown first preserves table structure, headers, and section boundaries. Financial documents are full of tables; this matters.

**Parallel embedding with throttle retry**
Sequential Bedrock calls take 2+ minutes for large documents. Parallel calls with 3 workers and exponential backoff on throttling bring this to under 30 seconds for most documents.

**MD5 hash caching**
Same document uploaded twice never gets re-embedded. The hash fingerprints the file, cached embeddings load from disk instantly. Critical for analyst workflows where the same filings get reviewed repeatedly.

**Per-document-type system prompts**
A 10-K and a 13F use completely different terminology for similar concepts. Generic prompts fail. Each document type has a targeted system prompt with terminology mappings — "Underlying" maps to "Reference Asset" for structured notes, "Holdings" maps to "Securities held" for 13F, etc.

**In-memory FAISS vs managed vector DB**
For a single-session document analysis tool, local FAISS gives sub-millisecond search with zero cost. For multi-user production scale this migrates to Amazon OpenSearch.

---

## Setup

```bash
git clone https://github.com/RohanPujari/rag-document-intelligence
cd rag-document-intelligence
pip install -r requirements.txt
```

Add AWS credentials to `.env`:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

Run locally:
```bash
streamlit run app.py
```

---

## Sample Questions by Document Type

**10-Q (Starbucks):**
- "What was revenue and profit this quarter?"
- "How did this quarter compare to last year?"
- "Extract quarterly financials as a table"

**Structured Note (HSBC):**
- "What is the Digital Upside Return?"
- "What happens if the S&P 500 drops more than 10%?"
- "Extract underlying, protection % and return as table"

**Municipal Bond:**
- "What is the credit rating and tax status?"
- "What is the bond type, rate, and maturity?"

---

## Validation Results

Tested against Starbucks 10-Q (Q2 FY2024):

| Field | Extracted | Actual | Match |
|-------|-----------|--------|-------|
| Company Name | Starbucks Corporation | Starbucks Corporation | ✅ |
| Quarter | Q2 2024 | Q2 FY2024 | ✅ |
| Revenue | $8,563.0M | $8,563.0M | ✅ |
| Net Income | $772.4M | $772.4M | ✅ |
| EPS | $0.68 | $0.68 | ✅ |
| Operating Income | $1,098.9M | $1,098.9M | ✅ |
| Operating Expenses | $7,532.1M | $7,532.1M | ✅ |
| YoY Revenue Change | -1.8% | -1.8% | ✅ |

**8/8 core financial fields correct.**

---

## What's Next (V2)

- **RAG-based extraction** — use similarity search to find relevant chunks per field instead of first N chunks — eliminates the "answer on page 50" problem
- **Cross-document comparison** — compare two filings side by side
- **Agentic news layer** — agent fetches financial news, RAG answers questions across articles + filings
- **MLflow experiment tracking** — log every query, latency, and relevance score
- **Amazon OpenSearch** — replace FAISS for multi-user production scale
- **AWS Lambda + API Gateway** — decouple frontend from Bedrock calls

---

*Built as part of an AWS MLOps portfolio. Demonstrates end-to-end RAG pipeline with production-grade AWS services, deployed and live.*
