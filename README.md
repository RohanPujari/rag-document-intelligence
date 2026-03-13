# rag-document-intelligence
RAG pipeline using AWS Bedrock and FAISS for financial document analysis

# RAG Document Intelligence Pipeline
### AWS Bedrock + FAISS + Streamlit

A production-style Retrieval Augmented Generation (RAG) pipeline 
for financial document analysis. Built to replicate real-world 
document intelligence workflows for financial analysts.

---

## What It Does

Financial analysts spend hours manually reading through dense 
documents — prospectuses, bond offerings, fund filings. This 
pipeline automates that by letting analysts ask natural language 
questions and getting precise answers grounded in the actual documents.

---

## Architecture
```
PDF Documents (prospectus, bond filings)
        ↓
Text Extraction (PyPDF)
        ↓
Chunking (500 chars, 50 overlap)
        ↓
Embedding (Amazon Titan Embeddings V2 via Bedrock)
        ↓
Vector Store (FAISS — local index)
        ↓
User Question → Embed → Similarity Search → Top 3 Chunks
        ↓
AWS Bedrock (Claude) → Grounded Answer
        ↓
Streamlit UI
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | AWS Bedrock (Claude 3 Haiku) |
| Embeddings | Amazon Titan Embeddings V2 |
| Vector Store | FAISS |
| PDF Processing | PyPDF |
| UI | Streamlit |
| AWS SDK | boto3 |

---

## Project Structure
```
rag-document-intelligence/
├── data/                  # PDF documents (not tracked in git)
├── ingest.py              # Extract → Chunk → Embed → Store
├── retriever.py           # Question → Search → Answer
├── app.py                 # Streamlit UI
└── .env                   # AWS credentials (not tracked in git)
```

---

## Key Technical Decisions

**Chunk size: 500 characters, Overlap: 50 characters**
Financial documents have dense information per sentence. 
500 characters captures enough context per chunk while 
keeping Bedrock payload small. 50 character overlap prevents 
key sentences from being split across chunk boundaries.

**Why FAISS over a managed vector DB?**
For a single-user document analysis tool, local FAISS gives 
sub-millisecond search with zero infrastructure cost. 
For production multi-user scale, this would migrate to 
Amazon OpenSearch or Pinecone.

**Why Amazon Titan Embeddings?**
AWS-native — no external API keys, IAM role controls access, 
consistent latency within the AWS network. 1536-dimension 
vectors give high semantic resolution for financial terminology.

---

## Setup

1. Clone the repo
2. Install dependencies:
```
   pip install boto3 pypdf faiss-cpu streamlit python-dotenv
```
3. Add AWS credentials to `.env`:
```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_DEFAULT_REGION=us-east-1
```
4. Add PDF files to `data/` folder
5. Run ingestion:
```
   python ingest.py
```
6. Launch UI:
```
   streamlit run app.py
```

---

## Sample Questions

- "What are the risk factors associated with these notes?"
- "What is the Digital Upside Return and when does it apply?"
- "What are the tax implications of investing in municipal bonds?"
- "What happens if the S&P 500 drops more than 10%?"

---

## What I'd Add in V2

- **Reranking** — retrieve top 20 chunks, cross-encoder rerank to top 5
- **Section-aware chunking** — detect document sections and chunk by section boundary
- **AWS Lambda + API Gateway** — replace direct boto3 calls with serverless API
- **Amazon OpenSearch** — replace local FAISS for multi-user production scale
- **MLflow** — track embedding experiments and retrieval quality metrics
- **RAGAS evaluation** — measure retrieval precision and answer faithfulness

---

*Built as part of AWS MLOps portfolio — demonstrates end-to-end 
RAG pipeline with production-grade AWS services.*
```
