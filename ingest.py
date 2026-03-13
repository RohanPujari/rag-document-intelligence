import os
import boto3
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ── STEP 1: READ THE PDF ──────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and extracts all text page by page.
    We store which page each text came from — useful for debugging
    and for telling the user where the answer came from.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():  # skip blank pages
            pages.append({
                "text": text,
                "page": page_num + 1,        # human readable page number
                "source": os.path.basename(pdf_path)  # which file it came from
            })

    print(f"  Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
    return pages


# ── STEP 2: CHUNK THE TEXT ────────────────────────────────────────────────────

def chunk_pages(pages, chunk_size=500, overlap=50):
    """
    Splits extracted text into smaller chunks.

    Why chunk at all?
    Bedrock has a context limit — you can't send 200 pages at once.
    Also smaller focused chunks = better answers than one giant blob.

    chunk_size = 500 characters per chunk
    overlap    = 50 characters repeated between chunks

    Why overlap?
    Imagine a sentence starts at character 498 and ends at 520.
    Without overlap that sentence gets CUT across two chunks
    and becomes unfindable. Overlap prevents that.

    Example with overlap=50:
    Chunk 1: characters 0   → 500
    Chunk 2: characters 450 → 950   ← repeats last 50 of chunk 1
    Chunk 3: characters 900 → 1400  ← repeats last 50 of chunk 2
    """
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():  # skip empty chunks
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                    "source": page["source"],
                    "chunk_id": len(chunks)
                })

            # move forward by chunk_size MINUS overlap
            # this is what creates the overlap between chunks
            start += (chunk_size - overlap)

    print(f"  Created {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap})")
    return chunks


# ── STEP 3: EMBED THE CHUNKS ──────────────────────────────────────────────────

def embed_chunks(chunks):
    """
    Converts each chunk of text into a vector — a list of numbers
    that represents the MEANING of that text.

    Why vectors?
    Computers can't compare meaning directly.
    But they CAN compare numbers.
    Similar meaning = vectors pointing in similar direction in space.

    We use Amazon Titan Embeddings via Bedrock.
    It takes text → returns 1536 numbers representing its meaning.

    Example:
    "management fees are 1.5%" → [0.23, -0.41, 0.87, ... 1536 numbers]
    "what are the fees?"       → [0.21, -0.39, 0.91, ... 1536 numbers]
    These two are CLOSE in vector space → retriever finds it ✅
    """
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    embeddings = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        import json

        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({
                "inputText": chunk["text"]
            }),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        embedding = result["embedding"]  # list of 1536 numbers
        embeddings.append(embedding)

        # progress update every 10 chunks
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Embedded {i+1}/{total} chunks...")

    return embeddings


# ── STEP 4: STORE IN FAISS ────────────────────────────────────────────────────

def build_vector_store(chunks, embeddings):
    """
    FAISS is a vector database that lives on your local machine.
    It stores all the embeddings and lets you search them instantly.

    Think of it as a super fast search index — like Google's index
    but for meaning, not keywords.

    We save two things:
    1. faiss_index.bin  → the actual vector index (for searching)
    2. chunks_store.pkl → the original text chunks (to retrieve)

    We need both because FAISS only stores vectors, not the text.
    When FAISS finds the closest vector, we use chunks_store.pkl
    to get the actual text back.
    """
    dimension = len(embeddings[0])  # 1536 for Titan
    vectors = np.array(embeddings).astype("float32")

    # IndexFlatL2 = exact search using L2 (Euclidean) distance
    # For production you'd use IndexIVFFlat for speed at scale
    # For our use case exact search is perfect
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # save both files
    faiss.write_index(index, "faiss_index.bin")

    with open("chunks_store.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"  Saved FAISS index with {index.ntotal} vectors")
    print(f"  Saved {len(chunks)} chunks to chunks_store.pkl")


# ── MAIN: RUN THE FULL PIPELINE ───────────────────────────────────────────────

def ingest_all_pdfs(data_folder="data"):
    """
    Runs the full pipeline on every PDF in the data folder.
    Extract → Chunk → Embed → Store
    """
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in data/ folder")
        return

    print(f"\nFound {len(pdf_files)} PDF files: {pdf_files}\n")

    all_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"Processing: {pdf_file}")

        # Step 1 — extract
        pages = extract_text_from_pdf(pdf_path)

        # Step 2 — chunk
        chunks = chunk_pages(pages, chunk_size=500, overlap=50)

        all_chunks.extend(chunks)
        print()

    print(f"Total chunks across all documents: {len(all_chunks)}")
    print("\nEmbedding chunks (this takes a minute)...")

    # Step 3 — embed
    embeddings = embed_chunks(all_chunks)

    # Step 4 — store
    print("\nBuilding vector store...")
    build_vector_store(all_chunks, embeddings)

    print("\n✅ Ingestion complete. Ready to answer questions.")
    print(f"   Total documents: {len(pdf_files)}")
    print(f"   Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    ingest_all_pdfs()


# ## Before You Run — Understand What This Does

# Four steps happen when you press play:
# ```
# Your PDFs
#    ↓
# Step 1: Extract — reads every page, stores text + page number
#    ↓
# Step 2: Chunk — splits into 500 character pieces, 50 overlap
#    ↓
# Step 3: Embed — sends each chunk to Titan on Bedrock
#          gets back 1536 numbers representing its meaning
#    ↓
# Step 4: Store — saves everything to faiss_index.bin
#                 and chunks_store.pkl on your machine
# ```

# After this runs — your documents are searchable by meaning. Not by keyword. By meaning.

# ---

# ## Now Run It

# Press the ▶️ play button in VS Code.

# You'll see something like:
# ```
# Found 2 PDF files: ['13f_filing.pdf', 'municipal_bond.pdf']

# Processing: 13f_filing.pdf
#   Extracted 4 pages from 13f_filing.pdf
#   Created 18 chunks (size=500, overlap=50)

# Processing: municipal_bond.pdf
#   Extracted 142 pages from municipal_bond.pdf
#   Created 891 chunks (size=500, overlap=50)

# Total chunks across all documents: 909

# Embedding chunks (this takes a minute)...
#   Embedded 10/909 chunks...
#   Embedded 20/909 chunks...
#   ...
#   Embedded 909/909 chunks...

# Building vector store...
#   Saved FAISS index with 909 vectors
#   Saved 909 chunks to chunks_store.pkl

# ✅ Ingestion complete. Ready to answer questions.
#    Total documents: 2
#    Total chunks: 909