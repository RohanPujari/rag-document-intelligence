import os
import json
import boto3
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── STEP 1: LOAD THE VECTOR STORE ────────────────────────────────────────────

def load_vector_store():
    """
    Loads the FAISS index and chunks we created in ingest.py.
    
    Remember:
    faiss_index.bin  = the vectors (numbers representing meaning)
    chunks_store.pkl = the actual text chunks
    
    We need both — FAISS finds which vectors are closest,
    then we use chunks_store to get the actual text back.
    """
    if not os.path.exists("faiss_index.bin"):
        print("❌ No vector store found.")
        print("   Please run ingest.py first.")
        return None, None

    index = faiss.read_index("faiss_index.bin")

    with open("chunks_store.pkl", "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded vector store — {index.ntotal} vectors")
    print(f"✅ Loaded {len(chunks)} chunks")
    return index, chunks


# ── STEP 2: EMBED THE QUESTION ────────────────────────────────────────────────

def embed_question(question):
    """
    Converts the user's question into a vector — same way
    we converted chunks in ingest.py.
    
    Why same model?
    The question and chunks must be in the same vector space
    to be comparable. If you embed chunks with Titan and 
    questions with a different model — the vectors are 
    incompatible. Like measuring distance in miles vs kilometers.
    """
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": question}),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return np.array([result["embedding"]]).astype("float32")


# ── STEP 3: FIND RELEVANT CHUNKS ─────────────────────────────────────────────

def find_relevant_chunks(question_vector, index, chunks, top_k=3):
    """
    Searches the FAISS index for the top_k most similar chunks.
    
    top_k=3 means we retrieve the 3 closest chunks.
    
    Why 3?
    - Too few (1) = might miss important context
    - Too many (10) = bloats the prompt, confuses the model,
                      costs more tokens on Bedrock
    - 3 is the sweet spot for focused financial Q&A
    
    distances = how far each chunk is from the question vector
                lower distance = more similar = more relevant
    indices   = which chunk numbers are closest
    """
    distances, indices = index.search(question_vector, top_k)

    relevant_chunks = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        relevant_chunks.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "distance": float(distances[0][i])
        })

    return relevant_chunks


# ── STEP 4: BUILD THE PROMPT ──────────────────────────────────────────────────

def build_prompt(question, relevant_chunks):
    """
    Combines the retrieved chunks with the question into
    a single prompt for Claude.
    
    The key instruction: "Answer ONLY using the context below"
    
    Why this matters:
    Without this instruction Claude might answer from its 
    general training knowledge — which could be outdated or 
    wrong for this specific document.
    
    With this instruction Claude stays grounded in YOUR 
    documents. If the answer isn't in the chunks, it says
    "I don't know" instead of hallucinating.
    
    This is what makes RAG trustworthy for financial documents
    where accuracy is critical.
    """
    context = ""
    for i, chunk in enumerate(relevant_chunks):
        context += f"\n--- Source: {chunk['source']} | "
        context += f"Page {chunk['page']} ---\n"
        context += chunk["text"]
        context += "\n"

    prompt = f"""You are a financial document analyst. 
Answer the question below using ONLY the context provided.
If the answer is not in the context, say "I cannot find 
this information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    return prompt


# ── STEP 5: GET ANSWER FROM BEDROCK ──────────────────────────────────────────

def get_answer(prompt):
    """
    Sends the prompt to Claude via AWS Bedrock and gets answer.
    
    We use Claude 3 Haiku — the fastest and cheapest Claude model.
    Perfect for Q&A tasks where speed matters.
    
    For more complex reasoning or summarization you'd upgrade
    to Claude 3 Sonnet or Claude 3 Opus.
    
    max_tokens=500 — limits the response length
    Enough for a detailed answer, not so much it rambles.
    """
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


# ── STEP 6: SHOW SOURCES ──────────────────────────────────────────────────────

def show_sources(relevant_chunks):
    """
    Shows which document and page each answer came from.
    
    This is critical for financial documents — analysts need
    to verify the source, not just trust the AI answer.
    This is what makes RAG auditable and trustworthy.
    """
    print("\n📄 Sources used:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"  {i+1}. {chunk['source']} "
              f"— Page {chunk['page']} "
              f"(relevance score: {chunk['distance']:.3f})")


# ── MAIN: ASK A QUESTION ──────────────────────────────────────────────────────

def ask(question):
    """
    Full pipeline:
    Question → Embed → Search → Build Prompt → Get Answer → Show Sources
    """
    print(f"\n🔍 Question: {question}")
    print("─" * 50)

    # Load vector store
    index, chunks = load_vector_store()
    if index is None:
        return

    # Embed the question
    print("Embedding question...")
    question_vector = embed_question(question)

    # Find relevant chunks
    print("Searching for relevant context...")
    relevant_chunks = find_relevant_chunks(question_vector, index, chunks)

    # Build prompt
    prompt = build_prompt(question, relevant_chunks)

    # Get answer
    print("Getting answer from Claude...\n")
    answer = get_answer(prompt)

    print(f"💡 Answer:\n{answer}")
    show_sources(relevant_chunks)
    return answer


# ── TEST IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is the Digital Upside Return for the HSBC notes?",
        "What are the tax benefits of municipal bonds?",
        "What happens if the S&P 500 drops more than 10%?"
    ]
    for question in questions:
        ask(question)
        print("\n" + "═" * 60 + "\n")