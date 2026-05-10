import boto3
import json
from dotenv import load_dotenv
load_dotenv()

# Schema + system prompt per document type
DOC_SCHEMAS = {
    "structured_note": {
        "system": """You are a Financial Document Analysis Engine.
RULES:
- Only extract what is explicitly in the document
- Return null if not found, never infer
- Return ONLY valid JSON, no extra text""",
        "schema": {
            "product_type": None,
            "issuer": None,
            "underlying_asset": None,
            "maturity": None,
            "principal_amount": None,
            "max_return_percent": None,
            "buffer_percent": None,
            "payoff_above_buffer": None,
            "payoff_below_buffer": None,
            "estimated_value": None,
            "issue_price": None,
            "implied_cost_percent": None,
            "interest_payments": None,
            "risks": [],
            "scenarios": [],
            "analyst_insight": None
        }
    },

    "form_10k": {
        "system": """You are a Financial Document Analysis Engine.
RULES:
- Only extract what is explicitly in the document
- Return null if not found, never infer
- Return ONLY valid JSON, no extra text""",
        "schema": {
            "company_name": None,
            "fiscal_year": None,
            "total_revenue": None,
            "revenue_previous_year": None,
            "net_income": None,
            "net_income_previous_year": None,
            "eps_basic": None,
            "eps_diluted": None,
            "total_assets": None,
            "total_liabilities": None,
            "operating_cash_flow": None,
            "key_risks": [],
            "business_segments": [],
            "analyst_insight": None
        }
    },

    "form_10q": {
        "system": """You are a Financial Document Analysis Engine.
RULES:
- Only extract what is explicitly in the document
- Return null if not found, never infer
- Return ONLY valid JSON, no extra text""",
        "schema": {
            "company_name": None,
            "quarter": None,
            "period_end_date": None,
            "quarterly_revenue": None,
            "revenue_same_quarter_prior_year": None,
            "net_income": None,
            "eps": None,
            "operating_expenses": None,
            "guidance": None,
            "key_changes": [],
            "analyst_insight": None
        }
    },

    "form_13f": {
        "system": """You are a Financial Document Analysis Engine.
RULES:
- Only extract what is explicitly in the document
- Return null if not found, never infer
- Return ONLY valid JSON, no extra text""",
        "schema": {
            "institution_name": None,
            "report_date": None,
            "total_portfolio_value": None,
            "top_holdings": [],
            "new_positions": [],
            "closed_positions": [],
            "largest_position": None,
            "analyst_insight": None
        }
    },

    "municipal_bond": {
        "system": """You are a Financial Document Analysis Engine.
RULES:
- Only extract what is explicitly in the document
- Return null if not found, never infer
- Return ONLY valid JSON, no extra text""",
        "schema": {
            "issuer": None,
            "bond_type": None,
            "principal_amount": None,
            "interest_rate": None,
            "maturity_date": None,
            "credit_rating": None,
            "tax_status": None,
            "use_of_proceeds": None,
            "security": None,
            "call_provisions": None,
            "risks": [],
            "analyst_insight": None
        }
    },

    "other": {
        "system": """You are a Financial Document Analysis Engine.
RULES:
- Only extract what is explicitly in the document
- Return null if not found, never infer
- Return ONLY valid JSON, no extra text""",
        "schema": {
            "document_type": None,
            "key_parties": [],
            "main_subject": None,
            "important_dates": [],
            "key_financial_figures": [],
            "risks": [],
            "analyst_insight": None
        }
    }
}


def extract_document_data(chunks, doc_type):
    """
    Sends first 20 chunks to Claude with schema.
    Claude fills in the schema from the document.
    Returns filled JSON dict.
    """
    bedrock = boto3.client(
        "bedrock-runtime", region_name="us-east-1"
    )

    config = DOC_SCHEMAS.get(doc_type, DOC_SCHEMAS["other"])

    # Use first 20 chunks for extraction
    context = "\n".join(
        f"--- Page {c['page']} ---\n{c['text']}"
        for c in chunks[:20]
    )

    schema_str = json.dumps(config["schema"], indent=2)

    response = bedrock.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "system": config["system"],
            "messages": [{
                "role": "user",
                "content": f"""Document content:
{context}

Fill this JSON schema using ONLY information from above.
Return null for missing fields.
Return ONLY valid JSON, no explanation.

Schema:
{schema_str}"""
            }]
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    raw = result["content"][0]["text"].strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    try:
        return json.loads(raw)
    except Exception:
        return config["schema"]

    # Embed each query and find relevant chunks
    def embed_text(text):
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text}),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return np.array([result["embedding"]]).astype("float32")

    # Collect unique relevant chunks across all queries
    seen_ids = set()
    relevant_chunks = []

    for query in queries:
        q_vec = embed_text(query)
        distances, indices = index.search(q_vec, 3)
        for idx in indices[0]:
            if idx not in seen_ids:
                seen_ids.add(idx)
                relevant_chunks.append(all_chunks[idx])

    # Build context from RAG-retrieved chunks
    context = "\n".join(
        f"--- Page {c['page']} ---\n{c['text']}"
        for c in relevant_chunks
    )

    schema_str = json.dumps(config["schema"], indent=2)

    response = bedrock.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "system": config["system"],
            "messages": [{
                "role": "user",
                "content": f"""Document content:
{context}

Fill this JSON schema using ONLY information from above.
Return null for missing fields.
Return ONLY valid JSON.

Schema:
{schema_str}"""
            }]
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    raw = result["content"][0]["text"].strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    try:
        return json.loads(raw)
    except Exception:
        return config["schema"]