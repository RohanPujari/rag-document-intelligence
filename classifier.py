import boto3
import json
from dotenv import load_dotenv
load_dotenv()

def classify_document(chunks):
    """
    Sends first 5 chunks to Claude.
    Claude tells us what type of document it is.
    Returns one of: structured_note, form_10k, form_10q, 
                    form_13f, municipal_bond, other
    """
    bedrock = boto3.client(
        "bedrock-runtime", region_name="us-east-1"
    )

    # Use first 5 chunks as sample — enough to classify
    sample = "\n".join(c["text"] for c in chunks[:5])

    response = bedrock.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "system": """You are a financial document classifier.
Return ONLY one of these exact values, nothing else:
structured_note
form_10k
form_10q
form_13f
municipal_bond
other""",
            "messages": [{
                "role": "user",
                "content": f"Classify this document:\n{sample}"
            }]
        }),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    doc_type = result["content"][0]["text"].strip().lower()

    # Validate it's one of our known types
    valid_types = [
        "structured_note", "form_10k", "form_10q",
        "form_13f", "municipal_bond", "other"
    ]
    return doc_type if doc_type in valid_types else "other"