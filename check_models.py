# import boto3
# from dotenv import load_dotenv

# load_dotenv()

# bedrock = boto3.client("bedrock", region_name="us-east-1")
# models = bedrock.list_foundation_models()

# print("\n✅ ACTIVE MODELS ON YOUR ACCOUNT:\n")
# for m in models["modelSummaries"]:
#     if m["modelLifecycle"]["status"] == "ACTIVE":
#         print(m["modelId"])

import boto3
import json
from dotenv import load_dotenv

load_dotenv()

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Test every possible Claude model ID format
models_to_test = [
    "anthropic.claude-haiku-4-5-20251001",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
]

for model_id in models_to_test:
    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "Hi"}
                ]
            }),
            contentType="application/json",
            accept="application/json"
        )
        print(f"✅ WORKS: {model_id}")
        break
    except Exception as e:
        print(f"❌ FAIL: {model_id}")
        print(f"   {str(e)[:80]}")