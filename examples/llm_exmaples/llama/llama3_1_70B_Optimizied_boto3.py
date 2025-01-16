# flake8: noqa
import json
import boto3

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You name is Chucky<|eot_id|><|start_header_id|>user<|end_header_id|>

Kể cho tôi câu truyện về chú chó tên Sigma<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

client = boto3.client("bedrock-runtime", region_name="us-west-2")

stream = client.invoke_model_with_response_stream(
    modelId="us.meta.llama3-1-70b-instruct-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(
        {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    ),
    trace="ENABLED",
    performanceConfigLatency="optimized",
)

for event in stream["body"]:
    chunk = json.loads(event.get("chunk")["bytes"].decode())
    print(chunk["generation"], end="", flush=True)
    if chunk["stop_reason"] is not None:
        print("\n")
    if chunk.get("amazon-bedrock-invocationMetrics"):
        print(chunk["amazon-bedrock-invocationMetrics"])
