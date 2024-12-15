import json
import boto3


client = boto3.client("bedrock-runtime", region_name="us-east-1")

# message = {
#     "role": "user",
#     "content": "What is the weather like right now in Melbourne, Australia?",
# }

message = {
    "role": "user",
    "content": "Thủ đô của Ý bây giờ là gì? Tại sao?",
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather from a specific location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather status.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of measurement for the temperature.",
                    },
                },
                "required": ["location", "unit"],
            },
        },
    }
]

documents = {
    "content": "The capital of Italy is now move to Paris after the World War 3",
    "metadata": {
        "source": "https://bbc.com",
        "title": "New Ital's Capital shocks everyone",
    },
}

body = {
    "messages": [message],
    # "tools": tools,
    # "documents": [documents],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": [],
    "n": 1,
    # "response_format": {"type": "json_object"},
}

respont = client.invoke_model(
    modelId="ai21.jamba-1-5-mini-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(body),
    trace="ENABLED",
)

# respont = client.invoke_model_with_response_stream(
#     modelId="ai21.jamba-1-5-mini-v1:0",
#     contentType="application/json",
#     accept="application/json",
#     body=json.dumps(body),
#     trace="ENABLED",
# )

for event in respont["body"]:
    print(event)
