from bedrock_llm import Client, ModelName


client = Client(region_name="us-east-1", model_name=ModelName.JAMBA_1_5_MINI)
result = client.generate("Write a story about a dog name Sigma.")

print(result.message.content)
