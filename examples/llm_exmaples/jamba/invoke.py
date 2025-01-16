from bedrock_llm import Client, ModelName


client = Client(region_name="us-east-1", model_name=ModelName.JAMBA_1_5_MINI)
result = client.generate("Viết câu truyện về một con chó tên Sigma")

print(result.message.content)
