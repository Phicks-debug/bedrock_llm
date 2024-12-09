from bedrock_llm import Client, ModelName
from bedrock_llm.schema import ToolMetadata, InputSchema, PropertyAttr

weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the weather for a given location usiing AccuWeather API",
    input_schema=InputSchema(
        properties={
            "location": PropertyAttr(
                type="string",
                description="The location for which to get the weather",
            ),
            "unit": PropertyAttr(
                type="string",
                enum=["celsius", "fahrenheit"],
                description="The unit of temperature",
            ),
        },
        required=["location"],
    ),
)

client = Client(region_name="us-east-1", model_name=ModelName.NOVA_PRO)
result = client.generate(
    prompt="What is the weather in Hanoi and Paris?",
    system="You have a weather tool to check weather",
    tools=[weather_tool],
)

print("Function invoke: ", result.message.content)
print("Stop reason: ", result.stop_reason.name)
