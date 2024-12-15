from bedrock_llm import Client, ModelName
from bedrock_llm.schema import ToolMetadata, InputSchema, PropertyAttr


weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the weather of a city",
    input_schema=InputSchema(
        type="object",
        properties={
            "city": PropertyAttr(
                type="string",
                description="The city to get the weather of",
            ),
            "unit": PropertyAttr(
                type="string",
                description="The unit of the temperature",
                enum=["celsius", "fahrenheit"],
            ),
        },
        required=["city", "unit"],
    ),
)

client = Client(region_name="us-east-1", model_name=ModelName.JAMBA_1_5_MINI)
result = client.generate(
    prompt="What is the weather in Hanoi and Paris?",
    system="You have a weather tool to check weather",
    tools=[weather_tool],
)

print("Function invoke: ", result.message.tool_calls)
print("Stop reason: ", result.stop_reason.name)
