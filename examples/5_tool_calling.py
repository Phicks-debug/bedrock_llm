# Add for print console with color
from termcolor import cprint

from bedrock_llm import Client, ModelConfig, ModelName, RetryConfig
from bedrock_llm.schema import InputSchema, PropertyAttr, ToolMetadata, MessageBlock

# Create a LLM client
client = Client(
    region_name="us-west-2",
    model_name=ModelName.LLAMA_3_2_3B,
    memory=[],
    retry_config=RetryConfig(max_attempts=3),
)

# Create a configuration for inference parameters
config = ModelConfig(temperature=0.1, top_p=0.9, max_tokens=512)

# Create tool definition
get_weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the real time weather of a city",
    input_schema=InputSchema(
        type="object",
        properties={
            "location": PropertyAttr(
                type="string", description="The city to get the weather of"
            )
        },
        required=["location"],
    ),
)

# Create a system prompt with a list of examples
prompt = [
    MessageBlock(role="user", content="What is the capital of France?"),
    MessageBlock(role="assistant", content="The capital of France is Paris."),
    MessageBlock(role="user", content="What is the capital of Germany?"),
    MessageBlock(role="assistant", content="The capital of Germany is Berlin."),
    MessageBlock(
        role="user",
        content="What is the weather in the capital of Italy?",
    ),
]
system = "You are a helpful assistant. You can use tools to get data weather of a city."

# Invoke the model and get results
response = client.generate(config=config, prompt=prompt, tools=[get_weather_tool])

# Print out the results
cprint(f"Calling function: {response.message.content}", "cyan")
cprint(response.stop_reason.name, "red")

# Send the tool result back to the model
response = client.generate(
    config=config,
    prompt=MessageBlock(role="tool", content="20*C"),
)

# Print out the results
cprint(response.message.content, "green")
cprint(response.stop_reason, "red")
