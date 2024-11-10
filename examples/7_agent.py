import sys
import os

# Add for print console with color
from termcolor import cprint

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import ModelName, MessageBlock
from src.bedrock_llm.agent import Agent
from src.bedrock_llm.schema.tools import ToolMetadata, InputSchema, PropertyAttr
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.config.base import RetryConfig
from src.bedrock_llm.types.enums import StopReason

system = "You are a helpful assistant. You have access to realtime information. You can use tools to get the real time data weather of a city."

# Create a LLM client
agent = Agent(
    region_name="us-east-1",
    model_name=ModelName.CLAUDE_3_5_HAIKU,
    retry_config=RetryConfig(
        max_attempts=3
    )
)

# Create a configuration for inference parameters
config = ModelConfig(
    temperature=0.1,
    top_p=0.9,
    max_tokens=512
)

# Create tool definition
get_weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the real time weather of a city",
    input_schema=InputSchema(
        type="object",
        properties={
            "location": PropertyAttr(
                type="string",
                description="The city to get the weather of"
            )
        },
        required=["location"]
    )
)

# Create user prompt
prompt = MessageBlock(role="user", content="What is the weather in New York and Toronto?")

@Agent.tool(get_weather_tool)
async def get_weather(location: str):
    # Mock function to get weather
    return f"tools_result: {location} is 20*C"


async def main():
    # Invoke the model and get results
    async for token, stop_reason, response, tool_result in agent.generate_and_action_async(
        config=config,
        prompt=prompt,
        system=system,
        tools=["get_weather"]
    ):
        # Print out the results
        if token:
            cprint(token, "green", end="", flush=True)
        
        if tool_result:
            cprint(f"\n{tool_result}", "yellow")
        
        if stop_reason == StopReason.TOOL_USE:
            for x in range(1, len(response.content)):
                cprint(f"\n{response.content[x].model_dump()}", "cyan", end="", flush=True)
            cprint(f"\n{stop_reason}", "red")
        elif stop_reason:
            cprint(f"\n{stop_reason}", "red")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())