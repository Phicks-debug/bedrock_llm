import asyncio

# Add for print console with color
from termcolor import cprint

from bedrock_llm import AsyncClient, ModelName, RetryConfig
from bedrock_llm.schema import MessageBlock


# Function to handle user input asynchronously
async def get_user_input(placeholder: str) -> str:
    return input(placeholder)


async def chat_with_titan():
    """
    Read more about prompting for Titan Text:
    https://d2eo22ngex1n9g.cloudfront.net/Documentation/User+Guides/Titan/Amazon+Titan+Text+Prompt+Engineering+Guidelines.pdf
    """

    # Initialize the client
    client = AsyncClient(
        region_name="us-east-1",
        model_name=ModelName.TITAN_PREMIER,
        memory=[],
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0),
    )

    await client.init_session()

    while True:
        input_prompt = await get_user_input("Enter a prompt: ")
        prompt = MessageBlock(role="user", content=input_prompt)

        # Simple text generation
        async for token, stop_reason, _ in client.generate_async(prompt=prompt):
            if stop_reason is None:
                cprint(token, color="green", end="", flush=True)
            else:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")

        # Check for bye bye
        if input_prompt.lower() == "/bye":
            break

    await client.close()


async def chat_with_claude():
    # Initialize the client with memory enable
    client = AsyncClient(
        region_name="us-west-2",
        model_name=ModelName.CLAUDE_3_5_SONNET,
        memory=[],
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0),
    )

    await client.init_session()

    while True:
        # Receive user input
        input_prompt = await get_user_input("Enter a prompt: ")
        prompt = MessageBlock(role="user", content=input_prompt)

        # Simple text generation
        async for token, stop_reason, _ in client.generate_async(prompt=prompt):
            if stop_reason is None:
                cprint(token, color="green", end="", flush=True)
            else:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")

        # Check for bye bye
        if input_prompt.lower() == "/bye":
            break

    await client.close()


async def chat_with_llama():
    # Initialize the client
    client = AsyncClient(
        region_name="us-west-2",
        model_name=ModelName.LLAMA_3_2_90B,
        memory=[],
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0),
    )

    await client.init_session()

    while True:
        # Get user input asynchronously
        input = await get_user_input("Enter a prompt: ")
        prompt = MessageBlock(role="user", content=input)

        # Generate a response from the model
        async for chunk, stop_reason, _ in client.generate_async(prompt=prompt):
            if stop_reason is None:
                cprint(chunk, color="green", end="", flush=True)
            else:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")

        if input.lower() == "/bye":
            break

    await client.close()


async def chat_with_mistral():
    # Initialize the client
    client = AsyncClient(
        region_name="us-west-2",
        model_name=ModelName.MISTRAL_7B,
        memory=[],
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0),
    )

    await client.init_session()

    # Predefine the system message to avoid repetition
    system = "You are a helpful AI assistant."

    while True:
        prompt = MessageBlock(
            role="user", content=await get_user_input("Enter a prompt: ")
        )

        # Generate a response from the model
        async for chunk, stop_reason, _ in client.generate_async(
            prompt=prompt, system=system
        ):
            if stop_reason is None:
                cprint(chunk, color="green", end="", flush=True)
            else:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")

        if prompt.content.lower() == "/bye":
            break

    await client.close()


async def chat_with_jamba():
    # Initialize the client
    client = AsyncClient(
        region_name="us-east-1",
        model_name=ModelName.JAMBA_1_5_MINI,
        memory=[],
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0),
    )

    await client.init_session()

    while True:
        input = await get_user_input("Enter a prompt: ")
        prompt = MessageBlock(role="user", content=input)

        # Generate a response from the model
        async for chunk, stop_reason, _ in client.generate_async(prompt=prompt):
            if stop_reason is None:
                cprint(chunk, color="green", end="", flush=True)
            else:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")

        # Check for bye bye
        if input.lower() == "/bye":
            break

    await client.close()


if __name__ == "__main__":
    model_selection = input(
        """Select (1 for Claude,
2 for Titan, 3 for Llama, 4 for Mistral and 5 for Jamba): """
    )
    if model_selection == "1":
        asyncio.run(chat_with_claude())
    elif model_selection == "2":
        asyncio.run(chat_with_titan())
    elif model_selection == "3":
        asyncio.run(chat_with_llama())
    elif model_selection == "4":
        asyncio.run(chat_with_mistral())
    elif model_selection == "5":
        asyncio.run(chat_with_jamba())
