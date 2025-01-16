import asyncio

# Add for print console with color
from termcolor import cprint

from bedrock_llm import AsyncClient, ModelConfig, ModelName, RetryConfig
from bedrock_llm.schema import MessageBlock


async def main():
    # Prompt format
    prompt = MessageBlock(role="user", content="Who are you and what can you do?")
    system = "You are a helpful AI. Answer only in Vietnamese"
    config = ModelConfig(temperature=0, max_tokens=512, top_p=1, top_k=70)
    retry_config = RetryConfig(max_retries=3, retry_delay=0.5)

    # Using Llama model
    for model in [
        ModelName.LLAMA_3_1_8B,
        ModelName.LLAMA_3_1_70B,
        ModelName.LLAMA_3_1_405B,
        ModelName.LLAMA_3_2_1B,
        ModelName.LLAMA_3_2_3B,
        ModelName.LLAMA_3_2_11B,
        ModelName.LLAMA_3_2_90B,
        ModelName.LLAMA_3_3_70B,
    ]:
        async with AsyncClient(
            region_name="us-west-2", model_name=model, retry_config=retry_config
        ) as llama_client:
            print("Model: ", model)
            async for token, res in llama_client.generate_async(
                config=config, prompt=prompt, system=system
            ):
                if res:
                    cprint(f"\nGeneration stopped: {res.stop_reason}", color="red")
                    break
                cprint(token, color="yellow", end="", flush=True)

    for model in [
        ModelName.NOVA_MICRO,
        ModelName.NOVA_LITE,
        ModelName.NOVA_PRO,
    ]:
        async with AsyncClient(
            region_name="us-east-1", model_name=model, retry_config=retry_config
        ) as titan_client:
            print("Model: ", model)
            async for token, res in titan_client.generate_async(
                config=config, prompt=prompt, system=system
            ):
                if res:
                    cprint(f"\nGeneration stopped: {res.stop_reason}", color="red")
                    break
                cprint(token, color="light_blue", end="", flush=True)

    # Using Claude model
    for model in [
        ModelName.CLAUDE_3_HAIKU,
        ModelName.CLAUDE_3_5_HAIKU,
        ModelName.CLAUDE_3_5_SONNET,
    ]:
        async with AsyncClient(
            region_name="us-west-2",
            model_name=model,
            retry_config=retry_config,
        ) as claude_client:
            print("Model: ", model)
            async for token, res in claude_client.generate_async(
                config=config, prompt=prompt, system=system
            ):
                if token:
                    cprint(token, color="green", end="", flush=True)
                if res:
                    cprint(f"\nGeneration stopped: {res.stop_reason}", color="red")
                    break

    # Using Jamba model
    for model in [ModelName.JAMBA_1_5_MINI, ModelName.JAMBA_1_5_LARGE]:
        async with AsyncClient(
            region_name="us-east-1",
            model_name=model,
            retry_config=retry_config,
        ) as jamba_client:
            print("Model: ", model)
            async for token, res in jamba_client.generate_async(
                config=config, prompt=prompt, system=system
            ):
                if res:
                    cprint(f"\nGeneration stopped: {res.stop_reason}", color="red")
                    break
                if token:
                    cprint(token, color="grey", end="", flush=True)

    # Using Mistral 7B Instruct model
    async with AsyncClient(
        region_name="us-west-2",
        model_name=ModelName.MISTRAL_7B,
        retry_config=retry_config,
    ) as mistral_client:
        print("Model: ", ModelName.MISTRAL_7B)
        async for token, res in mistral_client.generate_async(
            config=config, prompt=prompt, system=system
        ):
            if res:
                cprint(f"\nGeneration stopped: {res.stop_reason}", color="red")
                break
            cprint(token, color="magenta", end="", flush=True)

    # Using Mistral Large V2 model
    async with AsyncClient(
        region_name="us-west-2",
        model_name=ModelName.MISTRAL_LARGE_2,
        retry_config=retry_config,
    ) as mistral_lg_client:
        print("Model: ", ModelName.MISTRAL_LARGE_2)
        async for token, res in mistral_lg_client.generate_async(
            config=config, prompt=prompt, system=system
        ):
            if res:
                cprint(f"\nGeneration stopped: {res.stop_reason}", color="red")
                break
            cprint(token, color="blue", end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
