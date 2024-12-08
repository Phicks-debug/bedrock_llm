from bedrock_llm.monitor import monitor_sync
from bedrock_llm import Client, ModelName
from bedrock_llm.schema import MessageBlock


# Keep track of the function with monitor_sync decorator
@monitor_sync
def inovke_chat_bot_A(prompt: str) -> str:

    # Create a message block for the user prompt
    _prompt = MessageBlock("user", prompt)

    # Create a llm A with memory
    llm = Client(region_name="us-west-1", model_name=ModelName.LLAMA_3_2_3B, memory=[])

    # invoke the llm A and return response
    response, _ = llm.generate(prompt=_prompt)
    print(response.content)
    return response.content


@monitor_sync
def invoke_chat_bot_B(prompt: str) -> str:

    # Create a message block for the user prompt
    _prompt = MessageBlock("user", prompt)

    # Create a llm B with memory
    llm = Client(region_name="us-east-1", model_name=ModelName.LLAMA_3_2_90B)

    # invoke the llm B and return response
    response, _ = llm.generate(prompt=_prompt)
    print(response.content)
    return response.content


if __name__ == "__main__":
    answer1 = inovke_chat_bot_A("Hello")
    answer2 = invoke_chat_bot_B(answer1)
    answer3 = inovke_chat_bot_A(answer2)
    invoke_chat_bot_B(answer3)
