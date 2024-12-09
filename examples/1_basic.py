# Add for print console with color
from termcolor import cprint

from bedrock_llm import Client, ModelConfig, ModelName
from bedrock_llm.schema import MessageBlock

# Create a LLM client
client = Client(
    region_name="us-east-1",
    model_name=ModelName.TITAN_PREMIER,
)

# Create a configuration for inference parameters
config = ModelConfig(temperature=0.1, top_p=0.9, max_tokens=512)

# Create a prompt
prompt = "Who are you?"

# Invoke the model and get results
response = client.generate(prompt=prompt, config=config)

# Print out the results
cprint(response.message.content, "green")
cprint(response.stop_reason.name, "red")

# Create a system prompt with a list of examples
system = "Your name is Bob, you live in Paris"
prompt = [
    MessageBlock(role="user", content="What is the capital of France?"),
    MessageBlock(role="assistant", content="The capital of France is Paris."),
    MessageBlock(role="user", content="What is the capital of Germany?"),
    MessageBlock(role="assistant", content="The capital of Germany is Berlin."),
    MessageBlock(
        role="user", content="What is your name and what is the capital of Italy?"
    ),
]

# Invoke the model and get results
response = client.generate(prompt, system, config=config)

# Print out the results
cprint(response.message.content, "green")
cprint(response.stop_reason.name, "red")
