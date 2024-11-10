# Bedrock LLM

A Python library for building LLM applications using Amazon Bedrock Provider.

## Features

- Support for Retrieval-Augmented Generation (RAG)
- Support for Agent-based interactions
- Support for Multi-Agent systems (in progress)
- Support for creating workflows, nodes, and event-based systems (comming soon)

## Installation

You can install the Bedrock LLM library using pip:

```
pip install bedrock-llm

This library requires Python 3.7 or later.
```

## Usage

Here's a quick example of how to use the Bedrock LLM library:

```python
from bedrock_llm import LLMClient, ModelName, ModelConfig

# Create a LLM client
client = LLMClient(
    region_name="us-east-1",
    model_name=ModelName.TITAN_PREMIER
)

# Create a configuration for inference parameters
config = ModelConfig(
    temperature=0.1,
    top_p=0.9,
    max_tokens=512
)

# Create a prompt
prompt = "Who are you?"

# Invoke the model and get results
response, stop_reason = client.generate(config, prompt)

# Print out the results
cprint(response.content, "green")
cprint(stop_reason, "red")
```

For more detailed usage instructions and API documentation, please refer to our [documentation](https://github.com/yourusername/bedrock_llm/wiki).

## Requirements

- Python 3.7+
- pydantic>=1.8.0,<2.0.0
- boto3>=1.18.0
- botocore>=1.21.0

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.