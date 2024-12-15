# flake8: noqa: E501
import asyncio

# Add for print console with color
from termcolor import cprint

from bedrock_llm import AsyncClient, ModelName

# Create prompt
prompt = """Consider a Large Language Model (LLM) with a context-dependent embedding space of dimension $d$. The model is trained on a dataset of $n$ samples, each with a length $L$, using an objective function that encourages the model to predict the next token in the sequence given the previous tokens.
Suppose we want to analyze the distribution of the last hidden state of the model during inference. Specifically, let $\mathbf{h}_T$ denote the last hidden state at time step $T$. We can represent this distribution as a probability measure over the embedding space:
$$p(\mathbf{h}_T) = \frac{\exp\left(-\frac{1}{2\tau^2}|\mathbf{h}_T - \mu|^2\right)}{\sqrt{2\pi\tau^2}}$$ where $\mu$ is the mean of the embedding space, and $\tau$ is a hyperparameter that controls the width of the distribution.
**Question:** Show that under certain regularity conditions on the model weights, the distribution $p(\mathbf{h}_T)$ can be represented as a Gaussian mixture over a finite set of cluster centers. That is, suppose there exist $K$ distinct clusters $\{\mathcal{C}_k\}$ with corresponding cluster centers $\{\mu_k\}$ and weights $\{\pi_k\}$ such that: $$p(\mathbf{h}_T) = \sum_{k=1}^K \pi_k \delta\left(|\mathbf{h}_T - \mu_k|^2\right)$$ where $\delta$ is the Dirac delta function.

**Hint:** Use tools from non-linear algebra and machine learning theory, such as spectral analysis and the Karush-Kuhn-Tucker conditions."""


async def main():
    # Create async client
    client = AsyncClient(region_name="us-east-1", model_name=ModelName.MISTRAL_7B)

    await client.init_session()

    try:
        async for token, response_block in client.generate_async(prompt):
            if response_block:
                cprint(
                    f"\nGeneration stopped: {response_block.stop_reason}", color="red"
                )
                break
            cprint(token, color="green", end="", flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
