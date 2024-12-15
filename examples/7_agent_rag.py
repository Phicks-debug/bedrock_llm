# flake8: noqa: E501
import xml.etree.ElementTree as ET
from typing import Any, Dict

import boto3
import wikipedia
from termcolor import cprint

from bedrock_llm import Agent, ModelConfig, ModelName, RetryConfig, StopReason
from bedrock_llm.schema import InputSchema, MessageBlock, ToolMetadata
from bedrock_llm.monitor import monitor_sync

# Define the runtime for knowledge base
runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

# Create a LLM client
agent = Agent(
    region_name="us-east-1",
    model_name=ModelName.JAMBA_1_5_LARGE,
    retry_config=RetryConfig(max_attempts=3),
)

# Create a configuration for inference parameters
config = ModelConfig(temperature=0.1, top_p=0.9, max_tokens=2048)

# Create tool definition for Knowledge Base
knowledge_base_retrieve_tool = ToolMetadata(
    name="knowledge_base_retrieve",
    description="Search the knowledge base for information related to Eximbank website. The knowledge base contain all the information crawed from the Eximbank page. It included 'Khách hàng cá nhân', 'Khách hàng doanh nghiệp', 'Khách hàng ưu tiên', 'Dịch vụ ngân hàng số'. The knowledge base also contain the information about 'Nhà đầu tư', 'Thank lý tài sản', 'Hỗ trợ', and các 'Lãi xuất' trong ngân hàng.",
    input_schema=InputSchema(
        type="object",
        properties={
            "query": {
                "type": "string",
                "description": "The query to search for in the knowledge base",
            }
        },
        required=["query"],
    ),
)
# Create tool definition for searching on Wikipedia
wikipedia_retrieve_tool = ToolMetadata(
    name="wikipedia_retrieve",
    description="Search on wikipedia page",
    input_schema=InputSchema(
        type="object",
        properties={
            "query": {
                "type": "string",
                "description": "The query of name, event or object to search for in wikipedia page",
            }
        },
        required=["query"],
    ),
)


# Create a function for retrieve knowledge base from AWS
@Agent.tool(knowledge_base_retrieve_tool)
@monitor_sync
def knowledge_base_retrieve(query: str):
    kwargs = {
        "knowledgeBaseId": "VSR83TL8CR",  # Insert your knowledge base ID
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {"numberOfResults": 25}
        },
        "retrievalQuery": {"text": query},
    }

    # Run boto3 call in a thread pool since it's blocking
    result = runtime.retrieve(**kwargs)
    return build_context_kb_prompt(result)


# Create a function for retrieve information on wikipedia
@Agent.tool(wikipedia_retrieve_tool)
@monitor_sync
def wikipedia_retrieve(query: str) -> Dict[str, Any]:
    def get_page_info(title: str) -> Dict[str, Any]:
        """
        Helper function to asynchronously get page information for a single title.
        """
        try:
            # Run synchronous Wikipedia operations in a separate thread
            page = wikipedia.page(title)
            summary = wikipedia.summary(title, sentences=2)

            return {
                "summary": summary,
                "content": page.content[:500],  # First 500 characters of content
            }
        except wikipedia.exceptions.DisambiguationError as e:
            return {
                "summary": f"Disambiguation page. Possible matches: {', '.join(e.options[:5])}",
                "content": None,
            }
        except wikipedia.exceptions.PageError:
            return None

    try:
        # Search for pages matching the query
        search_results = wikipedia.search(query, results=5)

        # Create tasks for all page info retrievals
        result = []
        for title in search_results:
            result.append(get_page_info(title))

        return "/n/n".join(result)
    except Exception as e:
        return {"error": str(e)}


# Reformat the knowledge base result for more LLM
def build_context_kb_prompt(
    retrieved_json_file, min_relevant_percentage: float = 0.5, debug=False
):
    if not retrieved_json_file:
        return ""

    documents = ET.Element("documents")

    if retrieved_json_file["ResponseMetadata"]["HTTPStatusCode"] != 200:
        documents.text = (
            "Error in getting data source from knowledge base. No context is provided"
        )
    else:
        body = retrieved_json_file["retrievalResults"]
        for i, context_block in enumerate(body):
            if context_block["score"] < min_relevant_percentage:
                break
            document = ET.SubElement(documents, "document", {"index": str(i + 1)})
            source = ET.SubElement(document, "source")
            content = ET.SubElement(document, "document_content")
            source.text = iterate_through_location(context_block["location"])
            content.text = context_block["content"]["text"]

    return ET.tostring(documents, encoding="unicode", method="xml")


def iterate_through_location(location: dict):
    # Optimize by stopping early if uri or url is found
    for loc_data in location.values():
        if isinstance(loc_data, dict):
            uri = loc_data.get("uri")
            if uri:
                return uri
            url = loc_data.get("url")
            if url:
                return url
    return None


@monitor_sync
def main():
    # Create system prompt
    system = """- You are an RAG Agent for Eximbank in accessing Eximbank website to craw data and informations related to customer services and bank products.
- Your job is pass the user question to the retrieval tool to get the most relevant documents, and then answer the user's question using only those documents from the search results.
- You must use the wikipedia page to searching for other irrelevant information of Eximbank, or anything related to Vietnamese Government to compare it with the search from Eximbank database.
If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion."""

    # Create user prompt
    prompt = MessageBlock(
        role="user", content="How impact Eximbank is for the economic of Vietname?"
    )

    # Invoke the model
    result = agent.generate_and_action(
        config=config,
        prompt=prompt,
        system=system,
        tools=[knowledge_base_retrieve, wikipedia_retrieve],
    )

    # Parse the result the model and get results
    for response_block, tool_result in result:

        # Print out the tool result
        if tool_result:
            for x in tool_result:
                cprint(f"\n{x}", "yellow", flush=True)

        if response_block:
            cprint(response_block.message.content, "green")
            # Print out the function that need to use
            if response_block.stop_reason == StopReason.TOOL_USE:
                for x in response_block.message.tool_calls:
                    cprint(f"\n{x.model_dump()}", "cyan", end="", flush=True)
                cprint(f"\n{response_block.stop_reason}", "red", flush=True)
            elif response_block.stop_reason:
                cprint(f"\n{response_block.stop_reason}", "red", flush=True)


if __name__ == "__main__":
    main()
