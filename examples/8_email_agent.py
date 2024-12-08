# flake8: noqa: E501
import boto3
from botocore.exceptions import ClientError
from termcolor import cprint

from bedrock_llm import Agent, ModelConfig, ModelName, RetryConfig, StopReason
from bedrock_llm.schema import InputSchema, PropertyAttr, ToolMetadata, MessageBlock
from bedrock_llm.monitor import monitor_async


# Define the runtime for knowledge base
runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
ses_client = boto3.client("ses", region_name="ap-southeast-1")

# Create a LLM client
agent = Agent(
    region_name="us-west-2",
    model_name=ModelName.MISTRAL_LARGE_2,
    retry_config=RetryConfig(max_attempts=3),
)

# Create system prompt
system = """You are an sending email Agent

This is my email: an.tq@techxcorp.com
"""

# Create a configuration for inference parameters
config = ModelConfig(temperature=0.7, top_p=0.9, max_tokens=2048)

# Create tool definition for Knowledge Base
send_email_tool = ToolMetadata(
    name="send_email",
    description="Send an email using AWS SES.",
    input_schema=InputSchema(
        type="object",
        properties={
            "recipient_email": PropertyAttr(
                type="string", description="The recipient's email address."
            ),
            "sender_email": PropertyAttr(
                type="string", description="The sender's email address."
            ),
            "subject": PropertyAttr(
                type="string", description="The subject of the email."
            ),
            "body": PropertyAttr(type="string", description="The body of the email."),
        },
        required=["email", "subject", "body"],
    ),
)


# Create a function for sending email from outlook
@Agent.tool(send_email_tool)
@monitor_async
async def send_email(recipient_email: str, sender_email: str, subject: str, body: str):
    try:
        # Prepare the email
        email_message = {
            "Source": sender_email,
            "Destination": {"ToAddresses": [recipient_email]},
            "Message": {
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        }

        # Send the email
        response = ses_client.send_email(**email_message)

        return {
            "success": True,
            "message": f"Email sent successfully! MessageId: {response['MessageId']}",
            "status_code": 200,
        }

    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        return {
            "success": False,
            "message": f"Failed to send email: {error_message}",
            "status_code": 500,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"An unexpected error occurred: {str(e)}",
            "status_code": 500,
        }


async def main():

    # Create user prompt
    prompt = MessageBlock(
        role="user",
        content="Can you you send an email to my colleage 'mineran2003@gmail.com' to ask him when he can finish the function in python that I asked him to it.",
    )

    await agent.init_session()

    # Invoke the model and get results
    async for (
        token,
        stop_reason,
        response,
        tool_result,
    ) in agent.generate_and_action_async(
        config=config,
        prompt=prompt,
        system=system,
        tools=["send_email"],
    ):
        # Print out the results
        if token:
            cprint(token, "green", end="", flush=True)

        # Print out the tool result
        if tool_result:
            for x in tool_result:
                cprint(f"\n{x.content}", "yellow", flush=True)

        # Print out the function that need to use
        if stop_reason == StopReason.TOOL_USE:
            for x in response.tool_calls:
                cprint(f"\n{x.model_dump()}", "cyan", end="", flush=True)
            cprint(f"\n{stop_reason}", "red", flush=True)
        elif stop_reason:
            cprint(f"\n{stop_reason}", "red", flush=True)

    await agent.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
