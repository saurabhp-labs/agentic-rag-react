"""
ReAct Agent - LlamaIndex ReAct agent powered by Bedrock Claude.
Reasons about queries and decides which tools to call.
"""
from llama_index.core import Settings
from llama_index.core.agent import AgentWorkflow
from llama_index.llms.bedrock_converse import BedrockConverse

from config import config
from tools.vector_search import create_vector_search_tool


SYSTEM_PROMPT = """\
You are a knowledgeable research assistant with access to a document knowledge base.

When answering questions:
1. Use the vector_search tool to find relevant information from the knowledge base.
2. Answer the user's question DIRECTLY using specific details from the retrieved documents.
3. Do NOT describe what you found — instead, present the actual information to the user.
4. If the search results don't contain enough information, say so clearly.
5. Be concise and direct.
"""


def get_llm():
    """Create Bedrock Claude LLM instance using the Converse API."""
    return BedrockConverse(
        model=config.bedrock_model_id,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region,
        temperature=0.3,
        max_tokens=1024,
    )


def create_agent(verbose: bool = True) -> AgentWorkflow:
    """
    Create a ReAct agent workflow with Bedrock Claude LLM and vector search tool.

    Args:
        verbose: If True, print the agent's reasoning steps.
    """
    llm = get_llm()

    # Set global LLM so query engines use Bedrock instead of defaulting to OpenAI
    Settings.llm = llm

    tools = [create_vector_search_tool()]

    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=verbose,
    )
