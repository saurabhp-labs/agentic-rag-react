"""
ReAct Agent - LlamaIndex ReAct agent powered by Bedrock Claude.
Reasons about queries and decides which tools to call.
"""
from llama_index.core import Settings
from llama_index.core.agent import AgentWorkflow
from llama_index.llms.bedrock_converse import BedrockConverse

from config import config
from tools.vector_search import create_vector_search_tool
from tools.file_search import create_file_search_tools
from tools.web_search import create_web_search_tool


SYSTEM_PROMPT = """\
You are a research assistant with access to a local document knowledge base and the web.
You MUST use tools to answer questions — never answer from your own knowledge alone.

TOOL SELECTION (choose in this priority order):

1. **vector_search** — ALWAYS try this FIRST for any question that might be answered by \
the local documents. It searches semantically across all ingested documents (employee handbooks, \
policies, reports, contracts, manuals, etc.).

2. **file_search** — Use when looking for an exact term, name, number, or phrase. \
Searches .txt and .docx files by keyword matching.

3. **file_read** — Read the full content of a specific file by name. \
Use after file_search to get complete context.

4. **web_search** — ONLY use this as a LAST RESORT when the question is clearly about \
current events, real-time data, or topics that could NOT be in the local documents.

Rules:
- ALWAYS try vector_search or file_search before web_search.
- Answer DIRECTLY with specific details from the results. Do NOT summarize what you did.
- If results are insufficient, try a different tool before giving up.
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

    tools = [
        create_vector_search_tool(),
        *create_file_search_tools(),
        create_web_search_tool(),
    ]

    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=verbose,
    )
