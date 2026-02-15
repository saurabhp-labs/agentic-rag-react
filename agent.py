"""
ReAct Agent - LlamaIndex ReAct agent powered by Bedrock Claude.
Reasons about queries and decides which tools to call.
"""
from llama_index.core.agent import ReActAgent
from llama_index.llms.bedrock import Bedrock

from config import config
from tools.vector_search import create_vector_search_tool


SYSTEM_PROMPT = """\
You are a knowledgeable research assistant with access to a document knowledge base.

When answering questions:
1. Use the vector_search tool to find relevant information from the knowledge base.
2. Base your answers on the retrieved documents. Cite specific details.
3. If the search results don't contain enough information, say so clearly.
4. Be concise and direct in your answers.
"""


def create_agent(verbose: bool = True) -> ReActAgent:
    """
    Create a ReAct agent with Bedrock Claude LLM and vector search tool.

    Args:
        verbose: If True, print the agent's reasoning steps (Thought/Action/Observation).
    """
    llm = Bedrock(
        model=config.bedrock_model_id,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region,
        temperature=0.3,
        max_tokens=1024,
    )

    tools = [create_vector_search_tool()]

    return ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=verbose,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=5,
    )
