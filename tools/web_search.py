"""
Web search tool - uses Tavily API for real-time web search.
For questions that go beyond the local knowledge base.
"""
from tavily import TavilyClient
from llama_index.core.tools import FunctionTool

from config import config


def web_search(query: str) -> str:
    """
    Search the web for current information using Tavily.
    Use this for questions about recent events, general knowledge,
    or topics not covered in the local document knowledge base.

    Args:
        query: The search query to look up on the web.
    """
    if not config.tavily_api_key:
        return "Web search is not configured. Set TAVILY_API_KEY in .env file."

    try:
        client = TavilyClient(api_key=config.tavily_api_key)
        response = client.search(query=query, max_results=5)

        results = []
        for item in response.get("results", []):
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            results.append(f"**{title}**\n{content}\nSource: {url}")

        if not results:
            return f"No web results found for '{query}'."

        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"


def create_web_search_tool() -> FunctionTool:
    """Create a web search tool for the ReAct agent."""
    return FunctionTool.from_defaults(
        fn=web_search,
        name="web_search",
        description=(
            "Search the web for current, real-time information. "
            "Use this when the question is about recent events, general knowledge, "
            "or topics NOT covered in the local document knowledge base. "
            "Do NOT use this for questions that can be answered from the ingested documents."
        ),
    )
