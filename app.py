"""
Streamlit UI for the Agentic RAG ReAct Agent.
Shows the agent's reasoning traces alongside answers.

Usage: streamlit run app.py
"""
import asyncio
import streamlit as st
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)

from agent import create_agent


st.set_page_config(
    page_title="Agentic RAG - ReAct Agent",
    page_icon="🔍",
    layout="wide",
)

st.title("Agentic RAG - ReAct Agent")
st.caption("LlamaIndex + Bedrock Claude | Vector Search, File Search, Web Search")


@st.cache_resource
def load_agent():
    """Load the agent once and cache it."""
    return create_agent(verbose=False)


def init_session():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "traces" not in st.session_state:
        st.session_state.traces = []


async def run_agent_with_traces(agent, query: str):
    """Run the agent and capture reasoning traces."""
    traces = []
    handler = agent.run(user_msg=query)

    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            traces.append({
                "type": "tool_call",
                "tool": event.tool_name,
                "input": str(event.tool_kwargs) if hasattr(event, "tool_kwargs") else "",
            })
        elif isinstance(event, ToolCallResult):
            output = str(event.tool_output) if hasattr(event, "tool_output") else ""
            traces.append({
                "type": "tool_result",
                "tool": event.tool_name,
                "output": output[:500] + ("..." if len(output) > 500 else ""),
            })

    response = await handler
    return str(response), traces


def display_traces(traces):
    """Display reasoning traces in the sidebar."""
    if not traces:
        return

    for i, trace in enumerate(traces):
        if trace["type"] == "tool_call":
            st.markdown(f"**Step {i//2 + 1}: Called `{trace['tool']}`**")
            if trace.get("input"):
                st.code(trace["input"], language="json")
        elif trace["type"] == "tool_result":
            with st.expander(f"Result from `{trace['tool']}`", expanded=False):
                st.text(trace["output"])


def main():
    init_session()
    agent = load_agent()

    # Sidebar: reasoning traces
    with st.sidebar:
        st.header("Agent Reasoning")
        st.caption("Watch the agent's tool calls and decisions")

        if st.session_state.traces:
            display_traces(st.session_state.traces[-1])
        else:
            st.info("Ask a question to see the agent's reasoning process.")

        st.divider()
        st.markdown("**Available Tools:**")
        st.markdown("- `vector_search` — Semantic search over documents")
        st.markdown("- `file_search` — Keyword grep across files")
        st.markdown("- `file_read` — Read full file content")
        st.markdown("- `web_search` — Tavily web search")

    # Main chat area
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, traces = asyncio.run(run_agent_with_traces(agent, query))
                st.markdown(answer)

        # Store results
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.traces.append(traces)

        # Rerun to update sidebar with latest traces
        st.rerun()


if __name__ == "__main__":
    main()
