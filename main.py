"""
CLI entry point - interactive query loop with the ReAct agent.
Usage: python main.py
"""
import asyncio
from agent import create_agent


async def run_query(agent, query: str):
    """Run a single query through the agent."""
    response = await agent.run(user_msg=query)
    return response


def main():
    print("=" * 60)
    print("  Agentic RAG - ReAct Agent (LlamaIndex + Bedrock)")
    print("=" * 60)
    print("Loading agent...")

    agent = create_agent(verbose=True)

    print("Agent ready. Type your questions (or 'quit' to exit).\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        response = asyncio.run(run_query(agent, query))
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
