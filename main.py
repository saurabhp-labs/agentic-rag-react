"""
CLI entry point - interactive query loop with the ReAct agent.
Usage: python main.py
"""
from agent import create_agent


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
        response = agent.chat(query)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
