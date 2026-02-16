# Agentic RAG - ReAct Agent

A ReAct (Reason + Act) agent that autonomously decides which retrieval tool to use per query. Built with LlamaIndex and powered by AWS Bedrock Claude.

Unlike traditional RAG (which always runs the same pipeline), this agent **reasons about each query** and picks the best retrieval strategy at runtime.

## Architecture

```
User Query
    |
    v
+---------------------------+
|   ReAct Agent (LlamaIndex) |  <-- LLM reasons about what to do
|   Reason -> Act -> Observe  |
+---------------------------+
    |           |          |           |
    v           v          v           v
 vector     file        file         web
 search     search      read        search
(ChromaDB)  (grep)    (full file)  (Tavily)
```

## Tools

| Tool | What It Does | Best For |
|------|-------------|----------|
| `vector_search` | Semantic similarity search over ChromaDB | Conceptual questions about document content |
| `file_search` | Keyword grep across .txt/.docx files | Exact terms, names, numbers |
| `file_read` | Read full file content by name | Getting complete context |
| `web_search` | Tavily API web search | Current events, real-time data |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials and Tavily API key
```

### 3. Ingest documents

Place documents (.pdf, .txt, .docx) in the `data/` directory, then:

```bash
python ingest.py
```

### 4. Run the agent

**Streamlit UI:**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py
```

**MCP Server (for Claude Desktop):**
```bash
python mcp_server.py
```

## MCP Server

The MCP server exposes the ChromaDB knowledge base as tools that any MCP client can use:

- `search_documents` - Semantic vector search
- `list_documents` - List all ingested documents

### Connect Claude Desktop

Add to your Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "agentic-rag-knowledge-base": {
      "command": "python",
      "args": ["/path/to/agentic-rag-react/mcp_server.py"]
    }
  }
}
```

## Docker

```bash
docker compose up --build
# Open http://localhost:8501
```

## Project Structure

```
agentic-rag-react/
├── app.py                 # Streamlit UI
├── main.py                # CLI entry point
├── agent.py               # ReAct agent setup
├── ingest.py              # Document ingestion
├── config.py              # Configuration
├── mcp_server.py          # MCP server
├── tools/
│   ├── vector_search.py   # ChromaDB semantic search
│   ├── file_search.py     # File grep + read
│   └── web_search.py      # Tavily web search
├── data/                  # Documents to ingest
├── chroma_db/             # Vector store (auto-generated)
├── Dockerfile
└── docker-compose.yml
```

## Tech Stack

- **Agent Framework:** LlamaIndex (ReAct agent pattern)
- **LLM:** AWS Bedrock Claude (Converse API)
- **Embeddings:** HuggingFace nomic-embed-text-v1.5
- **Vector Store:** ChromaDB
- **Web Search:** Tavily
- **MCP:** Model Context Protocol SDK
- **UI:** Streamlit

## Part of the RAG Portfolio

| Repo | Focus |
|------|-------|
| [LocalRAG](https://github.com/user/LocalRAG) | Core RAG - hybrid search, semantic caching |
| **agentic-rag-react** | Autonomous ReAct agent with tool selection |
| agentic-rag-orchestrated | Production-grade orchestrated workflows (LangGraph) |
