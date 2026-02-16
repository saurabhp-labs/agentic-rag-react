"""
MCP Server - Exposes ChromaDB vector search as MCP tools.
Any MCP client (Claude Desktop, Cursor, etc.) can connect and query the knowledge base.

Usage:
  python mcp_server.py              # Run with stdio transport (for Claude Desktop)
  python mcp_server.py --sse        # Run with SSE transport (for web clients)
"""
import os
import sys
import chromadb
from mcp.server.fastmcp import FastMCP

from config import config

# Initialize MCP server
mcp = FastMCP(
    name="agentic-rag-knowledge-base",
    instructions=(
        "This MCP server provides access to a document knowledge base. "
        "Use the search_documents tool to find information from ingested documents "
        "(employee handbooks, policies, reports, contracts, manuals). "
        "Use list_documents to see what's available."
    ),
)

# Load embedding model once at startup
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _embed_model = HuggingFaceEmbedding(
            model_name=config.embedding_model,
            trust_remote_code=True,
        )
    return _embed_model


def get_chroma_collection():
    """Get the ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path=config.chroma_db_dir)
    return chroma_client.get_or_create_collection("documents")


@mcp.tool()
def search_documents(query: str, num_results: int = 5) -> str:
    """Search the document knowledge base using semantic vector similarity.

    Args:
        query: Natural language query describing what you're looking for.
        num_results: Number of results to return (default 5, max 10).
    """
    num_results = min(max(num_results, 1), 10)

    # Embed the query
    embed_model = get_embed_model()
    query_embedding = embed_model.get_query_embedding(query)

    # Query ChromaDB directly
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant documents found for your query."

    # Format results
    output = []
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        source = metadata.get("file_name", metadata.get("source", "unknown"))
        similarity = round(1 - distance, 3) if distance else "N/A"
        output.append(f"**Result {i+1}** (similarity: {similarity}, source: {source})\n{doc}")

    return "\n\n---\n\n".join(output)


@mcp.tool()
def list_documents() -> str:
    """List all documents available in the knowledge base with their chunk counts."""
    collection = get_chroma_collection()

    # Get all metadata to find unique sources
    all_data = collection.get(include=["metadatas"])
    if not all_data["metadatas"]:
        return "No documents in the knowledge base. Run `python ingest.py` first."

    # Count chunks per source file
    source_counts = {}
    for meta in all_data["metadatas"]:
        source = meta.get("file_name", meta.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1

    total = collection.count()
    lines = [f"Knowledge base: {total} chunks from {len(source_counts)} document(s)\n"]
    for source, count in sorted(source_counts.items()):
        lines.append(f"  - {source} ({count} chunks)")

    return "\n".join(lines)


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
