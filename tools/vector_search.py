"""
Vector search tool - wraps ChromaDB as a LlamaIndex tool for the ReAct agent.
"""
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import config


def load_index() -> VectorStoreIndex:
    """Load existing ChromaDB index."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.embedding_model,
        trust_remote_code=True,
    )

    chroma_client = chromadb.PersistentClient(path=config.chroma_db_dir)
    chroma_collection = chroma_client.get_or_create_collection("documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return VectorStoreIndex.from_vector_store(vector_store)


def create_vector_search_tool() -> QueryEngineTool:
    """Create a vector search tool the ReAct agent can call."""
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=5)

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="vector_search",
            description=(
                "Search through ingested documents using semantic vector similarity. "
                "Use this tool to find relevant information from the knowledge base. "
                "Input should be a natural language query describing what you're looking for."
            ),
        ),
    )
