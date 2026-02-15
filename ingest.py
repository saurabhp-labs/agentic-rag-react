"""
Document ingestion - reads files from data/ directory, chunks, embeds, and stores in ChromaDB.
Usage: python ingest.py
"""
import os
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import config


def get_embedding_model():
    """Load the HuggingFace embedding model (same as LocalRAG)."""
    return HuggingFaceEmbedding(
        model_name=config.embedding_model,
        trust_remote_code=True,
    )


def ingest_documents():
    """Read documents from data/ dir, chunk, embed, and store in ChromaDB."""
    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    files = [f for f in os.listdir(data_dir) if not f.startswith(".")]
    if not files:
        print(f"No files found in {data_dir}")
        return

    print(f"Found {len(files)} file(s) in {data_dir}:")
    for f in files:
        print(f"  - {f}")

    # Configure LlamaIndex settings
    Settings.embed_model = get_embedding_model()
    Settings.chunk_size = config.chunk_size
    Settings.chunk_overlap = config.chunk_overlap

    # Read documents
    print("\nReading documents...")
    reader = SimpleDirectoryReader(input_dir=data_dir)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document section(s)")

    # Chunk documents
    splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks")

    # Set up ChromaDB
    os.makedirs(config.chroma_db_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=config.chroma_db_dir)
    chroma_collection = chroma_client.get_or_create_collection("documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index (embeds and stores)
    print("Embedding and storing in ChromaDB...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )

    count = chroma_collection.count()
    print(f"\nDone! ChromaDB now has {count} vectors in '{config.chroma_db_dir}'")
    return index


if __name__ == "__main__":
    ingest_documents()
