"""
Configuration using Pydantic Settings.
Loads from environment variables or .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # AWS Bedrock
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # Bedrock Models
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"

    # Embeddings (local HuggingFace - consistent with LocalRAG)
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # Paths
    data_dir: str = "./data"
    chroma_db_dir: str = "./chroma_db"

    # Tavily Web Search
    tavily_api_key: str = ""

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50


config = Config()
