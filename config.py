import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the LlamaIndex-based system."""
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY_HERE")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "YOUR_PINECONE_ENVIRONMENT_HERE")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "policy-index")
    MAX_EMBEDDING_BATCH_SIZE = 100 

    DOCUMENTS_PATH: str = "./data/policies"
    
    LOG_LEVEL: str = "INFO"
    CHUNK_SIZE: int = 1200 
    CHUNK_OVERLAP: int = 300
    TOP_K_RESULTS: int = 6

    # Neo4j Graph configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    ENABLE_GRAPH_STORE: bool = os.getenv("ENABLE_GRAPH_STORE", "false").lower() in {"1", "true", "yes"}

    # Graphiti sidecar (episodes) configuration
    GRAPHITI_ENABLED: bool = os.getenv("GRAPHITI_ENABLED", "false").lower() in {"1", "true", "yes"}
    GRAPHITI_SERVER_URL: str = os.getenv("GRAPHITI_SERVER_URL", "")
    GRAPHITI_API_KEY: str = os.getenv("GRAPHITI_API_KEY", "")
    GRAPHITI_WORKSPACE: str = os.getenv("GRAPHITI_WORKSPACE", "default")
    GRAPHITI_EPISODE_DIR: str = os.getenv("GRAPHITI_EPISODE_DIR", "./graphiti_episodes")
    
    @classmethod
    def validate(cls) -> None:
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError("OPENAI_API_KEY is not set. Please set the environment variable or update the config.py file.")
        if not cls.PINECONE_API_KEY or cls.PINECONE_API_KEY == "YOUR_PINECONE_API_KEY_HERE":
            raise ValueError("PINECONE_API_KEY is not set. Please set the environment variable or update the config.py file.")
        if not cls.PINECONE_ENVIRONMENT or cls.PINECONE_ENVIRONMENT == "YOUR_PINECONE_ENVIRONMENT_HERE":
            raise ValueError("PINECONE_ENVIRONMENT is not set. Please set the environment variable or update the config.py file.")
        if not cls.NEO4J_URI:
            raise ValueError("NEO4J_URI is not set. Please set the environment variable or update the config.py file.")
        if not cls.NEO4J_USERNAME:
            raise ValueError("NEO4J_USERNAME is not set. Please set the environment variable or update the config.py file.")
        if not cls.NEO4J_PASSWORD or cls.NEO4J_PASSWORD == "password":
            raise ValueError("NEO4J_PASSWORD is not set. Please set the environment variable or update the config.py file.")
        # Graphiti optional; validate minimally if enabled
        if cls.GRAPHITI_ENABLED and not cls.GRAPHITI_SERVER_URL:
            raise ValueError("GRAPHITI_SERVER_URL must be set when GRAPHITI_ENABLED=true")

config = Config()