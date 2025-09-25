"""
Shared configuration for both ingestion and inference apps
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

def get_shared_config() -> Dict[str, Any]:
    """Get shared configuration used by both apps"""

    # Load environment variables from project root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Shared configuration
    config = {
        # Pinecone configuration (used by both apps)
        'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
        'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'medical-rag-index'),
        'pinecone_namespace': os.getenv('PINECONE_NAMESPACE', 'thera-rag'),
        'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws'),

        # Google AI configuration (used by both apps)
        'google_api_key': os.getenv('GOOGLE_API_KEY'),

        # Embedding/LLM defaults (consistent between ingestion and inference)
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'intfloat/e5-base'),
        'llm_model': os.getenv('LLM_MODEL', 'gemini-2.5-pro'),
        'llm_temperature': float(os.getenv('LLM_TEMPERATURE', '0.1')),
        'llm_max_tokens': int(os.getenv('LLM_MAX_TOKENS', '8192')),
        'retrieval_k': int(os.getenv('RETRIEVAL_K', '15')),

        # Project paths
        'project_root': str(project_root),
        'data_dir': str(project_root / 'data'),
        'output_dir': str(project_root / 'enhanced_output'),

        # Processing configuration
        'chunk_size': 500,
        'chunk_overlap': 50,
        'max_chunks_per_doc': 1000,

        # Metadata schema version
        'metadata_version': 'v2.1_simple'
    }

    return config

def validate_shared_config(config: Dict[str, Any]) -> bool:
    """Validate that required configuration is present"""
    required_keys = ['pinecone_api_key', 'google_api_key']

    for key in required_keys:
        if not config.get(key):
            print(f"[ERROR] Missing required configuration: {key}")
            return False

    return True
