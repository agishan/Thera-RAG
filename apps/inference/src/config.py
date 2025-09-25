import os
import streamlit as st
from pathlib import Path

# Load .env file from the same directory as this config file
def load_env_file():
    """Load environment variables from .env file in the same directory"""
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def get_secret(key, default=None):
    """Get secret from Streamlit secrets or environment variables"""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        secret = os.getenv(key, default)
        if secret:
            print(f"Secret {key} not found in Streamlit secrets, using environment variable")
        else:
            print(f"!!!Secret {key} not found in Streamlit secrets or environment variable!!!")
        return secret

def get_config():
    """Get all configuration settings"""
    # Load .env file first
    load_env_file()
    shared = get_shared_config()
    
    config = {
        # Required
        'pinecone_api_key': get_secret("PINECONE_API_KEY"),
        'google_api_key': get_secret("GOOGLE_API_KEY"),
        
        # Google Sheets (optional)
        'google_sheets_spreadsheet_id': get_secret("GOOGLE_SHEETS_SPREADSHEET_ID"),
        'google_sheets_creds_json': get_secret("GOOGLE_SHEETS_CREDS_JSON"),
        
        # Pinecone settings
        'pinecone_index_name': get_secret("PINECONE_INDEX_NAME", shared.get('pinecone_index_name')),
        'pinecone_namespace': get_secret("PINECONE_NAMESPACE", shared.get('pinecone_namespace')),
        
        # Model settings
        'embedding_model': get_secret("EMBEDDING_MODEL", shared.get('embedding_model')),
        'llm_model': get_secret("LLM_MODEL", shared.get('llm_model')),
        'llm_temperature': float(get_secret("LLM_TEMPERATURE", str(shared.get('llm_temperature', 0.1)))),
        'llm_max_tokens': int(get_secret("LLM_MAX_TOKENS", str(shared.get('llm_max_tokens', 8192)))),
        'retrieval_k': int(get_secret("RETRIEVAL_K", str(shared.get('retrieval_k', 15)))),
        
        # Sheets settings
        'sheets_name': get_secret("SHEETS_NAME", "Chat_Logs"),
    }
    
    # Add computed flags
    config['google_sheets_enabled'] = bool(
        config['google_sheets_spreadsheet_id'] and 
        config['google_sheets_creds_json']
    )
    
    return config

def validate_config(config):
    """Validate required configuration"""
    required = ['pinecone_api_key', 'google_api_key']
    missing = [key for key in required if not config.get(key)]
    
    if missing:
        st.error(f"❌ Missing required configuration: {', '.join(missing)}")
        st.stop()
from pathlib import Path
try:
    from shared.config.settings import get_shared_config
except Exception:
    # Fallback if import path differs in runtime
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'shared'))
    from config.settings import get_shared_config
