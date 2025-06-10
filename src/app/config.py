import os
import streamlit as st

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
    config = {
        # Required
        'pinecone_api_key': get_secret("PINECONE_API_KEY"),
        'google_api_key': get_secret("GOOGLE_API_KEY"),
        
        # Google Sheets (optional)
        'google_sheets_spreadsheet_id': get_secret("GOOGLE_SHEETS_SPREADSHEET_ID"),
        'google_sheets_creds_json': get_secret("GOOGLE_SHEETS_CREDS_JSON"),
        
        # Pinecone settings
        'pinecone_index_name': get_secret("PINECONE_INDEX_NAME", "medical-rag-index"),
        'pinecone_namespace': get_secret("PINECONE_NAMESPACE", "thera-rag"),
        
        # Model settings
        'embedding_model': get_secret("EMBEDDING_MODEL", "models/embedding-001"),
        'llm_model': get_secret("LLM_MODEL", "gemini-1.5-pro"),
        'llm_temperature': float(get_secret("LLM_TEMPERATURE", "0.1")),
        'llm_max_tokens': int(get_secret("LLM_MAX_TOKENS", "8192")),
        'retrieval_k': int(get_secret("RETRIEVAL_K", "30")),
        
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