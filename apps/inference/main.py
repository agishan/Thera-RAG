#!/usr/bin/env python3
"""
Thera-RAG Inference App - Streamlit Chatbot

Main entry point for the inference application.
Handles RAG queries and response generation.
"""

import sys
from pathlib import Path

# Add project root and shared components to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared"))
sys.path.insert(0, str(project_root / "apps" / "inference" / "src"))

# Import the Streamlit app
from src.main import main as streamlit_main

if __name__ == "__main__":
    # Run the Streamlit app
    streamlit_main()