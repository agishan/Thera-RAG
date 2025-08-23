"""
Streamlit App Entry Point
This file serves as the main entry point for Streamlit deployment.
"""

import sys
import os

# Add the src/app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'app'))

# Import and run the main app
if __name__ == "__main__":
    from main import main
    main()
