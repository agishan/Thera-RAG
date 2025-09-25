#!/usr/bin/env python3
"""
Test script to run pipeline non-interactively
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "scripts"))

from interactive_pipeline import InteractivePipeline

# Mock the user input to always return 'y'
import builtins
original_input = builtins.input
builtins.input = lambda prompt: 'y'

try:
    pipeline = InteractivePipeline(text_only=True, save_stages=False)
    results = pipeline.process_document_interactive("data/vha-guideline.pdf")
    print("SUCCESS!")
    print(f"Completed stages: {results.get('stages_completed', [])}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore original input
    builtins.input = original_input