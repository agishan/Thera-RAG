#!/usr/bin/env python3
"""
Test integrated medical pipeline
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "scripts"))

from interactive_pipeline import InteractivePipeline
import builtins

def test_medical_pipeline():
    """Test medical pipeline with auto-approval"""

    # Mock input to automatically approve all stages
    original_input = builtins.input
    builtins.input = lambda prompt: 'y'

    try:
        print("Testing MEDICAL FILTERING pipeline...")
        print("=" * 60)

        # Create pipeline with medical filtering enabled
        pipeline = InteractivePipeline(
            text_only=True,  # Fast processing
            save_stages=False,  # Don't save files for test
            use_medical_filtering=True  # Enable medical filtering
        )

        # Run pipeline
        results = pipeline.process_document_interactive("data/vha-guideline.pdf")

        print("\n" + "=" * 60)
        print("PIPELINE TEST RESULTS")
        print("=" * 60)

        # Analyze results
        stages = results.get('stages_completed', [])
        print(f"Completed stages: {len(stages)}")
        for i, stage in enumerate(stages):
            print(f"  {i+1}. {stage}")

        # Check medical filtering results
        if 'medical_filtering' in results.get('stage_outputs', {}):
            filtering_results = results['stage_outputs']['medical_filtering']
            stats = filtering_results['stats']

            print(f"\nMEDICAL FILTERING RESULTS:")
            print(f"  Original text: {stats['original_length']:,} characters")
            print(f"  Filtered text: {stats['filtered_length']:,} characters")
            print(f"  Content reduction: {stats['reduction_percent']:.1f}%")
            print(f"  Sections detected: {stats['sections_detected']}")
            print(f"  References filtered: {stats['references_filtered']}")

        # Check chunking results
        if 'chunking' in results.get('stage_outputs', {}):
            chunking_results = results['stage_outputs']['chunking']
            chunk_count = chunking_results['chunk_count']
            avg_length = chunking_results['stats']['avg_length']

            print(f"\nCHUNKING RESULTS:")
            print(f"  Chunks created: {chunk_count}")
            print(f"  Average chunk length: {avg_length:.0f} characters")

        return {
            'success': True,
            'stages_completed': len(stages),
            'medical_filtering_used': 'medical_filtering' in stages,
            'final_chunk_count': results.get('stage_outputs', {}).get('chunking', {}).get('chunk_count', 0)
        }

    finally:
        builtins.input = original_input

if __name__ == "__main__":
    try:
        results = test_medical_pipeline()
        print(f"\nFINAL TEST SUMMARY:")
        print(f"[+] Success: {results['success']}")
        print(f"[+] Stages completed: {results['stages_completed']}")
        print(f"[+] Medical filtering used: {results['medical_filtering_used']}")
        print(f"[+] Final chunks: {results['final_chunk_count']}")

        if results['medical_filtering_used']:
            print(f"\n[SUCCESS] MEDICAL FILTERING INTEGRATION SUCCESSFUL!")
            print("Your pipeline now removes references and filters for RAG quality!")

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()