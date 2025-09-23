#!/usr/bin/env python3
"""
CLI script to run the vanilla vs RAG comparison pipeline with cleansing
"""

import argparse
import os
from datetime import datetime
from tests.test_comparison_pipeline import ComparisonTestPipeline


def main():
    parser = argparse.ArgumentParser(description="Run vanilla vs RAG comparison pipeline on clinical questions")

    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Number of questions to sample for testing (default: all questions)"
    )

    parser.add_argument(
        "--output-name", "-o",
        type=str,
        default=None,
        help="Custom name for output file (default: auto-generated with timestamp)"
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        default=r"comparisons\VHA Guideline LLM Survey (Responses).csv",
        help="Path to input CSV file with questions"
    )

    args = parser.parse_args()

    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, args.input_csv)
    output_dir = os.path.join(base_dir, "tests", "results")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    if args.output_name:
        output_file = os.path.join(output_dir, f"{args.output_name}.csv")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_suffix = f"_sample{args.sample_size}" if args.sample_size else "_all"
        output_file = os.path.join(output_dir, f"vanilla_vs_rag_comparison{sample_suffix}_{timestamp}.csv")

    # Print configuration
    print("Vanilla vs RAG Comparison Pipeline")
    print("=" * 50)
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_file}")
    print(f"Sample size: {args.sample_size if args.sample_size else 'All questions'}")
    print("Features:")
    print("  - Vanilla LLM responses (original + cleansed)")
    print("  - RAG responses (original + with citations + cleansed)")
    print("  - Bias cleansing applied to both for fair comparison")
    print("=" * 50)

    # Initialize and run pipeline
    try:
        pipeline = ComparisonTestPipeline()
        pipeline.run_pipeline(
            csv_path=input_csv,
            output_path=output_file,
            sample_size=args.sample_size
        )
    except Exception as e:
        print(f"Error running comparison pipeline: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())