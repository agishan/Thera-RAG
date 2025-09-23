#!/usr/bin/env python3
"""
CLI script to run the RAG testing pipeline with custom parameters
"""

import argparse
import os
from datetime import datetime
from tests.test_pipeline import RAGTestPipeline


def main():
    parser = argparse.ArgumentParser(description="Run RAG testing pipeline on clinical questions")

    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Number of questions to sample for testing (default: all questions)"
    )

    parser.add_argument(
        "--retrieval-k", "-k",
        type=int,
        default=15,
        help="Number of documents to retrieve (default: 30)"
    )

    parser.add_argument(
        "--prompt-type", "-p",
        type=str,
        default="medical_rag",
        help="Type of prompt to use (default: medical_rag)"
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
        output_file = os.path.join(output_dir, f"rag_test_k{args.retrieval_k}{sample_suffix}_{timestamp}.csv")

    # Print configuration
    print("RAG Testing Pipeline")
    print("=" * 50)
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_file}")
    print(f"Sample size: {args.sample_size if args.sample_size else 'All questions'}")
    print(f"Retrieval K: {args.retrieval_k}")
    print(f"Prompt type: {args.prompt_type}")
    print("=" * 50)

    # Initialize and run pipeline
    try:
        pipeline = RAGTestPipeline()
        pipeline.run_pipeline(
            csv_path=input_csv,
            output_path=output_file,
            sample_size=args.sample_size,
            retrieval_k=args.retrieval_k,
            prompt_type=args.prompt_type
        )
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())