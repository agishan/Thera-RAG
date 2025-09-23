"""
Citation-Focused Document Ingestion

Clean, simple script highlighting LLM citation extraction as the core feature.
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# Clean imports from new structure
from src.ingestion import SimplePipeline, ingest_documents


def main():
    """Run citation-focused ingestion with clean output"""

    print("Medical Document Ingestion with LLM Citation Extraction")
    print("=" * 60)

    # Configuration
    input_dir = project_root / "data"
    output_dir = project_root / "enhanced_output"

    # API keys (citation extraction requires Google API key)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not google_api_key:
        print("WARNING: No GOOGLE_API_KEY found - citation extraction will be limited to regex only")
        print("For full LLM citation extraction, set GOOGLE_API_KEY environment variable")

    if not pinecone_api_key:
        print("INFO: No PINECONE_API_KEY found - will save enhanced chunks locally only")

    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"LLM citation extraction: {'Enabled' if google_api_key else 'Disabled (regex fallback)'}")
    print(f"Pinecone upload: {'Enabled' if pinecone_api_key else 'Disabled'}")

    # Check for input files
    if not input_dir.exists():
        print(f"\nERROR: Input directory {input_dir} not found")
        print("Please create the directory and add PDF files to process")
        return

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\nERROR: No PDF files found in {input_dir}")
        print("Please add PDF files to process")
        return

    print(f"\nFound {len(pdf_files)} PDF files to process")

    # Process documents with citation focus
    try:
        results = ingest_documents(
            input_path=str(input_dir),
            output_dir=str(output_dir),
            pinecone_api_key=pinecone_api_key,
            google_api_key=google_api_key
        )

        # Display citation-focused results
        print("\nCitation Extraction Results:")
        print("-" * 40)

        successful_results = [r for r in results if r.get("success")]

        if successful_results:
            for result in successful_results:
                print(f"File: {result['file']}")
                print(f"  Chunks: {result['chunks_created']}")
                print(f"  Citations: {result['citations_extracted']}")
                print(f"  Citation matches: {result['citation_matches']}")
                if 'citation_coverage' in result:
                    coverage = result['citation_coverage']
                    print(f"  Coverage: {coverage['citation_coverage_percent']:.1f}% of chunks have citations")
                print()

            # Overall stats
            total_citations = sum(r['citations_extracted'] for r in successful_results)
            total_chunks = sum(r['chunks_created'] for r in successful_results)

            print(f"TOTAL: {total_citations} citations extracted from {total_chunks} chunks")
            print(f"Average: {total_citations/len(successful_results):.1f} citations per document")

            if google_api_key:
                print("\nSUCCESS: LLM citation extraction completed!")
                print("Enhanced chunks with full citation metadata saved to enhanced_output/")
            else:
                print("\nNOTE: Only regex citation extraction was used.")
                print("For better results, set GOOGLE_API_KEY for LLM-powered extraction.")

        # Handle failures
        failed_results = [r for r in results if not r.get("success")]
        if failed_results:
            print(f"\nFailed files ({len(failed_results)}):")
            for result in failed_results:
                print(f"  {result['file']}: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\nERROR: Processing failed: {e}")
        return

    print(f"\nProcessing complete! Enhanced chunks saved to: {output_dir}")
    print("\nTo verify quality, run: python verify_output_quality.py")


if __name__ == "__main__":
    main()