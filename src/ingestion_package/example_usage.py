"""
Example usage of the Medical Document Ingestion Package

This file demonstrates various usage patterns for the ingestion package.
Run specific examples by uncommenting the function calls at the bottom.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from ingestion_package import (
    IngestionPipeline,
    ChunkingConfig,
    process_documents
)

# Load environment variables
load_dotenv()


def basic_usage_example():
    """Basic usage with minimal configuration"""

    # Simple processing without vector upload
    pipeline = IngestionPipeline(
        enable_upload=False,  # Just process locally
        enable_citations=False  # Skip citation extraction
    )

    # Process a single file
    result = pipeline.process_single_file(
        file_path="path/to/your/document.pdf",
        output_dir="output_directory",
        save_intermediate=True
    )

    print(f"Processed {result['chunks_created']} chunks")


def advanced_usage_example():
    """Advanced usage with full features"""

    # Get API keys from environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Custom chunking configuration
    chunking_config = ChunkingConfig(
        target_chunk_size=4000,      # Larger chunks
        overlap_size=400,            # 10% overlap
        section_aware=True,          # Use section-based chunking
        preserve_sentences=True      # Don't break sentences
    )

    # Initialize pipeline with all features
    pipeline = IngestionPipeline(
        chunking_config=chunking_config,
        pinecone_api_key=pinecone_api_key,
        google_api_key=google_api_key,
        index_name="medical-rag-index",
        namespace="thera-rag-v2",
        enable_citations=True,       # Extract citations
        enable_upload=True           # Upload to Pinecone
    )

    # Process all PDFs in a directory
    results = pipeline.process_directory(
        input_dir="input_pdfs/",
        output_dir="processed_output/",
        file_pattern="*.pdf",
        save_intermediate=True
    )

    # Print summary
    successful = [r for r in results if r.get("success")]
    print(f"Successfully processed {len(successful)} files")

    # Verify upload
    if pipeline.uploader:
        verification = pipeline.verify_upload("depression treatment")
        print(f"Upload verification: {verification['query_test']['test_passed']}")


def convenience_function_example():
    """Using the convenience function"""

    # Quick processing with defaults
    results = process_documents(
        input_path="input_pdfs/",
        output_dir="processed_output/",
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        # Use default chunking configuration
        namespace="quick-test",
        enable_citations=True
    )

    print(f"Processed {len(results)} files")


def citation_focused_example():
    """Example focusing on citation extraction"""

    pipeline = IngestionPipeline(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        enable_citations=True,
        enable_upload=False  # Just extract citations, don't upload
    )

    result = pipeline.process_single_file(
        file_path="research_paper.pdf",
        output_dir="citation_analysis/",
        save_intermediate=True
    )

    print(f"Found {result['citations_found']} citations")
    print(f"Linked {result['citation_matches']} inline citations to references")

    # Citation data will be saved in citation_analysis/research_paper_citations.json


def metadata_analysis_example():
    """Example focusing on metadata extraction"""

    pipeline = IngestionPipeline(
        enable_citations=False,
        enable_upload=False
    )

    result = pipeline.process_single_file(
        file_path="clinical_guideline.pdf",
        output_dir="metadata_analysis/",
        save_intermediate=True
    )

    doc_metadata = result['document_metadata']
    print(f"Title: {doc_metadata['source_title']}")
    print(f"Authors: {doc_metadata['authors']}")
    print(f"Year: {doc_metadata['publication_year']}")
    print(f"Type: {doc_metadata['document_type']}")

    chunk_analysis = result['chunk_analysis']
    print(f"Average chunk size: {chunk_analysis['avg_chunk_size']:.0f} characters")
    print(f"Sections with titles: {chunk_analysis['sections_with_titles']}")


if __name__ == "__main__":
    print("Medical Document Ingestion Package Examples")
    print("=" * 50)

    # Run basic example
    print("\n1. Basic Usage Example:")
    # basic_usage_example()

    print("\n2. Advanced Usage Example:")
    # advanced_usage_example()

    print("\n3. Convenience Function Example:")
    # convenience_function_example()

    print("\n4. Citation-Focused Example:")
    # citation_focused_example()

    print("\n5. Metadata Analysis Example:")
    # metadata_analysis_example()

    print("\nUncomment the function calls above to run specific examples")
    print("Make sure to set your API keys in a .env file:")
    print("PINECONE_API_KEY=your_pinecone_key")
    print("GOOGLE_API_KEY=your_google_key")