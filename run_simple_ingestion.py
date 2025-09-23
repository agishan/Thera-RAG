"""
Simple ingestion runner without Unicode characters for Windows compatibility
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add the ingestion package to Python path
sys.path.append('src')

try:
    from ingestion_package import (
        IngestionPipeline,
        ChunkingConfig
    )
    print("Successfully imported ingestion package")
except ImportError as e:
    print(f"Failed to import ingestion package: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()


def run_test_ingestion():
    """Run a simple test of the ingestion pipeline"""

    print("Enhanced Ingestion Test")
    print("=" * 40)

    # Check API keys
    pinecone_key = os.getenv("PINECONE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    print(f"Pinecone API Key: {'Found' if pinecone_key else 'Missing'}")
    print(f"Google API Key: {'Found' if google_key else 'Missing'}")

    if not pinecone_key and not google_key:
        print("WARNING: No API keys found. Running in local-only mode.")

    # Configure chunking
    config = ChunkingConfig(
        target_chunk_size=3500,
        overlap_size=350,
        section_aware=True,
        preserve_sentences=True
    )

    # Initialize pipeline
    pipeline = IngestionPipeline(
        chunking_config=config,
        pinecone_api_key=pinecone_key,
        google_api_key=google_key,
        index_name="medical-rag-index",
        namespace="test-enhanced",
        enable_citations=bool(google_key),
        enable_upload=bool(pinecone_key)
    )

    print(f"Citations enabled: {pipeline.enable_citations}")
    print(f"Upload enabled: {pipeline.enable_upload}")

    # Check for input files
    input_dir = Path("src/Ingestion/outputs")
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    # Find JSON files
    json_files = list(input_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.endswith("_stats.json")]

    if not json_files:
        print(f"ERROR: No chunk files found in {input_dir}")
        return

    print(f"Found {len(json_files)} files to process")

    # Create output directory
    output_dir = Path("enhanced_output")
    output_dir.mkdir(exist_ok=True)

    # Process first file as test
    test_file = json_files[0]
    print(f"\nProcessing test file: {test_file.name}")

    try:
        # Load existing chunks
        with open(test_file, 'r', encoding='utf-8') as f:
            existing_chunks = json.load(f)

        print(f"Loaded {len(existing_chunks)} existing chunks")

        # Convert to Document objects
        from langchain_core.documents import Document

        documents = []
        for i, chunk_data in enumerate(existing_chunks):
            content = chunk_data.get('content', chunk_data.get('page_content', ''))
            metadata = chunk_data.get('metadata', {})

            # Add enhanced metadata (NO 500 CHAR LIMIT)
            enhanced_metadata = {
                **metadata,
                "chunk_index": i,
                "source_file": test_file.name,
                "text": content,  # FULL CONTENT PRESERVED
                "chunk_size": len(content),
                "word_count": len(content.split()),
                "processed_at": datetime.now().isoformat(),
                "processing_version": "enhanced_v2.0"
            }

            doc = Document(page_content=content, metadata=enhanced_metadata)
            documents.append(doc)

        print(f"Created {len(documents)} enhanced chunks")

        # Test citation extraction if enabled
        citations_found = 0
        if pipeline.enable_citations:
            print("Testing citation extraction...")

            # Try to find markdown file
            md_file = test_file.with_suffix('.md')
            if md_file.exists():
                with open(md_file, 'r', encoding='utf-8') as f:
                    full_text = f.read()

                try:
                    citations, matches = pipeline.citation_extractor.extract_citations_from_document(
                        full_text, documents
                    )
                    citations_found = len(citations)
                    print(f"Found {citations_found} citations, {len(matches)} matches")

                    # Enhance chunks with citations
                    documents = pipeline.citation_extractor.enhance_chunks_with_citations(
                        documents, matches
                    )

                except Exception as e:
                    print(f"Citation extraction failed: {e}")

        # Test upload if enabled
        upload_count = 0
        if pipeline.enable_upload:
            print("Testing vector upload...")
            try:
                stats = pipeline.uploader.upload_chunks(documents[:5])  # Test with first 5 chunks
                upload_count = stats.get("uploaded", 0)
                print(f"Uploaded {upload_count} test chunks")
            except Exception as e:
                print(f"Upload failed: {e}")

        # Save enhanced chunks
        output_file = output_dir / f"enhanced_{test_file.name}"
        enhanced_data = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"Saved enhanced chunks to {output_file}")

        # Quality check
        print("\nQuality Check:")
        print(f"  Chunks processed: {len(documents)}")
        print(f"  Average chunk size: {sum(len(d.page_content) for d in documents) / len(documents):.0f} chars")
        print(f"  Citations found: {citations_found}")
        print(f"  Vectors uploaded: {upload_count}")

        # Check metadata quality
        full_text_count = sum(1 for d in documents if len(d.metadata.get('text', '')) > 500)
        print(f"  Full text preserved: {full_text_count}/{len(documents)} chunks")

        if full_text_count > len(documents) * 0.8:
            print("  QUALITY: GOOD - Full content preserved")
        else:
            print("  QUALITY: POOR - Content truncated")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        import traceback
        traceback.print_exc()


def verify_enhanced_output():
    """Verify the quality of enhanced output"""

    print("\nEnhanced Output Verification")
    print("=" * 40)

    output_dir = Path("enhanced_output")
    if not output_dir.exists():
        print("ERROR: No enhanced output directory found")
        return

    enhanced_files = list(output_dir.glob("enhanced_*.json"))
    if not enhanced_files:
        print("ERROR: No enhanced files found")
        return

    print(f"Found {len(enhanced_files)} enhanced files")

    total_chunks = 0
    total_size = 0
    full_text_count = 0
    citation_count = 0

    for file_path in enhanced_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            file_chunks = len(chunks)
            total_chunks += file_chunks

            print(f"  {file_path.name}: {file_chunks} chunks")

            for chunk in chunks:
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})

                total_size += len(content)

                # Check if full text is preserved
                if len(metadata.get('text', '')) > 500:
                    full_text_count += 1

                # Check for citations
                if metadata.get('citations') or metadata.get('citation_count'):
                    citation_count += 1

        except Exception as e:
            print(f"  ERROR analyzing {file_path.name}: {e}")

    if total_chunks > 0:
        avg_size = total_size / total_chunks
        full_text_rate = (full_text_count / total_chunks) * 100
        citation_rate = (citation_count / total_chunks) * 100

        print(f"\nQuality Summary:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Average size: {avg_size:.0f} characters")
        print(f"  Full text preserved: {full_text_rate:.1f}%")
        print(f"  Chunks with citations: {citation_rate:.1f}%")

        # Overall quality assessment
        quality_score = 0
        if 2000 <= avg_size <= 5000:
            quality_score += 25
        if full_text_rate > 80:
            quality_score += 35
        if citation_rate > 10:
            quality_score += 20
        quality_score += 20  # Base score for successful processing

        print(f"  Overall quality score: {quality_score}/100")

        if quality_score > 80:
            print("  RESULT: EXCELLENT quality")
        elif quality_score > 60:
            print("  RESULT: GOOD quality")
        else:
            print("  RESULT: NEEDS IMPROVEMENT")


if __name__ == "__main__":
    print("Simple Ingestion Test")
    print("=" * 50)

    try:
        run_test_ingestion()
        verify_enhanced_output()

        print("\nTest completed! Check the 'enhanced_output' directory for results.")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()