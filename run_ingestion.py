"""
Run the enhanced ingestion pipeline with quality verification

This script runs the complete ingestion process and provides comprehensive
quality verification for citations and chunks.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add the ingestion package to Python path
sys.path.append('src')

from ingestion_package import (
    IngestionPipeline,
    ChunkingConfig,
    process_documents
)

# Load environment variables
load_dotenv()


class IngestionRunner:
    """Runs ingestion with comprehensive quality verification"""

    def __init__(self):
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")

        # Verify API keys
        if not self.pinecone_key:
            print("WARNING: PINECONE_API_KEY not found in environment")
        if not self.google_key:
            print("WARNING: GOOGLE_API_KEY not found in environment")

    def run_basic_ingestion(self, input_path="src/Ingestion/outputs", test_mode=True):
        """Run basic ingestion with quality checks"""

        print("Starting Enhanced Ingestion Pipeline")
        print("=" * 50)

        # Configuration for medical documents
        config = ChunkingConfig(
            target_chunk_size=3500,    # Good for medical content
            overlap_size=350,          # 10% overlap
            section_aware=True,        # Preserve document structure
            preserve_sentences=True    # Don't break sentences
        )

        # Initialize pipeline
        pipeline = IngestionPipeline(
            chunking_config=config,
            pinecone_api_key=self.pinecone_key,
            google_api_key=self.google_key,
            index_name="medical-rag-index",
            namespace="enhanced-test" if test_mode else "enhanced-production",
            enable_citations=True,
            enable_upload=bool(self.pinecone_key)  # Only upload if key available
        )

        # Create output directory
        output_dir = "enhanced_output"
        Path(output_dir).mkdir(exist_ok=True)

        print(f"📁 Input: {input_path}")
        print(f"📁 Output: {output_dir}")
        print(f"🔗 Citations: {'Enabled' if self.google_key else 'Disabled (no API key)'}")
        print(f"⬆️  Upload: {'Enabled' if self.pinecone_key else 'Disabled (no API key)'}")
        print()

        # Check if we have existing chunks to process
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"❌ Input path does not exist: {input_path}")
            return None

        # Find JSON files (processed chunks)
        json_files = list(input_path.glob("*.json"))
        json_files = [f for f in json_files if not f.name.endswith("_stats.json")]

        if not json_files:
            print(f"❌ No chunk files found in {input_path}")
            print("   Looking for files like: curry_et_al-2018-british_journal_of_haematology.json")
            return None

        print(f"📄 Found {len(json_files)} chunk files to process")

        # Process the existing chunks with enhanced metadata
        results = []
        for json_file in json_files:
            print(f"\n📄 Processing: {json_file.name}")

            try:
                # Load existing chunks
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_chunks = json.load(f)

                print(f"   Loaded {len(existing_chunks)} existing chunks")

                # Convert to our enhanced format and process
                result = self._process_existing_chunks(
                    json_file, existing_chunks, pipeline, output_dir
                )
                results.append(result)

            except Exception as e:
                print(f"   ❌ Error processing {json_file.name}: {e}")
                results.append({
                    "success": False,
                    "file_path": str(json_file),
                    "error": str(e)
                })

        return results, pipeline

    def _process_existing_chunks(self, json_file, existing_chunks, pipeline, output_dir):
        """Process existing chunks with enhanced metadata"""

        # Create enhanced chunks
        from langchain_core.documents import Document

        # Convert existing chunks to Document objects
        documents = []
        for i, chunk_data in enumerate(existing_chunks):
            content = chunk_data.get('content', chunk_data.get('page_content', ''))
            metadata = chunk_data.get('metadata', {})

            # Add enhanced metadata
            enhanced_metadata = {
                **metadata,
                "chunk_index": i,
                "source_file": json_file.name,
                "text": content,  # Full content (no 500 char limit)
                "chunk_size": len(content),
                "word_count": len(content.split()),
                "processed_at": datetime.now().isoformat(),
                "processing_version": "enhanced_v2.0"
            }

            doc = Document(page_content=content, metadata=enhanced_metadata)
            documents.append(doc)

        print(f"   Created {len(documents)} enhanced chunks")

        # Extract citations if LLM available
        citations_found = 0
        citation_matches = 0

        if pipeline.enable_citations and pipeline.citation_extractor:
            try:
                # Load corresponding markdown file if available
                md_file = json_file.with_suffix('.md')
                full_text = ""
                if md_file.exists():
                    with open(md_file, 'r', encoding='utf-8') as f:
                        full_text = f.read()

                if full_text:
                    print("   🔍 Extracting citations...")
                    citations, matches = pipeline.citation_extractor.extract_citations_from_document(
                        full_text, documents
                    )

                    # Enhance chunks with citations
                    documents = pipeline.citation_extractor.enhance_chunks_with_citations(
                        documents, matches
                    )

                    citations_found = len(citations)
                    citation_matches = len(matches)
                    print(f"   📚 Found {citations_found} citations, {citation_matches} matches")

            except Exception as e:
                print(f"   ⚠️  Citation extraction failed: {e}")

        # Upload to Pinecone if enabled
        upload_stats = {"uploaded": 0, "skipped": 0, "errors": 0}
        if pipeline.enable_upload and pipeline.uploader:
            try:
                print("   ⬆️  Uploading to Pinecone...")
                upload_stats = pipeline.uploader.upload_chunks(documents, show_progress=False)
                print(f"   ✅ Uploaded {upload_stats['uploaded']} chunks")
            except Exception as e:
                print(f"   ❌ Upload failed: {e}")

        # Save enhanced chunks
        output_file = Path(output_dir) / f"enhanced_{json_file.name}"
        enhanced_data = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"   💾 Saved enhanced chunks to {output_file.name}")

        return {
            "success": True,
            "file_path": str(json_file),
            "chunks_created": len(documents),
            "citations_found": citations_found,
            "citation_matches": citation_matches,
            "upload_stats": upload_stats,
            "output_file": str(output_file)
        }

    def verify_citation_quality(self, results):
        """Comprehensive citation quality verification"""

        print("\n📚 Citation Quality Verification")
        print("=" * 40)

        total_files = len([r for r in results if r.get("success")])
        total_citations = sum(r.get("citations_found", 0) for r in results if r.get("success"))
        total_matches = sum(r.get("citation_matches", 0) for r in results if r.get("success"))

        print(f"📊 Citation Statistics:")
        print(f"   Files processed: {total_files}")
        print(f"   Total citations found: {total_citations}")
        print(f"   Inline citations matched: {total_matches}")

        if total_citations > 0:
            match_rate = (total_matches / total_citations) * 100
            print(f"   Match rate: {match_rate:.1f}%")

            # Quality assessment
            if match_rate > 80:
                print("   ✅ Excellent citation linking quality")
            elif match_rate > 60:
                print("   ⚠️  Good citation linking, some misses")
            else:
                print("   ❌ Poor citation linking, needs review")
        else:
            print("   ⚠️  No citations found (check document format)")

        # Detailed citation analysis
        print(f"\n🔍 Detailed Citation Analysis:")
        for result in results:
            if result.get("success") and result.get("citations_found", 0) > 0:
                file_name = Path(result["file_path"]).stem
                citations = result["citations_found"]
                matches = result["citation_matches"]

                print(f"   📄 {file_name}:")
                print(f"      Citations: {citations}, Matches: {matches}")

                if citations > 0:
                    rate = (matches / citations) * 100
                    print(f"      Match rate: {rate:.1f}%")

    def verify_chunk_quality(self, results, output_dir="enhanced_output"):
        """Comprehensive chunk quality verification"""

        print("\n✂️  Chunk Quality Verification")
        print("=" * 40)

        total_chunks = sum(r.get("chunks_created", 0) for r in results if r.get("success"))

        print(f"📊 Chunk Statistics:")
        print(f"   Total chunks created: {total_chunks}")

        # Analyze chunk files
        chunk_sizes = []
        metadata_quality = {
            "has_full_text": 0,
            "has_source_info": 0,
            "has_location": 0,
            "has_citations": 0
        }

        output_path = Path(output_dir)
        enhanced_files = list(output_path.glob("enhanced_*.json"))

        print(f"\n🔍 Analyzing {len(enhanced_files)} enhanced chunk files:")

        for enhanced_file in enhanced_files:
            try:
                with open(enhanced_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                file_chunks = len(chunks)
                print(f"   📄 {enhanced_file.name}: {file_chunks} chunks")

                # Analyze chunk quality
                for chunk in chunks:
                    content = chunk.get('content', '')
                    metadata = chunk.get('metadata', {})

                    chunk_sizes.append(len(content))

                    # Check metadata quality
                    if metadata.get('text') and len(metadata.get('text', '')) > 500:
                        metadata_quality["has_full_text"] += 1

                    if metadata.get('source_title') or metadata.get('source_file'):
                        metadata_quality["has_source_info"] += 1

                    if metadata.get('page_start') or metadata.get('hierarchy_breadcrumb'):
                        metadata_quality["has_location"] += 1

                    if metadata.get('citations') or metadata.get('citation_count'):
                        metadata_quality["has_citations"] += 1

            except Exception as e:
                print(f"   ❌ Error analyzing {enhanced_file.name}: {e}")

        # Quality metrics
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)

            print(f"\n📏 Chunk Size Analysis:")
            print(f"   Average size: {avg_size:.0f} characters")
            print(f"   Size range: {min_size} - {max_size} characters")

            # Size quality assessment
            if 2000 <= avg_size <= 5000:
                print("   ✅ Good average chunk size")
            elif avg_size < 1000:
                print("   ⚠️  Chunks may be too small")
            else:
                print("   ⚠️  Chunks may be too large")

        # Metadata quality
        if total_chunks > 0:
            print(f"\n📋 Metadata Quality:")
            for check, count in metadata_quality.items():
                percentage = (count / total_chunks) * 100
                status = "✅" if percentage > 80 else "⚠️" if percentage > 50 else "❌"
                print(f"   {status} {check.replace('_', ' ').title()}: {percentage:.1f}%")

    def verify_vector_upload(self, pipeline):
        """Verify Pinecone upload quality"""

        print("\n⬆️  Vector Upload Verification")
        print("=" * 40)

        if not pipeline.enable_upload or not pipeline.uploader:
            print("   ⚠️  Vector upload not enabled")
            return

        try:
            verification = pipeline.verify_upload("medical treatment")

            print(f"📊 Upload Verification:")
            print(f"   Namespace: {verification['namespace']}")
            print(f"   Total vectors: {verification['total_vectors']}")
            print(f"   Upload successful: {verification['upload_success']}")

            # Test query
            query_test = verification['query_test']
            print(f"\n🔍 Query Test:")
            print(f"   Query: '{query_test['query']}'")
            print(f"   Results found: {query_test['results_found']}")
            print(f"   Test passed: {query_test['test_passed']}")

            # Metadata quality
            metadata_quality = verification.get('metadata_quality', {})
            if metadata_quality:
                quality_score = metadata_quality.get('quality_score', 0)
                print(f"\n📋 Metadata Quality Score: {quality_score:.1f}%")

                if quality_score > 85:
                    print("   ✅ Excellent metadata quality")
                elif quality_score > 70:
                    print("   ⚠️  Good metadata quality")
                else:
                    print("   ❌ Poor metadata quality")

            # Sample metadata
            sample_meta = verification.get('sample_metadata', {})
            if sample_meta:
                print(f"\n🔬 Sample Metadata Check:")
                print(f"   Full text preserved: {sample_meta.get('has_full_text', False)}")
                print(f"   Citation info: {sample_meta.get('has_citations', False)}")
                print(f"   Location data: {sample_meta.get('has_location', False)}")

                fields = sample_meta.get('metadata_fields', [])
                if fields:
                    print(f"   Metadata fields: {len(fields)} fields available")
                    print(f"   Sample fields: {', '.join(fields[:5])}...")

        except Exception as e:
            print(f"   ❌ Verification failed: {e}")

    def run_comprehensive_test(self):
        """Run complete ingestion with all verifications"""

        print("🧪 Running Comprehensive Ingestion Test")
        print("=" * 50)

        # Run ingestion
        results, pipeline = self.run_basic_ingestion(test_mode=True)

        if not results:
            print("❌ Ingestion failed - no results")
            return

        # Verify quality
        self.verify_citation_quality(results)
        self.verify_chunk_quality(results)
        self.verify_vector_upload(pipeline)

        # Summary
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        print(f"\n📊 Final Summary:")
        print(f"   ✅ Successful files: {len(successful)}")
        print(f"   ❌ Failed files: {len(failed)}")
        print(f"   📄 Total chunks: {sum(r.get('chunks_created', 0) for r in successful)}")
        print(f"   📚 Total citations: {sum(r.get('citations_found', 0) for r in successful)}")

        if failed:
            print(f"\n❌ Failed Files:")
            for result in failed:
                print(f"   - {Path(result['file_path']).name}: {result.get('error', 'Unknown error')}")

        return results, pipeline


def main():
    """Main function to run ingestion"""

    runner = IngestionRunner()

    # Check API keys
    if not runner.pinecone_key and not runner.google_key:
        print("⚠️  No API keys found. Running in local-only mode.")
        print("   Set PINECONE_API_KEY and GOOGLE_API_KEY in .env for full functionality")

    # Run comprehensive test
    try:
        results, pipeline = runner.run_comprehensive_test()

        print("\n✅ Ingestion completed successfully!")
        print("\n📁 Check the 'enhanced_output' directory for processed files.")
        print("   Look for files like: enhanced_curry_et_al-2018-british_journal_of_haematology.json")

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()