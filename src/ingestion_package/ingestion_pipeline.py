"""
Complete ingestion pipeline orchestrating all components
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .document_loader import DoclingBookLoader, LoaderFactory
from .chunking_strategy import SmartChunker, ChunkingConfig
from .metadata_extractor import DocumentMetadataExtractor, ChunkMetadataEnhancer
from .citation_extractor import CitationExtractor
from .vector_uploader import EnhancedVectorUploader


class IngestionPipeline:
    """
    Complete document ingestion pipeline for medical documents
    """

    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        pinecone_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        index_name: str = "medical-rag-index",
        namespace: str = "thera-rag-enhanced",
        enable_citations: bool = True,
        enable_upload: bool = True
    ):
        """
        Initialize the ingestion pipeline

        Args:
            chunking_config: Configuration for chunking strategy
            pinecone_api_key: Pinecone API key for vector upload
            google_api_key: Google API key for LLM-based citation extraction
            index_name: Pinecone index name
            namespace: Pinecone namespace
            enable_citations: Whether to extract citations
            enable_upload: Whether to upload to Pinecone
        """
        # Initialize components
        self.chunker = SmartChunker(chunking_config or ChunkingConfig())
        self.metadata_extractor = DocumentMetadataExtractor()

        # Citation extraction (optional)
        self.enable_citations = enable_citations
        if enable_citations:
            self.citation_extractor = CitationExtractor(
                llm_api_key=google_api_key,
                use_llm=google_api_key is not None
            )
        else:
            self.citation_extractor = None

        # Vector upload (optional)
        self.enable_upload = enable_upload and pinecone_api_key is not None
        if self.enable_upload:
            self.uploader = EnhancedVectorUploader(
                api_key=pinecone_api_key,
                index_name=index_name,
                namespace=namespace
            )
        else:
            self.uploader = None

        # Processing statistics
        self.stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "citations_extracted": 0,
            "vectors_uploaded": 0,
            "errors": []
        }

    def process_single_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single document file

        Args:
            file_path: Path to the document file
            output_dir: Directory to save intermediate outputs
            save_intermediate: Whether to save intermediate files

        Returns:
            Processing results and statistics
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"\n📄 Processing: {file_path.name}")

        try:
            # Step 1: Load document
            print("🔍 Loading document...")
            loader = LoaderFactory.create_loader(str(file_path))
            structured_content = loader.extract_structured_content()
            full_text = structured_content["text"]

            # Step 2: Extract document-level metadata
            print("📊 Extracting document metadata...")
            doc_metadata = self.metadata_extractor.extract_document_metadata(
                full_text, file_path.name, structured_content
            )

            # Step 3: Create chunks
            print("✂️  Creating intelligent chunks...")
            base_metadata = {
                "source_file": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "processed_at": datetime.now().isoformat()
            }
            chunks = self.chunker.create_chunks(full_text, base_metadata)

            # Step 4: Enhance chunk metadata
            print("🔗 Enhancing chunk metadata...")
            enhancer = ChunkMetadataEnhancer(doc_metadata)
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = enhancer.enhance_chunk_metadata(chunk, full_text, i)
                enhanced_chunks.append(enhanced_chunk)

            # Step 5: Extract citations (if enabled)
            citations = []
            citation_matches = []
            if self.enable_citations and self.citation_extractor:
                print("📚 Extracting citations...")
                try:
                    citations, citation_matches = self.citation_extractor.extract_citations_from_document(
                        full_text, enhanced_chunks
                    )
                    # Enhance chunks with citation information
                    enhanced_chunks = self.citation_extractor.enhance_chunks_with_citations(
                        enhanced_chunks, citation_matches
                    )
                    self.stats["citations_extracted"] += len(citations)
                except Exception as e:
                    print(f"⚠️  Citation extraction failed: {e}")
                    self.stats["errors"].append(f"Citation extraction: {e}")

            # Step 6: Save intermediate files (if requested)
            if save_intermediate and output_dir:
                self._save_intermediate_files(
                    file_path, output_dir, full_text, enhanced_chunks,
                    doc_metadata, citations, citation_matches
                )

            # Step 7: Upload to vector database (if enabled)
            upload_stats = {"uploaded": 0, "skipped": 0, "errors": 0}
            if self.enable_upload and self.uploader:
                print("⬆️  Uploading to vector database...")
                try:
                    upload_stats = self.uploader.upload_chunks(enhanced_chunks)
                    self.stats["vectors_uploaded"] += upload_stats["uploaded"]
                except Exception as e:
                    print(f"❌ Upload failed: {e}")
                    self.stats["errors"].append(f"Vector upload: {e}")

            # Update statistics
            self.stats["files_processed"] += 1
            self.stats["total_chunks"] += len(enhanced_chunks)

            # Analysis results
            chunk_analysis = self.chunker.analyze_chunks(enhanced_chunks)

            return {
                "success": True,
                "file_path": str(file_path),
                "document_metadata": doc_metadata.__dict__,
                "chunks_created": len(enhanced_chunks),
                "citations_found": len(citations),
                "citation_matches": len(citation_matches),
                "upload_stats": upload_stats,
                "chunk_analysis": chunk_analysis,
                "processing_time": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Failed to process {file_path.name}: {e}"
            print(f"❌ {error_msg}")
            self.stats["errors"].append(error_msg)
            return {
                "success": False,
                "file_path": str(file_path),
                "error": str(e)
            }

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        file_pattern: str = "*.pdf",
        save_intermediate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all files in a directory

        Args:
            input_dir: Directory containing documents to process
            output_dir: Directory to save intermediate outputs
            file_pattern: Glob pattern for files to process
            save_intermediate: Whether to save intermediate files

        Returns:
            List of processing results for each file
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find files to process
        files = list(input_path.glob(file_pattern))
        if not files:
            print(f"⚠️  No files found matching pattern '{file_pattern}' in {input_dir}")
            return []

        print(f"📂 Found {len(files)} files to process")

        # Setup output directory
        if save_intermediate and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Process each file
        results = []
        for file_path in files:
            result = self.process_single_file(
                str(file_path),
                output_dir,
                save_intermediate
            )
            results.append(result)

        # Print summary
        self._print_processing_summary(results)

        return results

    def _save_intermediate_files(
        self,
        file_path: Path,
        output_dir: str,
        full_text: str,
        chunks: List,
        doc_metadata: Any,
        citations: List,
        citation_matches: List
    ):
        """Save intermediate processing files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = file_path.stem

        # Save markdown
        md_path = output_path / f"{base_name}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        # Save chunks with enhanced metadata
        chunks_path = output_path / f"{base_name}_chunks.json"
        chunk_data = [
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False, default=str)

        # Save document metadata
        metadata_path = output_path / f"{base_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(doc_metadata.__dict__, f, indent=2, ensure_ascii=False, default=str)

        # Save citations if extracted
        if citations:
            citations_path = output_path / f"{base_name}_citations.json"
            citation_data = {
                "full_citations": [
                    {
                        "authors": citation.authors,
                        "year": citation.year,
                        "title": citation.title,
                        "journal": citation.journal,
                        "doi": citation.doi,
                        "full_reference": citation.full_reference,
                        "citation_type": citation.citation_type
                    }
                    for citation in citations
                ],
                "citation_matches": [
                    {
                        "inline_citation": match.inline_citation,
                        "chunk_location": match.chunk_location,
                        "context_before": match.context_before[-100:],  # Last 100 chars
                        "context_after": match.context_after[:100],     # First 100 chars
                        "full_reference": match.full_reference.full_reference
                    }
                    for match in citation_matches
                ]
            }
            with open(citations_path, 'w', encoding='utf-8') as f:
                json.dump(citation_data, f, indent=2, ensure_ascii=False)

        # Save chunk analysis
        analysis_path = output_path / f"{base_name}_analysis.json"
        analysis = self.chunker.analyze_chunks(chunks)
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"💾 Saved intermediate files to {output_path}")

    def _print_processing_summary(self, results: List[Dict]):
        """Print summary of processing results"""
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        print(f"\n📊 Processing Summary:")
        print(f"   Files processed: {len(successful)}/{len(results)}")
        print(f"   Total chunks created: {self.stats['total_chunks']}")
        print(f"   Citations extracted: {self.stats['citations_extracted']}")
        print(f"   Vectors uploaded: {self.stats['vectors_uploaded']}")

        if failed:
            print(f"   Failed files: {len(failed)}")
            for result in failed:
                print(f"     - {Path(result['file_path']).name}: {result.get('error', 'Unknown error')}")

        if self.stats["errors"]:
            print(f"   Errors encountered: {len(self.stats['errors'])}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        return self.stats.copy()

    def verify_upload(self, sample_query: str = "medical treatment") -> Dict[str, Any]:
        """Verify vector upload if uploader is available"""
        if not self.uploader:
            return {"error": "Vector uploader not initialized"}

        return self.uploader.verify_upload(
            sample_query=sample_query,
            expected_chunks=self.stats["vectors_uploaded"]
        )


# Convenience function for quick processing
def process_documents(
    input_path: str,
    output_dir: str,
    pinecone_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    chunking_config: Optional[ChunkingConfig] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to process documents with default settings

    Args:
        input_path: Path to file or directory
        output_dir: Output directory for intermediate files
        pinecone_api_key: Pinecone API key
        google_api_key: Google API key for LLM features
        chunking_config: Custom chunking configuration
        **kwargs: Additional arguments for IngestionPipeline

    Returns:
        List of processing results
    """
    pipeline = IngestionPipeline(
        chunking_config=chunking_config,
        pinecone_api_key=pinecone_api_key,
        google_api_key=google_api_key,
        **kwargs
    )

    input_path = Path(input_path)
    if input_path.is_file():
        return [pipeline.process_single_file(str(input_path), output_dir)]
    elif input_path.is_dir():
        return pipeline.process_directory(str(input_path), output_dir)
    else:
        raise ValueError(f"Invalid input path: {input_path}")