"""
Simplified Medical Document Ingestion Pipeline

Focus: LLM-powered citation extraction with clean, minimal components
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document

# Clean imports for new structure
from .document_processor import DoclingBookLoader
from .chunk_processor import SmartChunker as ChunkProcessor, ChunkingConfig as ChunkConfig
from .metadata_enhancer import (
    ChunkMetadataEnhancer as MetadataEnhancer,
    DocumentMetadataExtractor,
)
from .citation_extractor import CitationExtractor
from .vector_uploader import EnhancedVectorUploader as VectorUploader
from .chunk_classifier import ChunkClassifier


class SimplePipeline:
    """
    Clean, focused pipeline for medical document ingestion with LLM citation extraction
    """

    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        index_name: str = "medical-rag-index",
        namespace: str = "thera-rag"
    ):
        """
        Initialize with minimal configuration

        Args:
            pinecone_api_key: For vector storage
            google_api_key: For LLM citation extraction (CORE FEATURE)
            index_name: Pinecone index
            namespace: Pinecone namespace
        """
        # Core components (created per document where needed)
        self.chunk_processor = ChunkProcessor(ChunkConfig())

        # Citation extraction - THE KEY FEATURE
        self.citation_extractor = CitationExtractor(
            llm_api_key=google_api_key,
            use_llm=google_api_key is not None
        )

        # Vector upload (optional)
        self.vector_uploader = None
        if pinecone_api_key:
            self.vector_uploader = VectorUploader(
                api_key=pinecone_api_key,
                index_name=index_name,
                namespace=namespace
            )

    def process_document(self, file_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single document with citation extraction focus

        Args:
            file_path: Path to PDF document
            output_dir: Optional output directory for enhanced chunks

        Returns:
            Processing results with citation stats
        """
        file_path = Path(file_path)
        print(f"Processing: {file_path.name}")

        try:
            # 1. Extract text with Docling
            print("  - Extracting text...")
            loader = DoclingBookLoader(str(file_path))
            structured_content = loader.extract_structured_content()
            full_text = structured_content.get("full_text") or structured_content.get("text", "")

            # 2. Create smart chunks
            print("  - Creating chunks...")
            base_metadata = {
                "source_file": file_path.name,
                "processed_at": datetime.now().isoformat()
            }
            chunks = self.chunk_processor.create_chunks(full_text, base_metadata)

            # 2.5 Classify chunks and filter non-medical content
            print("  - Classifying chunks (2.5)...")
            classifier = ChunkClassifier()
            filtered_chunks = []
            classification_counts = {}
            for ch in chunks:
                result = classifier.classify(ch.page_content)
                label = result.get("label", "medical_content")
                score = result.get("score", 0.0)
                ch.metadata["chunk_category"] = label
                ch.metadata["reference_likeness"] = score
                classification_counts[label] = classification_counts.get(label, 0) + 1
                if label in {"references", "appendix", "acknowledgments", "keywords"}:
                    continue
                filtered_chunks.append(ch)

            # 3. Enhance metadata (NO 500 char limit)
            print("  - Enhancing metadata...")
            enhanced_chunks = []
            doc_meta_extractor = DocumentMetadataExtractor()
            doc_ref = doc_meta_extractor.extract_document_metadata(
                full_text, file_path.name, structured_content
            )
            metadata_enhancer = MetadataEnhancer(doc_ref)

            for i, chunk in enumerate(filtered_chunks):
                enhanced_chunk = metadata_enhancer.enhance_chunk_metadata(
                    chunk, full_text, i
                )
                enhanced_chunks.append(enhanced_chunk)

            # 4. CORE FEATURE: Extract citations with LLM
            print("  - Extracting citations (LLM)...")
            citations, citation_matches = self.citation_extractor.extract_citations_from_document(
                full_text, enhanced_chunks
            )

            # 5. Link citations to chunks
            final_chunks = self.citation_extractor.enhance_chunks_with_citations(
                enhanced_chunks, citation_matches
            )

            # 6. Save enhanced output
            if output_dir:
                self._save_enhanced_chunks(file_path, output_dir, final_chunks)

            # 7. Upload to Pinecone (optional)
            upload_stats = {"uploaded": 0}
            if self.vector_uploader:
                print("  - Uploading to Pinecone...")
                upload_stats = self.vector_uploader.upload_chunks(final_chunks)

            # Results focused on citation quality
            return {
                "success": True,
                "file": file_path.name,
                "chunks_created": len(final_chunks),
                "citations_extracted": len(citations),
                "citation_matches": len(citation_matches),
                "upload_count": upload_stats["uploaded"],
                "citation_coverage": self._calculate_citation_coverage(final_chunks),
                "classification_counts": classification_counts,
                "filtered_out": len(chunks) - len(filtered_chunks)
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            return {
                "success": False,
                "file": file_path.name,
                "error": str(e)
            }

    def _save_enhanced_chunks(self, file_path: Path, output_dir: str, chunks: List[Document]):
        """Save enhanced chunks with full metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as enhanced JSON with full content preserved
        output_file = output_path / f"enhanced_{file_path.stem}.json"

        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"  - Saved: {output_file}")

    def _calculate_citation_coverage(self, chunks: List[Document]) -> Dict[str, Any]:
        """Calculate citation extraction quality metrics"""
        total_chunks = len(chunks)
        chunks_with_citations = sum(1 for chunk in chunks if chunk.metadata.get('citations'))
        total_citations = sum(len(chunk.metadata.get('citations', [])) for chunk in chunks)

        return {
            "chunks_with_citations": chunks_with_citations,
            "citation_coverage_percent": (chunks_with_citations / total_chunks * 100) if total_chunks > 0 else 0,
            "total_citations": total_citations,
            "avg_citations_per_chunk": total_citations / total_chunks if total_chunks > 0 else 0
        }

    def process_directory(self, input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """Process all PDFs in a directory"""
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return []

        print(f"Found {len(pdf_files)} PDF files")

        results = []
        for pdf_file in pdf_files:
            result = self.process_document(str(pdf_file), output_dir)
            results.append(result)

        # Print summary focused on citations
        self._print_citation_summary(results)
        return results

    def _print_citation_summary(self, results: List[Dict[str, Any]]):
        """Print summary focusing on citation extraction quality"""
        successful = [r for r in results if r.get("success")]

        if successful:
            total_chunks = sum(r["chunks_created"] for r in successful)
            total_citations = sum(r["citations_extracted"] for r in successful)
            total_matches = sum(r["citation_matches"] for r in successful)

            print(f"\nCitation Extraction Summary:")
            print(f"  Files processed: {len(successful)}")
            print(f"  Total chunks: {total_chunks}")
            print(f"  Citations extracted: {total_citations}")
            print(f"  Citation-chunk matches: {total_matches}")
            print(f"  Avg citations per file: {total_citations/len(successful):.1f}")

        failed = [r for r in results if not r.get("success")]
        if failed:
            print(f"  Failed files: {len(failed)}")


# Simple convenience function
def ingest_documents(
    input_path: str,
    output_dir: str,
    pinecone_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Simple document ingestion with citation extraction focus

    Args:
        input_path: File or directory path
        output_dir: Output directory for enhanced chunks
        pinecone_api_key: Pinecone API key (optional)
        google_api_key: Google API key for LLM citation extraction

    Returns:
        Processing results
    """
    pipeline = SimplePipeline(
        pinecone_api_key=pinecone_api_key,
        google_api_key=google_api_key
    )

    input_path = Path(input_path)
    if input_path.is_file():
        return [pipeline.process_document(str(input_path), output_dir)]
    elif input_path.is_dir():
        return pipeline.process_directory(str(input_path), output_dir)
    else:
        raise ValueError(f"Invalid path: {input_path}")
