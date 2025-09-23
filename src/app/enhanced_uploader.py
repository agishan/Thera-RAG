"""
Enhanced Pinecone Uploader with Rich Metadata
"""

import os
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm
from dotenv import load_dotenv


class EnhancedPineconeUploader:
    """Upload chunks with enhanced metadata to Pinecone"""

    def __init__(self, api_key: str, index_name: str, namespace: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.embedder = SentenceTransformer("intfloat/e5-base")

    def prepare_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for Pinecone with size and type constraints
        """
        # Pinecone metadata limits: 40KB per vector, values must be strings/numbers/booleans
        pinecone_metadata = {}

        # Essential fields (always include)
        essential_fields = [
            'chunk_id', 'source_title', 'source_authors', 'source_year',
            'page_start', 'page_end', 'chunk_index', 'citation_text'
        ]

        # Include full text (truncated if needed)
        text_content = metadata.get('text', '')
        if len(text_content) > 8000:  # Conservative limit for metadata
            pinecone_metadata['text'] = text_content[:8000] + "..."
            pinecone_metadata['text_truncated'] = True
        else:
            pinecone_metadata['text'] = text_content
            pinecone_metadata['text_truncated'] = False

        # Add essential metadata
        for field in essential_fields:
            if field in metadata:
                value = metadata[field]
                # Convert to Pinecone-compatible type
                if isinstance(value, (str, int, float, bool)):
                    pinecone_metadata[field] = value
                elif value is not None:
                    pinecone_metadata[field] = str(value)

        # Add other important fields with type conversion
        additional_fields = {
            'source_journal': str,
            'source_doi': str,
            'hierarchy_breadcrumb': str,
            'preview': str,
            'chunk_size': int,
            'word_count': int,
            'sentence_count': int,
            'avg_word_length': float,
            'processing_version': str,
            'reference_url': str
        }

        for field, expected_type in additional_fields.items():
            if field in metadata and metadata[field] is not None:
                try:
                    pinecone_metadata[field] = expected_type(metadata[field])
                except (ValueError, TypeError):
                    pinecone_metadata[field] = str(metadata[field])

        # Add section hierarchy as a searchable string
        if 'section_hierarchy' in metadata and metadata['section_hierarchy']:
            hierarchy = metadata['section_hierarchy']
            if isinstance(hierarchy, list):
                pinecone_metadata['sections'] = " | ".join(hierarchy)
            else:
                pinecone_metadata['sections'] = str(hierarchy)

        # Add document type classification
        pinecone_metadata['doc_type'] = self._classify_document_type(metadata)

        # Add searchable tags
        pinecone_metadata['search_tags'] = self._generate_search_tags(metadata)

        return pinecone_metadata

    def _classify_document_type(self, metadata: Dict[str, Any]) -> str:
        """Classify document type for better filtering"""
        source_title = metadata.get('source_title', '').lower()
        source_journal = metadata.get('source_journal', '').lower()

        if any(term in source_title for term in ['journal', 'article', 'study']):
            return 'research_paper'
        elif any(term in source_title for term in ['guide', 'manual', 'handbook']):
            return 'clinical_guide'
        elif any(term in source_title for term in ['review', 'systematic']):
            return 'review_paper'
        elif source_journal:
            return 'journal_article'
        else:
            return 'document'

    def _generate_search_tags(self, metadata: Dict[str, Any]) -> str:
        """Generate searchable tags from metadata"""
        tags = []

        # Add year if available
        if metadata.get('source_year'):
            tags.append(f"year_{metadata['source_year']}")

        # Add author last names
        authors = metadata.get('source_authors', '')
        if authors:
            # Extract last names (simple heuristic)
            author_parts = authors.replace(',', ' ').split()
            potential_surnames = [part for part in author_parts if len(part) > 2 and part[0].isupper()]
            tags.extend(potential_surnames[:3])  # First 3 surnames

        # Add section types
        hierarchy = metadata.get('section_hierarchy', [])
        if hierarchy:
            section_types = []
            for section in hierarchy:
                section_lower = section.lower()
                if any(term in section_lower for term in ['method', 'result', 'discussion', 'conclusion']):
                    section_types.append(section_lower.split()[0])
            tags.extend(section_types)

        return " ".join(tags[:10])  # Limit to 10 tags

    def upload_enhanced_chunks(self, chunks_file: str, batch_size: int = 50) -> Dict[str, int]:
        """Upload chunks with enhanced metadata"""

        print(f"📤 Loading chunks from {chunks_file}...")

        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"🧠 Embedding and uploading {len(chunks)} chunks...")

        vectors = []
        stats = {"uploaded": 0, "skipped": 0, "errors": 0}

        for i, chunk_data in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                content = chunk_data.get('content', '')
                metadata = chunk_data.get('metadata', {})

                if not content.strip():
                    stats["skipped"] += 1
                    continue

                # Generate embedding
                input_text = f"passage: {content.strip()}"
                embedding = self.embedder.encode(input_text, normalize_embeddings=True)

                # Prepare metadata for Pinecone
                pinecone_metadata = self.prepare_metadata_for_pinecone(metadata)

                # Create vector
                vector_id = metadata.get('chunk_id', f"chunk_{i:04d}")
                vectors.append({
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": pinecone_metadata
                })

                # Upload in batches
                if len(vectors) >= batch_size:
                    self._upload_batch(vectors)
                    stats["uploaded"] += len(vectors)
                    vectors = []

            except Exception as e:
                print(f"❌ Error processing chunk {i}: {e}")
                stats["errors"] += 1
                continue

        # Upload final batch
        if vectors:
            self._upload_batch(vectors)
            stats["uploaded"] += len(vectors)

        return stats

    def _upload_batch(self, vectors: List[Dict]) -> None:
        """Upload a batch of vectors to Pinecone"""
        try:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
        except Exception as e:
            print(f"❌ Error uploading batch: {e}")
            raise

    def verify_upload(self, sample_query: str = "medical treatment") -> Dict[str, Any]:
        """Verify upload by running a test query"""
        print(f"🧪 Testing upload with query: '{sample_query}'")

        # Generate query embedding
        query_embedding = self.embedder.encode(f"query: {sample_query}", normalize_embeddings=True)

        # Search
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=3,
            namespace=self.namespace,
            include_metadata=True
        )

        # Analyze results
        verification = {
            "query": sample_query,
            "results_found": len(results.matches),
            "has_enhanced_metadata": False,
            "sample_metadata_fields": [],
            "sample_citation": None
        }

        if results.matches:
            sample_metadata = results.matches[0].metadata
            verification["sample_metadata_fields"] = list(sample_metadata.keys())
            verification["has_enhanced_metadata"] = 'citation_text' in sample_metadata
            verification["sample_citation"] = sample_metadata.get('citation_text')

            print(f"✅ Found {len(results.matches)} results")
            print(f"📋 Sample metadata fields: {verification['sample_metadata_fields'][:10]}")
            if verification["sample_citation"]:
                print(f"📄 Sample citation: {verification['sample_citation']}")

        return verification


def main():
    """Main function to enhance and upload chunks"""
    load_dotenv()

    # Configuration
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = "medical-rag-index"
    namespace = "thera-rag-enhanced"

    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not found in environment")
        return

    # Initialize uploader
    uploader = EnhancedPineconeUploader(pinecone_api_key, index_name, namespace)

    # Find enhanced chunk files
    enhanced_dir = "src/Ingestion/enhanced_outputs"
    if not os.path.exists(enhanced_dir):
        print(f"❌ Enhanced chunks directory not found: {enhanced_dir}")
        print("   Run enhanced_ingestion.py first to create enhanced metadata")
        return

    enhanced_files = [f for f in os.listdir(enhanced_dir) if f.startswith('enhanced_') and f.endswith('.json')]

    if not enhanced_files:
        print(f"❌ No enhanced chunk files found in {enhanced_dir}")
        return

    print(f"📁 Found {len(enhanced_files)} enhanced chunk files")

    # Upload each file
    total_stats = {"uploaded": 0, "skipped": 0, "errors": 0}

    for chunk_file in enhanced_files:
        print(f"\n📤 Uploading {chunk_file}...")
        file_path = os.path.join(enhanced_dir, chunk_file)

        try:
            stats = uploader.upload_enhanced_chunks(file_path)
            for key in total_stats:
                total_stats[key] += stats[key]

            print(f"   ✅ Uploaded: {stats['uploaded']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")

        except Exception as e:
            print(f"   ❌ Failed to upload {chunk_file}: {e}")

    # Print summary
    print(f"\n📊 Upload Summary:")
    print(f"   Total uploaded: {total_stats['uploaded']}")
    print(f"   Total skipped: {total_stats['skipped']}")
    print(f"   Total errors: {total_stats['errors']}")

    # Verify upload
    if total_stats['uploaded'] > 0:
        verification = uploader.verify_upload()
        if verification["has_enhanced_metadata"]:
            print(f"\n✅ Enhanced metadata successfully uploaded!")
        else:
            print(f"\n⚠️ Upload successful but enhanced metadata may not be complete")


if __name__ == "__main__":
    main()