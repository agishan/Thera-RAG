# Usage Guide

Complete guide for using the Medical Document Ingestion Package in various scenarios.

## Table of Contents
- [Getting Started](#getting-started)
- [Basic Usage Patterns](#basic-usage-patterns)
- [Advanced Configuration](#advanced-configuration)
- [Use Case Examples](#use-case-examples)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Getting Started

### Prerequisites

1. **Install dependencies**:
```bash
pip install docling langchain langchain-google-genai sentence-transformers pinecone-client python-dotenv tqdm
```

2. **Set up environment variables**:
```bash
# Create .env file
echo "PINECONE_API_KEY=your_pinecone_api_key" >> .env
echo "GOOGLE_API_KEY=your_google_api_key" >> .env
```

3. **Verify Pinecone index**:
```python
from pinecone import Pinecone
pc = Pinecone(api_key="your_key")
print(pc.list_indexes())  # Should show your index
```

### First Steps

```python
from ingestion_package import process_documents

# Process a single file quickly
results = process_documents(
    input_path="sample_document.pdf",
    output_dir="output/",
    enable_upload=False  # Start without uploading
)

print(f"Success: Created {results[0]['chunks_created']} chunks")
```

---

## Basic Usage Patterns

### Pattern 1: Local Processing Only

Perfect for testing, analysis, or when you don't need vector database integration.

```python
from ingestion_package import IngestionPipeline

# Initialize without vector upload
pipeline = IngestionPipeline(
    enable_upload=False,
    enable_citations=True  # Still extract citations
)

# Process documents
result = pipeline.process_single_file(
    "research_paper.pdf",
    output_dir="analysis/",
    save_intermediate=True
)

# Analyze results
print(f"Document: {result['document_metadata']['source_title']}")
print(f"Authors: {result['document_metadata']['source_authors']}")
print(f"Chunks: {result['chunks_created']}")
print(f"Citations: {result['citations_found']}")
```

**Output files created**:
- `research_paper.md` - Extracted markdown
- `research_paper_chunks.json` - Processed chunks
- `research_paper_citations.json` - Citations and matches
- `research_paper_metadata.json` - Document metadata
- `research_paper_analysis.json` - Chunk statistics

### Pattern 2: Full Pipeline with Vector Upload

Complete processing with vector database integration.

```python
import os
from ingestion_package import IngestionPipeline, ChunkingConfig

# Load API keys
pinecone_key = os.getenv("PINECONE_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

# Configure chunking
config = ChunkingConfig(
    target_chunk_size=3500,
    overlap_size=350,
    section_aware=True
)

# Initialize full pipeline
pipeline = IngestionPipeline(
    chunking_config=config,
    pinecone_api_key=pinecone_key,
    google_api_key=google_key,
    index_name="medical-docs",
    namespace="clinical-papers",
    enable_citations=True,
    enable_upload=True
)

# Process directory
results = pipeline.process_directory(
    input_dir="medical_papers/",
    output_dir="processed/",
    file_pattern="*.pdf"
)

# Get processing statistics
stats = pipeline.get_processing_stats()
print(f"Files processed: {stats['files_processed']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Vectors uploaded: {stats['vectors_uploaded']}")

# Verify upload
verification = pipeline.verify_upload("diabetes treatment")
print(f"Upload verified: {verification['query_test']['test_passed']}")
```

### Pattern 3: Batch Processing

Process large document collections efficiently.

```python
from ingestion_package import IngestionPipeline
import os
from pathlib import Path

def process_document_collection(base_dir, collection_name):
    """Process a collection of documents with consistent naming"""

    pipeline = IngestionPipeline(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        namespace=f"collection_{collection_name}",
        enable_citations=True,
        enable_upload=True
    )

    input_dir = Path(base_dir) / collection_name
    output_dir = Path("processed") / collection_name

    print(f"Processing collection: {collection_name}")

    results = pipeline.process_directory(
        str(input_dir),
        str(output_dir),
        "*.pdf"
    )

    # Summary report
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"Collection {collection_name}:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total chunks: {sum(r.get('chunks_created', 0) for r in successful)}")

    return results

# Process multiple collections
collections = ["clinical_guidelines", "research_papers", "case_studies"]
all_results = {}

for collection in collections:
    all_results[collection] = process_document_collection("documents", collection)
```

---

## Advanced Configuration

### Custom Chunking Strategies

#### Large Context Chunks
For documents where you need maximum context preservation:

```python
from ingestion_package import ChunkingConfig, IngestionPipeline

# Large chunks with significant overlap
large_context_config = ChunkingConfig(
    target_chunk_size=6000,    # Large chunks
    overlap_size=1200,         # 20% overlap
    max_chunk_size=8000,       # Allow even larger if needed
    section_aware=True,        # Preserve sections
    preserve_sentences=True    # Don't break sentences
)

pipeline = IngestionPipeline(chunking_config=large_context_config)
```

#### Fine-Grained Chunks
For precise retrieval and detailed analysis:

```python
# Small, precise chunks
fine_grained_config = ChunkingConfig(
    target_chunk_size=1500,    # Small chunks
    overlap_size=150,          # 10% overlap
    min_chunk_size=300,        # Allow smaller chunks
    preserve_sentences=True,   # Maintain sentence integrity
    section_aware=False        # Use recursive splitting
)

pipeline = IngestionPipeline(chunking_config=fine_grained_config)
```

#### Section-Based Chunking
For documents with clear hierarchical structure:

```python
# Section-focused chunking
section_config = ChunkingConfig(
    target_chunk_size=4000,
    overlap_size=200,          # Minimal overlap
    section_aware=True,        # Primary strategy
    preserve_paragraphs=True   # Maintain paragraph structure
)

pipeline = IngestionPipeline(chunking_config=section_config)
```

### Citation Extraction Configurations

#### High-Precision Citation Extraction
```python
# Focus on citation quality
pipeline = IngestionPipeline(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    enable_citations=True,
    enable_upload=False  # Process locally for analysis
)

# Process and analyze citations
result = pipeline.process_single_file("research_paper.pdf")

# Access citation data
citations = result.get('citations_found', 0)
matches = result.get('citation_matches', 0)
print(f"Found {citations} citations, matched {matches} inline references")
```

#### Regex-Only Citation Extraction
For faster processing or when LLM access is limited:

```python
# Initialize without LLM
pipeline = IngestionPipeline(
    google_api_key=None,  # No LLM access
    enable_citations=True  # Still extract using regex
)
```

### Vector Database Configurations

#### Multiple Namespaces
Organize different document types in separate namespaces:

```python
def create_specialized_pipeline(doc_type, namespace_suffix):
    return IngestionPipeline(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="medical-docs",
        namespace=f"{doc_type}_{namespace_suffix}",
        enable_upload=True
    )

# Create specialized pipelines
research_pipeline = create_specialized_pipeline("research", "v1")
clinical_pipeline = create_specialized_pipeline("clinical", "v1")
guidelines_pipeline = create_specialized_pipeline("guidelines", "v1")

# Process different document types
research_results = research_pipeline.process_directory("research_papers/")
clinical_results = clinical_pipeline.process_directory("clinical_trials/")
guidelines_results = guidelines_pipeline.process_directory("guidelines/")
```

#### Custom Embedding Models
```python
from ingestion_package.vector_uploader import EnhancedVectorUploader

# Use different embedding model
uploader = EnhancedVectorUploader(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="medical-docs",
    namespace="custom-embeddings",
    embedding_model="all-MiniLM-L6-v2"  # Different model
)

# Manual upload process
chunks = chunker.create_chunks(text, metadata)
stats = uploader.upload_chunks(chunks)
```

---

## Use Case Examples

### Use Case 1: Literature Review Assistant

Create a system for literature review and citation analysis.

```python
class LiteratureReviewProcessor:
    def __init__(self):
        self.pipeline = IngestionPipeline(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            namespace="literature_review",
            enable_citations=True,
            enable_upload=True
        )

        # Track processed papers
        self.processed_papers = {}
        self.citation_network = {}

    def process_paper_collection(self, papers_dir):
        """Process a collection of research papers"""
        results = self.pipeline.process_directory(papers_dir, "processed_papers/")

        for result in results:
            if result.get("success"):
                paper_id = Path(result["file_path"]).stem
                self.processed_papers[paper_id] = {
                    "metadata": result["document_metadata"],
                    "chunks": result["chunks_created"],
                    "citations": result["citations_found"]
                }

        return results

    def generate_literature_summary(self):
        """Generate summary of processed literature"""
        total_papers = len(self.processed_papers)
        total_citations = sum(p["citations"] for p in self.processed_papers.values())

        print(f"Literature Review Summary:")
        print(f"Papers processed: {total_papers}")
        print(f"Total citations extracted: {total_citations}")

        # Most cited authors
        authors = {}
        for paper in self.processed_papers.values():
            author = paper["metadata"].get("source_authors", "Unknown")
            authors[author] = authors.get(author, 0) + 1

        print("Most frequent authors:")
        for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {author}: {count} papers")

# Usage
processor = LiteratureReviewProcessor()
results = processor.process_paper_collection("literature_papers/")
processor.generate_literature_summary()
```

### Use Case 2: Clinical Guidelines Database

Build a searchable database of clinical guidelines.

```python
class ClinicalGuidelinesDB:
    def __init__(self):
        self.pipeline = IngestionPipeline(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            index_name="clinical-guidelines",
            namespace="guidelines_v2",
            chunking_config=ChunkingConfig(
                target_chunk_size=2500,  # Smaller for precise retrieval
                section_aware=True       # Preserve guideline structure
            ),
            enable_citations=False,      # Guidelines may not have citations
            enable_upload=True
        )

    def process_guidelines(self, guidelines_dir):
        """Process clinical guidelines with special handling"""
        results = self.pipeline.process_directory(
            guidelines_dir,
            "processed_guidelines/",
            "*.pdf"
        )

        # Categorize guidelines
        categories = {}
        for result in results:
            if result.get("success"):
                doc_metadata = result["document_metadata"]
                title = doc_metadata.get("source_title", "")

                # Simple categorization based on title
                category = self._categorize_guideline(title)
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)

        return categories

    def _categorize_guideline(self, title):
        """Categorize guidelines based on title"""
        title_lower = title.lower()

        if any(term in title_lower for term in ["diabetes", "glucose", "insulin"]):
            return "diabetes"
        elif any(term in title_lower for term in ["hypertension", "blood pressure"]):
            return "hypertension"
        elif any(term in title_lower for term in ["depression", "anxiety", "mental"]):
            return "mental_health"
        elif any(term in title_lower for term in ["cardiac", "heart", "cardiovascular"]):
            return "cardiovascular"
        else:
            return "general"

# Usage
guidelines_db = ClinicalGuidelinesDB()
categorized_guidelines = guidelines_db.process_guidelines("clinical_guidelines/")

for category, guidelines in categorized_guidelines.items():
    print(f"{category.title()}: {len(guidelines)} guidelines")
```

### Use Case 3: Research Data Extraction

Extract specific data points from research papers.

```python
class ResearchDataExtractor:
    def __init__(self):
        # Focus on local processing for data extraction
        self.pipeline = IngestionPipeline(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            enable_citations=True,
            enable_upload=False  # Local processing
        )

        self.extracted_data = []

    def extract_study_data(self, papers_dir):
        """Extract key data from research papers"""
        results = self.pipeline.process_directory(papers_dir, "extracted_data/")

        for result in results:
            if result.get("success"):
                study_data = self._extract_study_info(result)
                self.extracted_data.append(study_data)

        return self.extracted_data

    def _extract_study_info(self, result):
        """Extract specific study information"""
        metadata = result["document_metadata"]

        return {
            "title": metadata.get("source_title"),
            "authors": metadata.get("source_authors"),
            "year": metadata.get("source_year"),
            "journal": metadata.get("source_journal"),
            "doi": metadata.get("source_doi"),
            "chunks_count": result["chunks_created"],
            "citations_count": result["citations_found"],
            "document_type": metadata.get("document_type")
        }

    def export_to_csv(self, filename):
        """Export extracted data to CSV"""
        import csv

        if not self.extracted_data:
            print("No data to export")
            return

        fieldnames = self.extracted_data[0].keys()

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.extracted_data)

        print(f"Exported {len(self.extracted_data)} studies to {filename}")

# Usage
extractor = ResearchDataExtractor()
studies = extractor.extract_study_data("research_papers/")
extractor.export_to_csv("extracted_studies.csv")
```

---

## Performance Optimization

### Memory Management

#### Processing Large Document Collections
```python
def process_large_collection(base_dir, batch_size=10):
    """Process large collections in batches to manage memory"""

    from pathlib import Path
    import gc

    pipeline = IngestionPipeline(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        enable_upload=True
    )

    pdf_files = list(Path(base_dir).glob("*.pdf"))
    total_files = len(pdf_files)

    print(f"Processing {total_files} files in batches of {batch_size}")

    all_results = []
    for i in range(0, total_files, batch_size):
        batch_files = pdf_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1}")

        # Process batch
        batch_results = []
        for file_path in batch_files:
            result = pipeline.process_single_file(
                str(file_path),
                save_intermediate=False  # Save memory
            )
            batch_results.append(result)

        all_results.extend(batch_results)

        # Force garbage collection
        gc.collect()

        print(f"Batch completed. Processed: {len(batch_results)} files")

    return all_results
```

#### Chunking Optimization
```python
# For memory-constrained environments
memory_optimized_config = ChunkingConfig(
    target_chunk_size=2000,    # Smaller chunks
    overlap_size=200,          # Less overlap
    max_chunk_size=3000        # Strict size limits
)

# For speed-optimized processing
speed_optimized_config = ChunkingConfig(
    target_chunk_size=4000,    # Larger chunks (fewer total)
    section_aware=False,       # Skip section detection
    preserve_paragraphs=False  # Simpler splitting
)
```

### Upload Optimization

#### Batch Size Tuning
```python
from ingestion_package.vector_uploader import EnhancedVectorUploader

# For stable networks - larger batches
fast_uploader = EnhancedVectorUploader(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="medical-docs",
    namespace="fast_upload"
)

# Upload with larger batches
stats = fast_uploader.upload_chunks(chunks, batch_size=100)

# For unstable networks - smaller batches
reliable_uploader = EnhancedVectorUploader(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="medical-docs",
    namespace="reliable_upload"
)

# Upload with smaller, more reliable batches
stats = reliable_uploader.upload_chunks(chunks, batch_size=25)
```

### Citation Extraction Optimization

#### Disable Features for Speed
```python
# Fast processing - minimal features
fast_pipeline = IngestionPipeline(
    enable_citations=False,    # Skip citation extraction
    enable_upload=True,
    chunking_config=ChunkingConfig(section_aware=False)
)

# Citation-focused - slower but comprehensive
citation_pipeline = IngestionPipeline(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    enable_citations=True,     # Full citation extraction
    enable_upload=False        # Local analysis only
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Failed to extract text from PDF"
**Causes**: Corrupted PDF, unsupported format, or OCR issues

**Solutions**:
```python
# 1. Check PDF file
import PyPDF2
with open("document.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    print(f"Pages: {len(reader.pages)}")

# 2. Try with OCR disabled
loader = DoclingBookLoader("document.pdf", do_ocr=False)

# 3. Use different processing options
loader = DoclingBookLoader(
    "document.pdf",
    num_threads=4,           # Reduce threads
    accelerator_device="cpu" # Force CPU processing
)
```

#### Issue: "Pinecone upload failed"
**Causes**: Network issues, quota limits, or dimension mismatches

**Solutions**:
```python
# 1. Verify index configuration
from pinecone import Pinecone
pc = Pinecone(api_key="your_key")
index_info = pc.describe_index("your_index")
print(index_info)

# 2. Check embeddings dimension
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/e5-base")
test_embedding = model.encode("test")
print(f"Embedding dimension: {len(test_embedding)}")

# 3. Use smaller batches
uploader.upload_chunks(chunks, batch_size=10)
```

#### Issue: "Citation extraction returns no results"
**Causes**: Document format, missing references section, or API issues

**Solutions**:
```python
# 1. Check for references section manually
with open("extracted_text.md", "r") as f:
    text = f.read()
    if "references" in text.lower():
        print("References section found")
    else:
        print("No references section detected")

# 2. Use regex-only extraction
extractor = CitationExtractor(use_llm=False)

# 3. Check Google API key
import os
if not os.getenv("GOOGLE_API_KEY"):
    print("Google API key not set")
```

#### Issue: "Memory errors with large documents"
**Solutions**:
```python
# 1. Use smaller chunk sizes
config = ChunkingConfig(target_chunk_size=1500)

# 2. Disable intermediate file saving
result = pipeline.process_single_file("large_doc.pdf", save_intermediate=False)

# 3. Process in smaller batches
for pdf_file in pdf_files:
    result = pipeline.process_single_file(pdf_file)
    # Process immediately, don't accumulate
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add custom logging to pipeline
class DebugPipeline(IngestionPipeline):
    def process_single_file(self, *args, **kwargs):
        print(f"Processing: {args[0]}")

        try:
            result = super().process_single_file(*args, **kwargs)
            print(f"Success: {result['chunks_created']} chunks created")
            return result
        except Exception as e:
            print(f"Error: {e}")
            raise

debug_pipeline = DebugPipeline()
```

---

## Best Practices

### 1. Project Organization

```
project/
├── documents/
│   ├── research_papers/
│   ├── clinical_guidelines/
│   └── case_studies/
├── processed/
│   ├── research_papers/
│   ├── clinical_guidelines/
│   └── case_studies/
├── scripts/
│   ├── process_research.py
│   ├── process_guidelines.py
│   └── analyze_citations.py
├── config/
│   ├── research_config.py
│   └── guidelines_config.py
└── .env
```

### 2. Configuration Management

```python
# config/base_config.py
import os
from ingestion_package import ChunkingConfig

BASE_CONFIG = {
    "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
    "google_api_key": os.getenv("GOOGLE_API_KEY"),
    "index_name": "medical-docs"
}

RESEARCH_CONFIG = ChunkingConfig(
    target_chunk_size=4000,
    overlap_size=400,
    section_aware=True
)

GUIDELINES_CONFIG = ChunkingConfig(
    target_chunk_size=2500,
    overlap_size=250,
    section_aware=True
)
```

### 3. Error Handling

```python
def robust_processing(file_path, max_retries=3):
    """Process file with retry logic"""

    for attempt in range(max_retries):
        try:
            result = pipeline.process_single_file(file_path)
            if result.get("success"):
                return result
            else:
                print(f"Attempt {attempt + 1} failed: {result.get('error')}")
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff

    return {"success": False, "error": "All retries failed"}
```

### 4. Quality Assurance

```python
def validate_processing_results(results):
    """Validate processing results for quality"""

    issues = []

    for result in results:
        if not result.get("success"):
            issues.append(f"Failed: {result['file_path']}")
            continue

        # Check chunk count
        chunks = result.get("chunks_created", 0)
        if chunks < 5:
            issues.append(f"Too few chunks ({chunks}): {result['file_path']}")

        # Check metadata completeness
        metadata = result.get("document_metadata", {})
        if not metadata.get("source_title"):
            issues.append(f"Missing title: {result['file_path']}")

    if issues:
        print("Quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All processing results passed quality checks")

    return len(issues) == 0
```

### 5. Monitoring and Logging

```python
import time
from datetime import datetime

class ProcessingMonitor:
    def __init__(self):
        self.start_time = None
        self.processed_files = 0
        self.total_chunks = 0

    def start_monitoring(self, total_files):
        self.start_time = time.time()
        self.total_files = total_files
        print(f"Starting processing of {total_files} files at {datetime.now()}")

    def update_progress(self, result):
        self.processed_files += 1
        if result.get("success"):
            self.total_chunks += result.get("chunks_created", 0)

        elapsed = time.time() - self.start_time
        rate = self.processed_files / elapsed if elapsed > 0 else 0

        print(f"Progress: {self.processed_files}/{self.total_files} "
              f"({self.processed_files/self.total_files*100:.1f}%) "
              f"Rate: {rate:.2f} files/sec")

    def finish_monitoring(self):
        elapsed = time.time() - self.start_time
        print(f"Completed in {elapsed:.1f}s")
        print(f"Total chunks created: {self.total_chunks}")
        print(f"Average chunks per file: {self.total_chunks/self.processed_files:.1f}")

# Usage
monitor = ProcessingMonitor()
monitor.start_monitoring(len(pdf_files))

for pdf_file in pdf_files:
    result = pipeline.process_single_file(pdf_file)
    monitor.update_progress(result)

monitor.finish_monitoring()
```

This comprehensive usage guide covers all major scenarios and best practices for using the Medical Document Ingestion Package effectively.