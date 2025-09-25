# Pinecone Upload CLI Commands

## Overview

Added **`upload`** and **`verify-upload`** commands to the Thera-RAG ingestion CLI for seamless Pinecone vector database deployment.

## New Commands

### **`upload` - Upload Optimized Chunks to Pinecone**

Uploads processed document chunks to Pinecone with optimized metadata for inference.

**Basic Usage:**
```bash
# Set your Pinecone API key
export PINECONE_API_KEY="your-pinecone-api-key"

# Upload with defaults (uses medical-rag-index and thera-rag namespace)
python main.py upload --pdf data/vha-guideline.pdf
```

**Advanced Usage:**
```bash
# Upload with custom index/namespace
python main.py upload --pdf data/vha-guideline.pdf --index my-index --namespace my-namespace

# Upload with specific metadata profile
python main.py upload --pdf data/vha-guideline.pdf --profile MINIMAL

# Upload using document name instead of PDF path
python main.py upload --doc vha-guideline
```

**Options:**
- `--pdf`: Path to PDF (default: `data/vha-guideline.pdf`)
- `--doc`: Document name (alternative to --pdf)
- `--index`: Pinecone index name (default: `medical-rag-index`)
- `--namespace`: Pinecone namespace (default: `thera-rag`)
- `--profile`: Metadata profile - `MINIMAL`, `INFERENCE` (default), or `FULL`

### **`verify-upload` - Verify Pinecone Upload**

Tests the uploaded vectors and verifies retrieval functionality.

**Basic Usage:**
```bash
# Verify upload with defaults
python main.py verify-upload
```

**Advanced Usage:**
```bash
# Verify specific index/namespace
python main.py verify-upload --index my-index --namespace my-namespace

# Test with custom query
python main.py verify-upload --query "ROTEM analysis"

# Check expected chunk count
python main.py verify-upload --expected-chunks 33
```

**Options:**
- `--index`: Pinecone index name (default: `medical-rag-index`)
- `--namespace`: Pinecone namespace (default: `thera-rag`)
- `--query`: Test query (default: `medical treatment`)
- `--expected-chunks`: Expected number of chunks for validation

## Complete Workflow Example

```bash
# 1. Set API keys
export PINECONE_API_KEY="your-pinecone-api-key"
export GOOGLE_API_KEY="your-google-api-key"

# 2. Run complete pipeline (Stages 1-5)
python main.py run --pdf data/vha-guideline.pdf --auto-continue

# 3. Upload optimized chunks to Pinecone
python main.py upload --pdf data/vha-guideline.pdf

# 4. Verify the upload
python main.py verify-upload --expected-chunks 33
```

## Automatic Optimizations

### **Smart Chunk Selection**
The `upload` command automatically uses the best available chunks:
1. **Stage 5 optimized chunks** (preferred) - `enhanced_vha-guideline_vector_ready.json`
2. **Stage 4 chunks** (fallback) - `enhanced_vha-guideline.json`

### **Metadata Profiles**

| Profile | Size | Use Case |
|---------|------|----------|
| **MINIMAL** | ~2KB/chunk | Basic retrieval, maximum speed |
| **INFERENCE** | ~5-8KB/chunk | **Default** - Full RAG capability |
| **FULL** | ~15KB+/chunk | Debug mode with all metadata |

### **Size Monitoring**
- **Warns** at 30KB metadata per chunk
- **Truncates** at 35KB to stay under Pinecone's 40KB limit
- **Reports** size statistics after upload

## Output Examples

### Upload Output
```
Using chunks: enhanced_output/vha-guideline/5_citation_index/enhanced_vha-guideline_vector_ready.json
Uploading 33 chunks to Pinecone...
   Index: medical-rag-index
   Namespace: thera-rag
   Profile: INFERENCE

Processing chunks: 100%|████████| 33/33 [00:15<00:00,  2.15it/s]

Upload complete:
   Uploaded: 33 chunks
   Skipped: 0 chunks
   Errors: 0 chunks
   Saved upload info to manifest
```

### Verification Output
```
Upload Verification Results:
   Namespace: thera-rag
   Total vectors: 33
   Upload success: True

Query Test ('medical treatment'):
   Results found: 3
   Test passed: True

Metadata Quality (Score: 92.1):
   has_full_text: 100.0%
   has_source_info: 100.0%
   has_location: 100.0%
   has_citations: 85.7%
   has_search_tags: 100.0%

Sample Metadata:
   Full text preserved: True
   Citation info: True
   Location data: True
   Metadata size: 5.2KB
   Profile: INFERENCE
```

## Environment Variables

```bash
# Required
export PINECONE_API_KEY="your-pinecone-api-key"

# Optional - Metadata Control
export PINECONE_METADATA_PROFILE="INFERENCE"  # MINIMAL, INFERENCE, FULL
export PINECONE_METADATA_WARN_SIZE_KB="30"    # Warn at 30KB
export PINECONE_METADATA_MAX_SIZE_KB="35"     # Truncate at 35KB
```

## Integration with Citation System

The upload commands work seamlessly with the citation optimization:

1. **Stage 4**: Creates full citation objects (bloated)
2. **Stage 5**: Optimizes to lightweight references + citation index
3. **Upload**: Uses optimized chunks for Pinecone storage
4. **Inference**: Resolves citations using the citation index

This maintains full citation capability while dramatically reducing vector storage costs.

## Error Handling

- **Missing API Key**: Clear error message with setup instructions
- **Missing Chunks**: Suggests running pipeline stages first
- **Upload Failures**: Detailed error messages with troubleshooting hints
- **Size Warnings**: Automatic metadata truncation when approaching limits

## Next Steps

After upload, the chunks are ready for RAG inference! Use the citation resolver in your inference pipeline to get full citation information when needed:

```python
from apps.ingestion.src.citation_resolver import CitationResolver

resolver = CitationResolver('enhanced_output/vha-guideline/5_citation_index/citation_index.json')
resolved_chunks = resolver.resolve_chunks_citations(retrieved_chunks)
```