# RAG vs Vanilla Comparison Experiments

This directory contains scripts and tools for comparing RAG (Retrieval-Augmented Generation) responses against vanilla LLM responses.

## Files

- `comparison_eval.py` - Main comparison script
- `sample_questions.csv` - Example input CSV with clinical questions
- `README.md` - This documentation

## Usage

### Basic Usage
```bash
cd experiments
python comparison_eval.py sample_questions.csv results.csv
```

### Advanced Usage
```bash
python comparison_eval.py input.csv output.csv --k-chunks 20 --k-citation-chunks 15 --delay 45.0
```

### Arguments
- `input_csv` - Path to CSV file with questions
- `output_csv` - Path to output CSV file for results
- `--k-chunks` - Number of chunks to retrieve (default: 15)
- `--k-citation-chunks` - Number of citation chunks (default: 10)
- `--delay` - Delay between API calls in seconds (default: 30.0)

## Input CSV Format

Required columns:
- `question` - The clinical questions to evaluate

Optional columns:
- `k_chunks` - Override default chunks for specific questions
- `k_citation_chunks` - Override default citation chunks for specific questions

## Output CSV Format

The output CSV includes all input columns plus:

### Response Columns
- `rag_response` - Response using retrieved chunks and citations
- `vanilla_response` - Response without any context
- `rag_response_cleaned` - Bias-cleansed RAG response
- `vanilla_response_cleaned` - Bias-cleansed vanilla response

### Metadata Columns
- `retrieved_chunks` - Number of chunks retrieved for RAG
- `citation_chunks` - Number of chunks with citations extracted
- `citations_found` - Formatted citations from retrieved chunks
- `rag_response_time` - Time taken for RAG response (seconds)
- `vanilla_response_time` - Time taken for vanilla response (seconds)
- `rag_cleansing_time` - Time taken to cleanse RAG response (seconds)
- `vanilla_cleansing_time` - Time taken to cleanse vanilla response (seconds)

### Error Tracking
- `error_rag` - Any errors during RAG processing
- `error_vanilla` - Any errors during vanilla processing
- `error_rag_cleansing` - Any errors during RAG response cleansing
- `error_vanilla_cleansing` - Any errors during vanilla response cleansing
- `processed_at` - Timestamp when question was processed

## Rate Limiting

The script implements rate limiting to respect Gemini API limits:
- Default: 30 seconds between API calls
- Each question makes 4 API calls (RAG, Vanilla, Cleanse RAG, Cleanse Vanilla)
- Estimated time: ~2 minutes per question with default settings

## Requirements

- All dependencies from the main inference app
- Valid API keys in configuration
- Access to the Pinecone vector database