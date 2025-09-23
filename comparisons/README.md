# RAG Comparison Pipeline

Clean RAG vs Vanilla LLM comparison with template-based prompts.

## Usage

### Basic Comparison
```bash
python rag_comparison.py "How should I interpret R-time prolongation on TEG?"
```

## Prompt Types

The system uses 3 template-based prompts:

1. **RAG Prompt** (`medical_rag`) - Medical Q&A with retrieved context
2. **Vanilla Prompt** (`vanilla_medical`) - Medical Q&A without context
3. **Cleansing Prompt** (`cleansing_medical`) - Removes bias indicators for fair comparison

## Features

- **Template-Based Architecture**: All prompts managed through the template system
- **Bias-Cleansed Comparisons**: Fair comparison between RAG and vanilla outputs
- **Citation Integration**: Automatic citation extraction and formatting
- **Structured Output**: JSON and text summary formats
- **Simplified Interface**: Single command, no complex options

## Prompt Management

Prompts are stored in `src/app/prompts/templates/`:
- `medical_rag.txt` - RAG prompt
- `vanilla_medical.txt` - Vanilla LLM prompt
- `cleansing_medical.txt` - Bias removal prompt

## Integration

Uses the same prompt management system as the main RAG application for consistency.