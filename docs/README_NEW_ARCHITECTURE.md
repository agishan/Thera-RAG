# Thera-RAG: Two-App Architecture

## Overview

The system is now split into two focused applications:

### 🔄 **Ingestion App** (Local Only)
- **Purpose**: PDF → Pinecone pipeline with HITL control
- **Location**: `apps/ingestion/`
- **Usage**: Local document processing, quality control, batch processing

### 💬 **Inference App** (Streamlit Deployment)
- **Purpose**: RAG chatbot for medical queries
- **Location**: `apps/inference/`
- **Usage**: Web-based chat interface, deployed to Streamlit Cloud

## Usage

### Ingestion CLI\n```bash\n# Initialize and run Stage 1\npython apps/ingestion/main.py init --pdf data/vha-guideline.pdf --text-only\npython apps/ingestion/main.py run --pdf data/vha-guideline.pdf\n\n# Approve and proceed\npython apps/ingestion/main.py approve --pdf data/vha-guideline.pdf --stage 1\npython apps/ingestion/main.py run --pdf data/vha-guideline.pdf\n```\n\n### Ingestion App
```bash
# New ingestion app (same functionality, organized better)
python apps/ingestion/main.py stage --stage 2.5 --density-threshold 0.7

# Future: Batch processing
python apps/ingestion/main.py batch "data/*.pdf" --config medical_precise

# Future: Quality verification
python apps/ingestion/main.py verify --index medical-papers
```

### New Inference App
```bash
# Run the Streamlit chatbot
cd apps/inference
streamlit run main.py

# Or from project root
python apps/inference/main.py
```

## Directory Structure

```
thera-rag/
├── apps/
│   ├── ingestion/              # LOCAL PROCESSING
│   │   ├── src/                # All ingestion processors
│   │   ├── (deprecated)/
│   │   ├── main.py             # New CLI entry point
│   │   └── requirements.txt    # Ingestion dependencies
│   │
│   └── inference/              # STREAMLIT CHATBOT
│       ├── src/                # All chat/RAG components
│       ├── main.py             # Streamlit entry point
│       └── requirements.txt    # Inference dependencies
│
├── shared/                     # COMMON COMPONENTS
│   ├── models/                 # Pinecone schema, data models
│   ├── config/                 # Shared configuration
│   └── utils/                  # Common utilities
│
├── scripts/                    # OLD LOCATION (still works)
├── src/                        # OLD LOCATION (still works)
└── requirements.txt            # OLD REQUIREMENTS (still works)
```

## Migration Status

### ✅ Phase 1: Complete
- [x] Created monorepo structure
- [x] Moved ingestion components to `apps/ingestion/`
- [x] Moved inference components to `apps/inference/`
- [x] Set up shared configuration system
- [x] **Your current workflow still works unchanged**

### 🔄 Phase 2: Next Steps
- [ ] Test new ingestion app entry point
- [ ] Update import paths in new locations
- [ ] Test inference app deployment
- [ ] Enhanced HITL features

## Key Benefits

### 🎯 **Separation of Concerns**
- **Ingestion**: 100% focused on getting quality data into Pinecone
- **Inference**: 100% focused on fast, accurate RAG responses

### 🔧 **Development Benefits**
- Independent dependency management
- Cleaner testing (test ingestion separate from chatbot)
- Faster development cycles
- No more mixing UI code with document processing

### 🚀 **Deployment Benefits**
- **Ingestion**: Local only, no deployment needed
- **Inference**: Clean Streamlit deployment with minimal dependencies
- Independent scaling and updates

### 📊 **Your HITL Focus**
- All HITL features go in `apps/ingestion/`
- Clean separation from user-facing chat interface
- Better control and quality assurance tools

## Next Steps

1. **Test the new ingestion app**: Try the new CLI interface
2. **Verify inference app**: Make sure Streamlit still works
3. **Enhanced HITL features**: Add the interactive controls you need
4. **Gradual migration**: Use new structure when ready, keep old as backup

**Your current workflow is preserved - no rush to migrate!**

## Detailed Ingestion Flow

For a deep dive into the ingestion pipeline stages, parameters, and artifacts, see:

- `docs/INGESTION_PIPELINE.md` — includes Mermaid flow diagrams and per‑stage documentation.


