# Project Cleanup Plan: Citation-Aware Medical RAG

## Current Problems

This project has accumulated significant complexity and duplication:
- **3 different ingestion systems** (notebook, core, package)
- **5+ run scripts** with overlapping functionality
- **7+ documentation files** scattered across directories
- **Duplicate components** (retrievers, uploaders, configs)
- **Legacy comparison files** with timestamps
- **Unclear organization** and file structure

## Cleanup Strategy: Focus on Citation Extraction

**Goal**: Transform this into a clean, focused system that showcases LLM citation extraction as the killer feature.

---

## Phase 1: Remove Legacy & Duplicates

### 🗑️ Files to DELETE

#### Legacy Ingestion (Replaced by citation-focused system)
```bash
rm -rf src/Ingestion/                     # Legacy notebook-based ingestion
rm run_ingestion.py                       # Legacy run script
rm run_simple_ingestion.py               # Intermediate version
rm src/app/enhanced_ingestion.py         # Duplicate ingestion
rm src/app/enhanced_uploader.py          # Duplicate uploader
```

#### Duplicate Documentation (Consolidate into main README)
```bash
rm CITATION_FOCUSED_README.md            # Merge into main README
rm src/ingestion_package/README.md       # Package-specific docs
rm src/ingestion_package/API_REFERENCE.md
rm src/ingestion_package/USAGE_GUIDE.md
rm DEPLOYMENT.md                         # Outdated deployment info
rm PROMPT_ARCHITECTURE.md               # Legacy prompt docs
```

#### Legacy Comparison Files
```bash
rm -rf comparisons/                       # Timestamped comparison results
rm run_comparison_pipeline.py            # Comparison scripts
rm run_test_pipeline.py                  # Test pipeline scripts
```

#### Duplicate App Components
```bash
rm streamlit_app.py                       # Duplicate of src/app/main.py
rm src/app/retrievers.py                 # Superseded by enhanced_retriever.py
rm src/app/references.json               # Legacy reference file
```

#### Over-engineered Package Components
```bash
rm src/ingestion_package/setup.py        # Over-engineered packaging
rm src/ingestion_package/requirements.txt # Duplicate requirements
rm src/ingestion_package/example_usage.py # Superseded by main run script
```

#### Legacy Verification Scripts
```bash
rm verify_quality.py                     # Keep only verify_simple_quality.py
```

---

## Phase 2: Reorganize & Consolidate

### 📁 New Clean Structure

```
Citation-Aware-Medical-RAG/
├── README.md                           # Main documentation (already updated)
├── requirements.txt                    # Dependencies
├── .env.example                        # Environment template
├── run_ingestion.py                    # Single ingestion script (rename current citation-focused)
│
├── src/
│   ├── app/                           # Streamlit RAG application
│   │   ├── main.py                    # Main Streamlit app
│   │   ├── rag_service.py             # RAG orchestration
│   │   ├── enhanced_retriever.py      # Citation-aware retriever
│   │   ├── config.py                  # Configuration
│   │   ├── content_utils.py           # Content utilities
│   │   ├── sheets_service.py          # Optional logging
│   │   └── prompts/                   # Prompt templates
│   │       ├── __init__.py
│   │       ├── base_prompts.py
│   │       └── medical_prompts.py
│   │
│   └── ingestion/                     # Single, clean ingestion system
│       ├── __init__.py
│       ├── pipeline.py                # Main ingestion pipeline
│       ├── citation_extractor.py      # LLM citation extraction (THE CORE)
│       ├── document_processor.py      # PDF processing
│       ├── chunk_processor.py         # Smart chunking
│       ├── metadata_enhancer.py       # Metadata enrichment
│       └── vector_uploader.py         # Pinecone upload
│
├── tests/                             # Organized test suite
│   ├── __init__.py
│   ├── test_citation_extraction.py    # Citation system tests
│   ├── test_ingestion_pipeline.py     # Pipeline tests
│   └── test_rag_service.py           # RAG service tests
│
├── data/                              # Input PDFs
├── enhanced_output/                   # Enhanced chunks output
├── examples/                          # Usage examples
│   ├── basic_usage.py
│   └── citation_analysis.py
│
└── docs/                              # Additional documentation
    ├── citation_system.md             # Deep dive on citation extraction
    ├── api_reference.md               # API documentation
    └── deployment.md                  # Deployment guide
```

### 🔄 Files to MOVE/RENAME

#### Consolidate Ingestion Systems
```bash
# Keep the best components from each system
mkdir src/ingestion/

# Move core citation extractor (the star)
mv src/ingestion_package/citation_extractor.py src/ingestion/

# Consolidate other components
mv src/ingestion_core/simple_pipeline.py src/ingestion/pipeline.py
mv src/ingestion_package/document_loader.py src/ingestion/document_processor.py
mv src/ingestion_package/chunking_strategy.py src/ingestion/chunk_processor.py
mv src/ingestion_package/metadata_extractor.py src/ingestion/metadata_enhancer.py
mv src/ingestion_package/vector_uploader.py src/ingestion/vector_uploader.py

# Remove old directories
rm -rf src/ingestion_core/
rm -rf src/ingestion_package/
```

#### Rename & Consolidate Scripts
```bash
# Single ingestion script
mv run_citation_focused_ingestion.py run_ingestion.py

# Single verification script
mv verify_simple_quality.py verify_output_quality.py
```

#### Organize Tests
```bash
# Keep relevant tests, rename for clarity
mv tests/test_citation_integration.py tests/test_citation_extraction.py
mv tests/test_pipeline.py tests/test_ingestion_pipeline.py
mv tests/test_content_utils.py tests/test_rag_service.py

# Remove comparison tests
rm tests/test_comparison_pipeline.py
rm tests/test_rag_comparison.py
rm tests/test_week1_improvements.py
```

---

## Phase 3: Create Clean Examples & Documentation

### 📝 New Documentation Structure

#### Main Documentation (Keep/Update)
- `README.md` ✅ (already updated to focus on citations)
- `requirements.txt` ✅ (keep existing)

#### Create New Focused Docs
```bash
mkdir docs/
mkdir examples/
```

**docs/citation_system.md** - Deep dive into LLM citation extraction
**docs/api_reference.md** - Clean API documentation
**docs/deployment.md** - Modern deployment guide
**examples/basic_usage.py** - Simple usage example
**examples/citation_analysis.py** - Citation quality analysis

### 🎯 Focus Areas for New Structure

1. **Citation Extraction** - The star component gets top billing
2. **Clean Ingestion** - Single pipeline focused on citation quality
3. **Simple RAG** - No over-engineering, citation-aware retrieval
4. **Clear Examples** - Show citation extraction value
5. **Organized Tests** - Test the core value proposition

---

## Implementation Commands

### Step 1: Backup Current State
```bash
git add . && git commit -m "Backup before cleanup"
git branch backup-before-cleanup
```

### Step 2: Execute Deletions
```bash
# Remove legacy systems
rm -rf src/Ingestion/ src/ingestion_core/ src/ingestion_package/
rm -rf comparisons/
rm run_ingestion.py run_simple_ingestion.py run_comparison_pipeline.py run_test_pipeline.py
rm streamlit_app.py verify_quality.py
rm CITATION_FOCUSED_README.md DEPLOYMENT.md PROMPT_ARCHITECTURE.md
rm src/app/enhanced_ingestion.py src/app/enhanced_uploader.py src/app/retrievers.py src/app/references.json
```

### Step 3: Create New Structure
```bash
# Create new directories
mkdir src/ingestion/ docs/ examples/

# Move citation extractor (the core component)
cp src/ingestion_package/citation_extractor.py src/ingestion/

# Rename main ingestion script
mv run_citation_focused_ingestion.py run_ingestion.py
mv verify_simple_quality.py verify_output_quality.py
```

### Step 4: Update Imports & References
- Update all import statements to reflect new structure
- Update documentation references
- Test that everything still works

### Step 5: Create New Examples & Docs
- Write focused examples showing citation extraction
- Create clean API documentation
- Write deployment guide

---

## Expected Benefits

After cleanup:
- ✅ **50% fewer files** - Remove duplication and legacy code
- ✅ **Clear value proposition** - Citation extraction front and center
- ✅ **Easier maintenance** - Single ingestion system, clear structure
- ✅ **Better onboarding** - Focused documentation and examples
- ✅ **Cleaner codebase** - Remove over-engineering and abstractions

## Validation Checklist

- [ ] System still processes PDFs correctly
- [ ] Citation extraction works with LLM + regex
- [ ] Streamlit app runs without errors
- [ ] Enhanced chunks have citation metadata
- [ ] Vector upload to Pinecone works
- [ ] Quality verification passes
- [ ] Documentation is accurate and complete
- [ ] Examples run successfully

---

**Bottom Line**: Transform this from a complex, over-engineered system into a clean, focused showcase of LLM-powered citation extraction for medical RAG.