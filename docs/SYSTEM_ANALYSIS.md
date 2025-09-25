# Thera‑RAG System Analysis and Improvement Plan

This document inventories technologies and parameters in use, maps the current architecture and workflows, and lists concrete issues with recommended fixes and a prioritized plan. It reflects the two‑app split (Ingestion vs Inference) and the HITL focus.

## Repository Overview

- Apps
  - Ingestion (local): `apps/ingestion/`
    - Pipeline and processors: `apps/ingestion/src/*.py`
    - CLI: `apps/ingestion/main.py` (stage-aware)
    - Tests: `apps/ingestion/tests/*.py`
  - Inference (Streamlit): `apps/inference/`
    - RAG service, retriever, prompts, UI utils: `apps/inference/src/*.py`
    - Tests: `apps/inference/tests/*.py`
- Shared: `shared/` (config, models, tests)
- Docs: `docs/` (architecture, interactive usage, plans)
- Data and outputs: `data/`, `enhanced_output/`

## Technologies & Dependencies

Core (by app; versions from requirements files):
- Common
  - Python 3.9+
  - `python-dotenv==1.1.0`
  - `requests==2.32.3`, `pydantic==2.11.4`
- Ingestion (`apps/ingestion/requirements.txt`)
  - PDF processing: `docling>=2.0.0`
  - LangChain: `langchain>=0.1.0`, `langchain-core>=0.1.0`, `langchain-community>=0.0.30`
  - LLM: `google-generativeai==0.8.5`
  - Vectors: `pinecone>=6.0.0`, `sentence-transformers>=2.0.0`
  - Parsing: `beautifulsoup4==4.13.4`, `lxml==5.4.0`
  - Data: `pandas==2.2.3`, `numpy==2.2.6`
  - Citation extras: `rapidfuzz>=3.0.0`, `jsonschema>=4.0.0`
  - CLI UX: `tqdm>=4.66.0`
- Inference (`apps/inference/requirements.txt`)
  - UI: `streamlit==1.45.1`
  - LangChain: `langchain>=0.1.0`, `langchain-core>=0.1.0`, `langchain-community>=0.0.30`, `langchain-google-genai>=1.0.0`, `langchain-pinecone>=0.1.0`
  - LLM: `google-generativeai==0.8.5`, `google-ai-generativelanguage==0.6.15`
  - Vectors: `pinecone>=6.0.0`, `sentence-transformers>=2.0.0`
  - Data: `pandas==2.2.3`, `numpy==2.2.6`
  - Google Sheets (optional): `google-api-python-client==2.169.0`, `google-auth==2.40.1`, `google-auth-httplib2==0.2.0`

## Configuration & Parameters (exhaustive)

Environment variables (primary):
- `GOOGLE_API_KEY` — Google Generative AI key (LLM)
- `PINECONE_API_KEY` — Pinecone key
- `PINECONE_INDEX_NAME` — Overrides index name
- `PINECONE_ENVIRONMENT` — Legacy env (new Pinecone client is regionless)
- `GOOGLE_SHEETS_SPREADSHEET_ID`, `GOOGLE_SHEETS_CREDS_JSON` — Optional Sheets logging
- Inference‑specific (via `.env` or Streamlit secrets): `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `RETRIEVAL_K`, `EMBEDDING_MODEL`, `PINECONE_NAMESPACE`, `SHEETS_NAME`

Shared config (`shared/config/settings.py`)
- `pinecone_index_name` default: `medical-papers` (line 23)
- `pinecone_environment` default: `us-east-1-aws` (line 24)
- `google_api_key` from env (line 27)
- `embedding_model`: `intfloat/e5-base` (line 30)
- `data_dir`: `data` (line 36)
- `output_dir`: `enhanced_output` (line 37)
- `chunk_size`: 500 (line 38)
- `chunk_overlap`: 50 (line 39)
- `max_chunks_per_doc`: 1000 (line 40)
- `metadata_version`: `v2.1_simple` (line 43)

Ingestion
- Document loader (Docling) (`apps/ingestion/src/document_processor.py`)
  - `num_threads`: 8
  - `do_ocr`: True
  - `do_table_structure`: True
  - `accelerator_device`: `auto` (supports `cpu`/`gpu` if available)
  - `text_only`: False (disables OCR/tables and lowers threads to 1 when True)
  - Methods: `extract_text()`, `extract_structured_content()` (markdown export)
- Chunking (`apps/ingestion/src/chunk_processor.py`)
  - `target_chunk_size`: 500
  - `overlap_size`: 50 (capped at ≤20% of chunk size)
  - `max_chunk_size`: 750
  - `min_chunk_size`: 250
  - `preserve_sentences`: True, `preserve_paragraphs`: True, `section_aware`: True
  - Recursive splitter separators: `\n\n`, `\n`, `. `, `! `, `? `, ` `, `` (keep separators)
  - Metadata per chunk: `chunk_id`, `chunk_size`, `word_count`, `sentence_count`, `starts_with`, `avg_word_length`, `content_hash`, `section_title`, flags (`is_small_chunk`, `is_sub_chunk`)
- Metadata enhancer (`apps/ingestion/src/metadata_enhancer.py`)
  - DocumentReference fields: `source_title`, `authors`, `publication_year`, `journal`, `doi`, `document_type`, etc.
  - Chunk metadata adds: `chunk_id`, `content_hash`, `chunk_index`, `source_*`, `page_start/end`, `section_hierarchy`, `hierarchy_breadcrumb`, `text` (FULL), `preview`, counts, `citation_text`, `reference_url`, `processed_at`, `processing_version`
- Citation extractor (`apps/ingestion/src/citation_extractor.py`)
  - LLM: `ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.1)`
  - `ExtractorConfig` (defaults):
    - `use_llm_for_refs=True`, `sentence_aware=True`
    - `context_before_chars=100`, `context_after_chars=100`
    - Matching weights: `title_weight=0.3`, `author_weight=0.3`, `year_weight=0.2`, `doi_weight=0.2`
    - `match_threshold=0.6`
  - Heuristics: reference density detection; numbered styles; DOI/URL normalization; RapidFuzz title similarity if available; JSON schema guardrails
- Vector uploader (`apps/ingestion/src/vector_uploader.py`)
  - Embedding model: `intfloat/e5-base`, with `passage:`/`query:` formatting and normalization
  - Batch size: 50
  - Metadata: stores FULL `text` content and many computed fields; namespace configurable

Inference
- Config (`apps/inference/src/config.py`)
  - Defaults: `pinecone_index_name='medical-rag-index'`, `pinecone_namespace='thera-rag'`, `embedding_model='models/embedding-001'`, `llm_model='gemini-2.5-pro'`, `llm_temperature=0.1`, `llm_max_tokens=8192`, `retrieval_k=15`, `sheets_name='Chat_Logs'`
- RAG service (`apps/inference/src/rag_service.py`)
  - Embedding: `SentenceTransformer('intfloat/e5-base')` (hard‑coded)
  - Retriever: `EnhancedPineconeRetriever(target_k=retrieval_k, enable_reranking=False, source_diversity_weight=0.2)`
  - LLM: `ChatGoogleGenerativeAI(model=config['llm_model'], temperature, max_tokens)`
  - Chain: `ConversationalRetrievalChain` with a custom medical prompt
- Enhanced retriever (`apps/inference/src/enhanced_retriever.py`)
  - Initial K = target_k × 1.0 by default; optional CrossEncoder re‑ranking (disabled by default)
  - Source diversity balancing; metadata enhancement for display (citation, location, relevance %)
- Prompts (`apps/inference/src/prompts`) — `medical_rag`, `vanilla_medical`, `cleansing_medical`
- UI (`apps/inference/src/main.py`) — Streamlit layout, chunk/source rendering, optional Sheets logging

Outputs
- Interactive pipeline saves staged artifacts under `enhanced_output/<doc_name>/` with five stages (raw text, chunks, enhanced chunks, citations, embeddings)

## Architecture & Workflows (high level)

- Ingestion (Interactive)
  1) Docling extract → 2) Optional medical filtering → 3) Chunking → 4) Metadata enhancement → 5) Citation extraction/linking → 6) Optional embeddings/upload
- Inference (Streamlit)
  - Query → embed (e5) → Pinecone search (namespace) → optional re‑rank → LLM answer with prompt → render sources/citations → optional Sheets log

## Issues and Risks (with examples and fixes)

P0 — Functional correctness
1) Broken SimplePipeline API usage
   - Files: `apps/ingestion/src/pipeline.py`
   - Problems:
     - Instantiates `DocumentProcessor()` with no `file_path`, but class requires it and exposes `extract_*` (not `process_document`) methods.
     - Instantiates `ChunkMetadataEnhancer()` with no `DocumentReference` (constructor requires it).
     - Calls `self.doc_processor.process_document(...)` which doesn’t exist.
   - Impact: Non‑interactive ingestion path will error.
   - Fix: Align to interactive pipeline: construct `DoclingBookLoader(file_path, text_only=...)`, call `extract_structured_content()`. Create `DocumentReference` and pass into `ChunkMetadataEnhancer(doc_ref)`. Adjust imports and return types.

2) Out‑of‑date tests vs current schema
   - File: `shared/tests/test_schema_detailed.py`
   - Problem: Imports `MetadataProfile`, `PineconeMetadataManager`, and `estimate_size(profile)` which no longer exist in `shared/models/pinecone_schema.py`.
   - Impact: Test fails; misleads about supported schema API.
   - Fix: Update test to use `simple_pinecone_schema.py` or current dataclasses; remove references to deprecated classes; or archive this test.

P0 — Config coherence
3) Embedding model inconsistency
   - Inference config default `EMBEDDING_MODEL='models/embedding-001'` but code loads `SentenceTransformer('intfloat/e5-base')`.
   - Impact: Confusing/unused config; risk of mismatch between ingestion/inference embeddings.
   - Fix: Single source of truth (shared config) and pass through; enforce same model end‑to‑end. ENSURE ITS e5-base

4) Index/name inconsistencies
   - Shared default index: `medical-papers`; inference default: `medical-rag-index`; ingestion `SimplePipeline` default: `medical-rag-index`.
   - Impact: Upload/search may target different indexes unintentionally.
   - Fix: Centralize in `shared/config` and import from there across apps. ENSURE ITS medical-rag-index

P1 — Scalability/cost/performance


1) Sys.path hacks and import fragility
   - Multiple scripts insert paths at runtime (`sys.path.insert`), e.g., `apps/ingestion/main.py`, `apps/inference/main.py`.
   - Impact: Fragile execution across environments.
   - Fix: Turn `apps/ingestion/src` and `apps/inference/src` into packages; use relative imports; optionally add a `pyproject.toml` for editable installs in dev.

2) Hard‑coded absolute paths in tests
   - File: `apps/ingestion/tests/test_ingestion_pipeline.py` uses Windows absolute paths.
   - Fix: Parameterize input CSV via env/args and keep test data under repo (or mark as example script, not a test).

P1 — Data quality & UX
9) Citation matching identifiers vs scoring
   - `_extract_citation_identifiers` rarely returns `authors`, `title`, or `doi`; matching weights include these fields, reducing effectiveness.
   - Fix: Extract richer identifiers (e.g., first author list, possible titles near inline mentions; numeric citations mapped to reference index). Adjust scoring accordingly.

10) UI citation extraction vs ingestion metadata
   - `content_utils.extract_citations` only handles author‑year; doesn’t consume ingestion’s `metadata['citations']` nor numeric styles.
   - Fix: Prefer citations from chunk metadata when available; extend regex to numeric forms.

11) Documentation drift
   - `README.md` “Architecture Deep Dive” paths still reference `src/ingestion/...` and `src/app/...` (old locations).
   - Fix: Update links to `apps/ingestion/src/*` and `apps/inference/src/*`.

P2 — Reliability & polish
12) Error handling for LLM calls
   - `rag_service.py` rate‑limit parsing expects “retry in <s>”; Google errors often don’t include that token. Broaden handling and backoff hints.

13) Streamlit resource loading
   - `SentenceTransformer` loads per service init; ensure caching or singleton to avoid repeated load in hot‑reload scenarios.

14) Security hygiene
   - Secrets handled via env/Streamlit secrets; ensure no accidental logging of secret values and consider `.env.example` with all keys.

## Quick Cross‑Check (Key Parameters)

- Embeddings
  - Ingestion: `intfloat/e5-base`
  - Inference: hard‑coded `intfloat/e5-base` (config default differs)
- LLM
  - Ingestion: `gemini-1.5-flash`, `temperature=0.1`
  - Inference: default `gemini-2.5-pro`, `temperature=0.1`, `max_tokens=8192`
- Pinecone
  - Default index names vary: `medical-papers` vs `medical-rag-index`; default namespace `thera-rag`
    - only use `medical-rag-index`
- Chunking
  - Size 500, overlap 50; section‑aware; min/max 250/750
- Docling
  - `text_only` mode disables OCR and tables; threads=1

## Validation/Next Steps

- Run interactive pipeline end‑to‑end on a sample PDF (already in `data/vha-guideline.pdf`) and verify staged outputs in `enhanced_output/`.
- After P0 fixes, add a simple smoke script for the non‑interactive path.
- Confirm Pinecone metadata size per vector and adjust tier/profile defaults accordingly.
- Align embedding model across ingestion/inference; rebuild vectors if needed when changing models.

---

Appendices

### File/Class References (selected)
- Ingestion
  - Loader: `apps/ingestion/src/document_processor.py`
  - Chunking: `apps/ingestion/src/chunk_processor.py`
  - Metadata enhancer: `apps/ingestion/src/metadata_enhancer.py`
  - Citations: `apps/ingestion/src/citation_extractor.py`
  - Uploader: `apps/ingestion/src/vector_uploader.py`
- Inference
  - RAG service: `apps/inference/src/rag_service.py`
  - Retriever: `apps/inference/src/enhanced_retriever.py`
  - Prompts: `apps/inference/src/prompts/*`
  - Content utils: `apps/inference/src/content_utils.py`
- Shared
  - Config: `shared/config/settings.py`
  - Schemas: `shared/models/*.py`


