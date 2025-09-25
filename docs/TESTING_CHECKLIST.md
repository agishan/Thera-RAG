# End‑to‑End Testing Checklist (Ingestion + Inference)

Use this manual checklist to validate both apps locally. Commands are shown for PowerShell/Windows; adapt for your shell as needed.

## 0) Prerequisites
- [ ] Python 3.9+
- [ ] Internet access for model downloads and API calls (Gemini, Pinecone)
- [ ] A Pinecone index (default: `medical-rag-index`) and API key

## 1) Create virtual environment
```powershellwa
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies
```powershell
pip install -r apps/ingestion/requirements.txt
pip install -r apps/inference/requirements.txt
```

## 3) Configure environment
Create a `.env` file at the repo root with the following keys (adjust as needed):
```ini
# Required for citations (LLM)
GOOGLE_API_KEY=your-google-api-key

# Required for vector upload + retrieval
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=medical-rag-index
PINECONE_NAMESPACE=thera-rag

# Optional: Inference tuning
LLM_MODEL=gemini-2.5-pro
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=8192
RETRIEVAL_K=15
EMBEDDING_MODEL=intfloat/e5-base
```

## 4) Prepare data
- [ ] Ensure a PDF exists at `data/` (sample present: `data/vha-guideline.pdf`)
- [ ] If testing your own PDFs, copy them into `data/`

## 5) Test ingestion - CLI (HITL)
Runs stage-by-stage with approval and writes artifacts under `enhanced_output/<doc>/`.

Initialize and Stage 1:
```powershell
python apps/ingestion/main.py init --pdf data\vha-guideline.pdf --text-only
python apps/ingestion/main.py run --pdf data\vha-guideline.pdf
python apps/ingestion/main.py inspect --pdf data\vha-guideline.pdf --stage 1
python apps/ingestion/main.py approve --pdf data\vha-guideline.pdf --stage 1
```

Stage 2 and 2.5 iteration:
```powershell
python apps/ingestion/main.py run --pdf data\vha-guideline.pdf
python apps/ingestion/main.py inspect --pdf data\vha-guideline.pdf --stage 2
python apps/ingestion/main.py approve --pdf data\vha-guideline.pdf --stage 2

# Tune 2.5 if needed
python apps/ingestion/main.py stage --pdf data\vha-guideline.pdf --stage 2.5 --density-threshold 0.7 --keep references appendix
python apps/ingestion/main.py inspect --pdf data\vha-guideline.pdf --stage 2.5 --json
python apps/ingestion/main.py approve --pdf data\vha-guideline.pdf --stage 2.5 --keep references
```

Stage 3 and 4:
```powershell
python apps/ingestion/main.py run --pdf data\vha-guideline.pdf --auto-continue
python apps/ingestion/main.py inspect --pdf data\vha-guideline.pdf
```

Confirm the following folders appear (examples):
- [ ] `enhanced_output\vha-guideline\1_raw_text\`
- [ ] `enhanced_output\vha-guideline\2_chunks\`
- [ ] `enhanced_output\vha-guideline\2.5_classify\`
- [ ] `enhanced_output\vha-guideline\3_enhanced_chunks\`
- [ ] `enhanced_output\vha-guideline\4_citations\`

### Alternative: HITL Ingestion Webapp (optional, later)
If you prefer a UI later, you can adapt the CLI controller to Streamlit (`apps/ingestion/webapp/`). For now, the CLI is the source of truth.

### Alternative: Run a stage window
Re-run a stage window with approvals enforced and stop after each stage:
```powershell
python apps/ingestion/main.py run --pdf data\vha-guideline.pdf --from-stage 2 --until-stage 3 --force
```## 7) Verify ingestion quality (optional)\n```powershell\n# Inspect stage-level stats\npython apps/ingestion/main.py inspect --pdf data\\vha-guideline.pdf --stage 2.5 --json\npython apps/ingestion/main.py inspect --pdf data\\vha-guideline.pdf --stage 4 --json\n```\n\n## 8) Test inference — Streamlit app
Option A (recommended):
```powershell
cd apps/inference
streamlit run main.py
```
Option B (entry module wrapper):
```powershell
cd ../../
python apps/inference/main.py
```
Open the app (usually http://localhost:8501). Ask a question, e.g.:
- “How should ROTEM/TEG guide transfusion in major bleeding?”

Validate:
- [ ] An answer is returned without errors
- [ ] “Citations from Sources” (or “References”) show titles matched from chunks
- [ ] Retrieved Chunks list includes citation/location metadata and links (DOI when present)

## 9) Retrieval sanity checks
- [ ] Ensure `PINECONE_INDEX_NAME` matches the index used during ingestion
- [ ] If no results: confirm your index has vectors in the `thera-rag` namespace (default), or set `PINECONE_NAMESPACE` consistently in `.env`

## 10) Troubleshooting
- Docling extraction errors
  - Try `--text-only` for faster, lighter extraction
  - Ensure the file opens normally in a PDF viewer (corruption can cause failures)
- Gemini API errors (429/rate limits)
  - Wait 60s and retry; keep temperature low (0.1) for parsing stability
- Pinecone not returning results
  - Verify the index name/namespace and that vectors exist; check Pinecone console
  - Confirm ingestion completed with upload step
- Missing citations
  - Ensure `GOOGLE_API_KEY` is set (LLM parsing); regex fallback will extract fewer/shallower citations

## 11) Optional validation steps
- [ ] Test with a second PDF and compare coverage/quality
- [ ] Toggle medical filtering off and compare retrieval noise
- [ ] Increase `RETRIEVAL_K` to 20–30 for complex queries, then compare response quality and latency

## 12) Cleanup / Re‑run
- [ ] Delete old `enhanced_output/<doc_name>/` before a fresh run if you want a clean slate
- [ ] Re‑run scripted or interactive ingestion
- [ ] Re‑open the Streamlit app to test end‑to‑end

---

For detailed diagrams and stage docs, see:
- `docs/INGESTION_PIPELINE.md` — Ingestion flow
- `docs/CitationExtraction.md` — Citation internals


