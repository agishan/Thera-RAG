# TODO — Ingestion First, Quality RAG

Priority: Fix ingestion pipeline for a single guideline (vha-guideline.pdf), with precise, inspectable stages and minimal, predictable LLM usage.

## P0 — Implement Robust Stage CLI (1 → 4)
- [ ] Minimal CLI at `apps/ingestion/main.py` with subcommands:
  - `init` — create workspace + manifest for `--pdf` (default: `data/vha-guideline.pdf`)
  - `status` — show manifest summary (approved stages, params, counts)
  - `stage` — run a single stage: `--stage {1|2|2.5|3|4}` (5/6 later)
  - `inspect` — print stage summaries (counts, samples) and `--json` dump
  - `approve` — mark stage approved (gates downstream)
  - `run` — run sequentially with `--from`/`--until`
- [ ] Manifest `enhanced_output/<doc>/manifest.json`:
  - Track input_pdf, options, per‑stage params/stats/timestamps/approved, artifact lists
- [x] Stage folders (atomic writes): `1_raw_text/`, `2_chunks/`, `2.5_classify/`, `3_enhanced_chunks/`, `4_citations/`

## P0 — Pipeline Hooks & Params
- [ ] `SimplePipeline.process_document`: accept `text_only` and pass to Docling loader
- [ ] Stage 2.5 (already added: heuristics) — store `chunk_category`, `reference_likeness` in metadata; write `classification_summary.json`
- [ ] Stage 3 — keep existing rule‑based enrichment; always preserve full text in metadata['text']
- [ ] Stage 4 — unchanged core; ensure final chunks carry `citations[]` and `citation_count`

## P0 — LLM Utilities (Lean, Strict, Cached)
- [ ] `apps/ingestion/src/llm_utils.py` with:
  - Small Gemini 1.5 Flash client (temp=0.1), JSON schema validation, retry on parse fail
  - Content hashing cache (per stage, per prompt_version)
- [ ] Stage 2.5 (LLM‑assisted) — optional flags:
  - `--llm-simple on|off` (default on)
  - `--max-llm-chunks 100` (cap how many chunks go to LLM)
  - `--max-llm-chars 1000` (truncate chunk text for prompt)
  - Minimal outputs per chunk: `section_type`, `confidence`, `inline_citations[]` with spans + style + id_hint
- [ ] Stage 3 (LLM metadata) — optional flags:
  - `--llm-metadata on|off` (default off)
  - `--summ-len 150`, `--max-llm-chunks 100`
  - Outputs per chunk: `semantic_title`, `summary`, `key_terms[]`, `content_type`, `confidence`

## P0 - Docs & Cleanup
- [x] Remove notebooks directory (as agreed)
- [x] Update README and TESTING_CHECKLIST to point to the new CLI
- [ ] Update `docs/INGESTION_PIPELINE.md` to show Stage 2.5 LLM outputs + Stage 3 optional LLM summary/terms
- [ ] Update or archive ingestion tests that reference `interactive_pipeline` to use stage-aware modules or mark as legacy
- [ ] Add cross-link from `docs/SYSTEM_ANALYSIS.md` to `docs/INGESTION_CLI.md` ("How to run ingestion")

## P1 — Human‑Controlled LLM Gating (Requested)
- [ ] Add `inspect --stage 2.5` to print:
  - `classification_counts`, sample filtered chunks (show `chunk_id`, first 200 chars)
  - flags in use (`max-llm-chunks`, `max-llm-chars`, `keep`)
- [ ] Add `approve --stage 2.5` flow with an option to `--keep <label>` at approval time (writes to manifest)
- [ ] Provide `export --stage 2.5` to write a review file (CSV/JSON) so a human can bulk edit `keep` overrides and re‑run 2.5

## P1 — Bridge to Streamlit UI
- [ ] Ensure CLI stages map 1:1 to thin controller functions so a Streamlit UI can call `stage/inspect/approve` over the same manifest without duplication.
- [ ] Add a simple REST‑ish controller (optional) or keep direct module calls; guarantee artifact/manifest schema stability for UI reuse.

## P1 — Stage 4 Hints (Non‑breaking)
- [ ] Use `metadata['inline_citations']` from 2.5 as hints in Stage 4 matching (keep existing robust logic)

## P2 — Later (Optional)
- [ ] Extend CLI to stages 5/6 (embeddings/upload) with verify subcommand
- [ ] Add unit tests for llm_utils JSON parsing + caching
- [ ] Add a compact retrieval smoke test command

---

Notes
- Cost/latency control: `--max-llm-chunks` and `--max-llm-chars` bound spend; caching by `content_hash` avoids repeats.
- Guardrails: strict JSON schema + retry, then fallback to heuristics only (no hard failures).
- Full text fidelity: Never truncate stored content; only truncate LLM input.
- Single‑doc focus: Default `--pdf data/vha-guideline.pdf` keeps CLI friction low.

