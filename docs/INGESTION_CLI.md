# Ingestion CLI

This CLI drives the ingestion pipeline stage-by-stage with human approval. It lets you pause after any stage, inspect the artifacts, tweak parameters (especially Stage 2.5), and continue when satisfied.

## Quick Start

Environment prerequisites:
- Python 3.9+
- Install dependencies: `pip install -r apps/ingestion/requirements.txt`
- Set env vars as needed: `GOOGLE_API_KEY` (citations), `PINECONE_API_KEY` (optional, upload later)

Initialize a manifest for a PDF:
- `python apps/ingestion/main.py init --pdf data/vha-guideline.pdf --text-only`

Run stages with approval gates:
1) Stage 1
- `python apps/ingestion/main.py run --pdf data/vha-guideline.pdf`
- Inspect: `python apps/ingestion/main.py inspect --pdf data/vha-guideline.pdf --stage 1`
- Approve: `python apps/ingestion/main.py approve --pdf data/vha-guideline.pdf --stage 1`

2) Stage 2 (chunking)
- `python apps/ingestion/main.py run --pdf data/vha-guideline.pdf`
- **Shows chunking configuration** (target_chunk_size, overlap_size, etc.)
- Inspect: `python apps/ingestion/main.py inspect --pdf data/vha-guideline.pdf --stage 2`
- Reject if unhappy: `python apps/ingestion/main.py reject --pdf data/vha-guideline.pdf --stage 2`
- Approve: `python apps/ingestion/main.py approve --pdf data/vha-guideline.pdf --stage 2`

2.5) Stage 2.5 (classification and filtering)
- Run with params: `python apps/ingestion/main.py stage --pdf data/vha-guideline.pdf --stage 2.5 --density-threshold 0.7 --keep references appendix`
- Inspect: `python apps/ingestion/main.py inspect --pdf data/vha-guideline.pdf --stage 2.5 --json`
- Approve (record overrides): `python apps/ingestion/main.py approve --pdf data/vha-guideline.pdf --stage 2.5 --keep references`

3) Stage 3
- `python apps/ingestion/main.py run --pdf data/vha-guideline.pdf`

4) Stage 4 (citations)
- `python apps/ingestion/main.py run --pdf data/vha-guideline.pdf`
- Inspect consolidated: `python apps/ingestion/main.py inspect --pdf data/vha-guideline.pdf`

## Commands

`init`
- Creates `enhanced_output/<doc>/manifest.json` and records `input_pdf`, `output_dir`, and options.
- Flags:
  - `--pdf`: input PDF path (default `data/vha-guideline.pdf`)
  - `--text-only`: use lightweight Docling text extraction

`status`
- Shows per-stage `completed` and `approved` state.
- Flags: `--pdf` or `--doc` to locate the manifest

`stage`
- Runs exactly one stage (no gating beyond upstream completion).
- Flags:
  - `--stage {1|2|2.5|3|4}`
  - `--text-only` (Stage 1)
  - `--density-threshold <float>` and `--keep <labels...>` (Stage 2.5)

`approve`
- Marks a stage as approved and optionally writes param overrides (e.g., `--keep` labels for 2.5) to the manifest.

`reject`
- Removes a stage's outputs and manifest entry, plus all dependent stages.
- Useful for parameter changes: reject → edit config → re-run.
- Flags: `--stage {1|2|2.5|3|4}`, `--pdf` or `--doc`

`inspect`
- Without `--stage`: prints totals from the consolidated enhanced file `enhanced_<doc>.json`.
- With `--stage`: prints stage-specific summary (counts, key stats; 2.5 shows classification counts and sample filtered IDs).
- Flags: `--json` to return machine-friendly payloads

`run`
- Executes stages in order and stops at the first unapproved stage to allow review and approval.
- Flags:
  - `--auto-continue`: keep going until gated by a missing approval
  - `--force`: re-run even if a stage is already approved
  - `--from-stage` / `--until-stage`: run a window (e.g., `--from-stage 2 --until-stage 3`)

## Gating Rules
- Stage N requires Stage N-1 to be approved before it runs during `run`.
- `stage` ignores approvals and runs the chosen stage (relies on upstream completion only).
- All per-stage stats, params, artifacts, and approvals are recorded in `manifest.json` for reproducibility.

## Output Locations
- `enhanced_output/<doc>/1_raw_text/*`
- `enhanced_output/<doc>/2_chunks/*`
- `enhanced_output/<doc>/2.5_classify/*`
- `enhanced_output/<doc>/3_enhanced_chunks/*`
- `enhanced_output/<doc>/4_citations/*`
- Consolidated final file: `enhanced_output/<doc>/enhanced_<doc>.json`

## Tips
- Set `GOOGLE_API_KEY` to enable LLM-assisted citation discovery and parsing in Stage 4.
- For Stage 2.5, iterate until satisfied: run → inspect → adjust `--keep` labels → approve.
- **For chunking parameter changes**: `reject --stage 2` → edit `apps/ingestion/src/chunk_processor.py` → `stage --stage 2`
- Use `--from-stage`/`--until-stage` + `--force` to re-run a window efficiently after parameter changes.

## Chunking Parameter Workflow
1. Run Stage 2: `python apps/ingestion/main.py stage --stage 2` (shows current config)
2. If unhappy with chunks: `python apps/ingestion/main.py reject --stage 2`
3. Edit chunking config in `apps/ingestion/src/chunk_processor.py` lines 16-22
4. Re-run: `python apps/ingestion/main.py stage --stage 2` (shows new config)
5. Continue pipeline: `python apps/ingestion/main.py run`



## Appendix: Manifest Schema (short)

File: `enhanced_output/<doc>/manifest.json`

Top-level fields:
- `doc_name`: string
- `input_pdf`: absolute file path
- `output_dir`: absolute output directory
- `options`: object
  - `text_only`: bool
- `stages`: object keyed by stage ("1", "2", "2.5", "3", "4")

Per-stage entry (example):
```json
{
  "1": {
    "completed": true,
    "approved": true,
    "runs": ["2025-05-18T12:34:56"],
    "approvals": ["2025-05-18T12:40:02"],
    "params": { "text_only": true },
    "stats": { "characters": 123456, "words": 23000 },
    "artifacts": [
      "enhanced_output/vha-guideline/1_raw_text/full_text.txt",
      "enhanced_output/vha-guideline/1_raw_text/structured_content.json"
    ]
  }
}
```

Notes:
- `params` is merged per approval/run to preserve the latest overrides (e.g., Stage 2.5 `keep`).
- `stats` is stage-specific (e.g., Stage 2 `chunking_stats`, Stage 4 `citation_stats`).
- Only `run` enforces approval gating; `stage` just runs the given stage.
