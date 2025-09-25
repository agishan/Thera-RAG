#!/usr/bin/env python3
"""
Thera-RAG Ingestion CLI (stage-aware with approvals)

Commands:
  init      - Initialize workspace + manifest for a PDF (default: data/vha-guideline.pdf)
  status    - Show manifest summary (per-stage completion/approval)
  stage     - Run a single stage: --stage {1|2|2.5|3|4}
  approve   - Mark a stage approved (optionally with param overrides)
  reject    - Reject a stage (removes outputs and manifest entry)
  inspect   - Inspect outputs, optionally per stage: --stage {1|2|2.5|3|4}
  run       - Run sequentially and stop at first unapproved stage (gated)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.pipeline import SimplePipeline
from src import stages as stg


def doc_dir_for(pdf_path: Path, out_base: Path = Path("enhanced_output")) -> Path:
    name = pdf_path.stem
    d = out_base / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(doc_name: str, out_base: Path = Path("enhanced_output")) -> Path:
    return out_base / doc_name / "manifest.json"


def load_manifest(doc_name: str) -> Dict[str, Any] | None:
    p = manifest_path(doc_name)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_manifest(doc_name: str, manifest: Dict[str, Any]) -> None:
    p = manifest_path(doc_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)


# Stage order and helpers for stage-aware flow
STAGE_ORDER: List[str] = ["1", "2", "2.5", "3", "4", "5"]


def _ensure_manifest(pdf: Path, out_base: Path, text_only: bool = False) -> Dict[str, Any]:
    doc_name = pdf.stem
    m = load_manifest(doc_name)
    if not m:
        m = {
            "doc_name": doc_name,
            "input_pdf": str(pdf),
            "output_dir": str(doc_dir_for(pdf, out_base)),
            "options": {"text_only": bool(text_only)},
            "stages": {},
        }
    return m


def _update_stage_manifest(
    m: Dict[str, Any], stage: str,
    stats: Dict[str, Any] | None = None,
    params: Dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    approved: bool | None = None,
) -> None:
    s = m.setdefault("stages", {}).setdefault(stage, {})
    s["completed"] = True
    s.setdefault("runs", []).append(datetime.now().isoformat())
    if stats is not None:
        s["stats"] = stats
    if params is not None:
        # merge/overwrite
        cur = s.get("params", {}) or {}
        cur.update(params)
        s["params"] = cur
    if artifacts is not None:
        s["artifacts"] = artifacts
    if approved is not None:
        s["approved"] = bool(approved)
        if approved:
            s.setdefault("approvals", []).append(datetime.now().isoformat())


def _check_dependencies(m: Dict[str, Any], stage: str) -> Tuple[bool, List[str]]:
    idx = STAGE_ORDER.index(stage)
    missing = []
    for prev in STAGE_ORDER[:idx]:
        s = (m.get("stages", {}) or {}).get(prev)
        if not s or not s.get("completed"):
            missing.append(prev)
    return (len(missing) == 0), missing


def cmd_stage(args: argparse.Namespace) -> None:
    pdf = Path(args.pdf or "data/vha-guideline.pdf").resolve()
    out_base = Path(args.output or "enhanced_output").resolve()
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    stage = str(args.stage)
    if stage not in STAGE_ORDER:
        raise SystemExit(f"Invalid stage '{stage}'. Choose one of {STAGE_ORDER}")

    m = _ensure_manifest(pdf, out_base, text_only=bool(getattr(args, "text_only", False)))
    out_dir = Path(m["output_dir"])

    ok, missing = _check_dependencies(m, stage)
    if not ok:
        raise SystemExit(f"Dependencies not met. Run stages: {', '.join(missing)} first.")

    # Dispatch
    if stage == "1":
        res = stg.stage1_extract(pdf, out_dir, text_only=bool(args.text_only))
        params = {"text_only": bool(args.text_only)}
    elif stage == "2":
        # Print chunking configuration before running
        from .chunk_processor import ChunkingConfig
        config = ChunkingConfig()
        if config.chunking_method == "header_based":
            print(f"Header-Based Chunking Configuration:")
            print(f"   chunking_method: {config.chunking_method}")
            print(f"   target_tokens: {config.target_tokens}")
            print(f"   min_tokens: {config.min_tokens}")
            print(f"   max_tokens: {config.max_tokens}")
            print(f"   overlap_tokens: {config.overlap_tokens}")
            print(f"   merge_small_sections: {config.merge_small_sections}")
            print(f"   preserve_references: {config.preserve_references}")
            print(f"   preserve_abstracts: {config.preserve_abstracts}")
        else:
            print(f"Docling HierarchicalChunker Configuration:")
            print(f"   chunking_method: {config.chunking_method}")
            print(f"   target_tokens: {config.target_tokens}")
        print()

        res = stg.stage2_chunk(pdf, out_dir)
        params = {
            "chunking_method": config.chunking_method,
            "target_tokens": config.target_tokens,
            "min_tokens": config.min_tokens,
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "tokenizer_model": config.tokenizer_model
        }
    elif stage == "2.5":
        keep = args.keep or []
        res = stg.stage2_5_classify(out_dir, density_threshold=float(args.density_threshold), keep_labels=keep)
        params = {"density_threshold": float(args.density_threshold), "keep": keep}
    elif stage == "3":
        res = stg.stage3_enhance(pdf, out_dir)
        params = {}
    elif stage == "4":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        res = stg.stage4_citations(out_dir, google_api_key)
        params = {"use_llm": bool(google_api_key)}
    elif stage == "5":
        res = stg.stage5_citation_optimization(out_dir)
        params = {}
    else:
        raise SystemExit(f"Unknown stage: {stage}")

    _update_stage_manifest(m, stage, stats=res.get("stats"), params=params, artifacts=res.get("artifacts"))
    save_manifest(pdf.stem, m)
    print(json.dumps({"stage": stage, "completed": True, "stats": res.get("stats", {}), "params": params}, indent=2))


def cmd_approve(args: argparse.Namespace) -> None:
    doc_name = args.doc or Path(args.pdf).stem if args.pdf else None
    if not doc_name:
        raise SystemExit("Provide --doc or --pdf to locate the manifest.")
    m = load_manifest(doc_name)
    if not m:
        raise SystemExit(f"Manifest not found for '{doc_name}'. Run 'init' or 'stage --stage 1' first.")

    stage = str(args.stage)
    if stage not in STAGE_ORDER:
        raise SystemExit(f"Invalid stage '{stage}'. Choose one of {STAGE_ORDER}")
    if stage not in (m.get("stages", {}) or {}):
        raise SystemExit(f"Stage '{stage}' has not run yet.")

    params_override: Dict[str, Any] = {}
    if stage == "2.5" and args.keep:
        params_override["keep"] = list(args.keep)

    _update_stage_manifest(m, stage, params=params_override if params_override else None, approved=True)
    save_manifest(doc_name, m)
    print(f"Approved stage {stage} for '{doc_name}'.")


def cmd_init(args: argparse.Namespace) -> None:
    pdf = Path(args.pdf).resolve()
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    out_dir = doc_dir_for(pdf)
    manifest = {
        "doc_name": pdf.stem,
        "input_pdf": str(pdf),
        "output_dir": str(out_dir),
        "options": {
            "text_only": bool(args.text_only),
        },
        "stages": {},
    }
    save_manifest(pdf.stem, manifest)
    print(f"Initialized manifest at {manifest_path(pdf.stem)}")


def cmd_status(args: argparse.Namespace) -> None:
    doc_name = args.doc or Path(args.pdf).stem if args.pdf else None
    if not doc_name:
        raise SystemExit("Provide --doc or --pdf to locate the manifest.")
    m = load_manifest(doc_name)
    if not m:
        raise SystemExit(f"Manifest not found for '{doc_name}'. Run 'init' first.")
    stages = m.get("stages", {})
    print(json.dumps({
        "doc_name": m.get("doc_name"),
        "input_pdf": m.get("input_pdf"),
        "output_dir": m.get("output_dir"),
        "stages": {k: {"completed": v.get("completed", False), "approved": v.get("approved", False)} for k, v in stages.items()},
    }, indent=2))



def cmd_run(args: argparse.Namespace) -> None:
    # Determine PDF and output
    pdf = Path(args.pdf or "data/vha-guideline.pdf").resolve()
    out_base = Path(args.output or "enhanced_output").resolve()
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    # Ensure/Load manifest
    m = _ensure_manifest(pdf, out_base, text_only=bool(args.text_only))
    doc_name = pdf.stem
    out_dir = Path(m["output_dir"]) 

    # Stage window
    stage_from = str(getattr(args, 'from_stage', None) or STAGE_ORDER[0])
    stage_until = str(getattr(args, 'until_stage', None) or STAGE_ORDER[-1])
    if stage_from not in STAGE_ORDER or stage_until not in STAGE_ORDER:
        raise SystemExit(f"--from/--until must be in {STAGE_ORDER}")
    start_idx = STAGE_ORDER.index(stage_from)
    end_idx = STAGE_ORDER.index(stage_until)
    if start_idx > end_idx:
        raise SystemExit("--from must be <= --until")

    # Run with approval gating
    for stage in STAGE_ORDER[start_idx:end_idx+1]:
        # Check approval of previous stage
        prev_idx = STAGE_ORDER.index(stage)
        if prev_idx > 0:
            prev_stage = STAGE_ORDER[prev_idx-1]
            prev_meta = (m.get('stages', {}) or {}).get(prev_stage)
            if not prev_meta or not prev_meta.get('approved'):
                print(f"Halting: Stage {prev_stage} is not approved. Inspect and approve before proceeding.")
                break

        # Skip if approved and not forcing
        s_meta = (m.get('stages', {}) or {}).get(stage)
        if s_meta and s_meta.get('completed') and s_meta.get('approved') and not getattr(args, 'force', False):
            print(f"Skipping stage {stage} (already completed and approved).")
            continue

        # Dispatch stage
        if stage == '1':
            res = stg.stage1_extract(pdf, out_dir, text_only=bool(args.text_only))
            params = { 'text_only': bool(args.text_only) }
        elif stage == '2':
            res = stg.stage2_chunk(pdf, out_dir)
            params = {}
        elif stage == '2.5':
            # Use defaults here; tune via 'stage' command if needed
            res = stg.stage2_5_classify(out_dir)
            params = { 'density_threshold': 0.7, 'keep': [] }
        elif stage == '3':
            res = stg.stage3_enhance(pdf, out_dir)
            params = {}
        elif stage == '4':
            google_api_key = os.getenv('GOOGLE_API_KEY')
            res = stg.stage4_citations(out_dir, google_api_key)
            params = { 'use_llm': bool(google_api_key) }
        elif stage == '5':
            res = stg.stage5_citation_optimization(out_dir)
            params = {}
        else:
            raise SystemExit(f"Unknown stage: {stage}")

        _update_stage_manifest(m, stage, stats=res.get('stats'), params=params, artifacts=res.get('artifacts'))
        save_manifest(doc_name, m)
        print(f"Completed stage {stage}. Review with 'inspect --stage {stage}' and approve to continue.")

        # Stop after each stage unless auto-continue is set
        if not getattr(args, 'auto_continue', False):
            break

def cmd_reject(args: argparse.Namespace) -> None:
    doc_name = args.doc or Path(args.pdf).stem if args.pdf else None
    if not doc_name:
        raise SystemExit("Provide --doc or --pdf to locate the manifest.")

    m = load_manifest(doc_name)
    if not m:
        raise SystemExit(f"Manifest not found for '{doc_name}'. Run 'init' first.")

    stage = str(args.stage)
    if stage not in STAGE_ORDER:
        raise SystemExit(f"Invalid stage '{stage}'. Choose one of {STAGE_ORDER}")

    out_dir = Path(m["output_dir"])

    # Remove stage directory based on stage
    stage_dirs = {
        "1": out_dir / "1_raw_text",
        "2": out_dir / "2_chunks",
        "2.5": out_dir / "2.5_classify",
        "3": out_dir / "3_enhanced_chunks",
        "4": out_dir / "4_citations",
        "5": out_dir / "5_citation_index"
    }

    stage_dir = stage_dirs.get(stage)
    if stage_dir and stage_dir.exists():
        import shutil
        shutil.rmtree(stage_dir)
        print(f"Removed stage {stage} directory: {stage_dir}")

    # Remove stage from manifest
    stages = m.get("stages", {})
    if stage in stages:
        del stages[stage]
        print(f"Removed stage {stage} from manifest")

    # Also remove dependent stages
    stage_idx = STAGE_ORDER.index(stage)
    for dependent_stage in STAGE_ORDER[stage_idx + 1:]:
        dep_dir = stage_dirs.get(dependent_stage)
        if dep_dir and dep_dir.exists():
            shutil.rmtree(dep_dir)
            print(f"Removed dependent stage {dependent_stage} directory: {dep_dir}")
        if dependent_stage in stages:
            del stages[dependent_stage]
            print(f"Removed dependent stage {dependent_stage} from manifest")

    # Remove final enhanced file if stage 4 was rejected
    if stage in ["1", "2", "2.5", "3", "4"]:
        final_file = out_dir / f"enhanced_{doc_name}.json"
        if final_file.exists():
            final_file.unlink()
            print(f"Removed final enhanced file: {final_file}")

    save_manifest(doc_name, m)
    print(f"Stage {stage} and dependents rejected successfully")


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload optimized chunks to Pinecone"""
    import os
    from src.vector_uploader_optimized import OptimizedVectorUploader
    from langchain_core.documents import Document

    # Get document info
    doc_name = args.doc or Path(args.pdf).stem if args.pdf else None
    if not doc_name:
        raise SystemExit("Provide --doc or --pdf to locate the chunks.")

    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY environment variable is required.")

    # Set metadata profile if specified
    if args.profile:
        os.environ["PINECONE_METADATA_PROFILE"] = args.profile.upper()

    # Find the best chunks file (optimized from Stage 5, or fallback to Stage 4)
    out_dir = Path("enhanced_output") / doc_name
    try:
        from src.stages import get_vector_ready_chunks
        chunks_file = get_vector_ready_chunks(out_dir)
        print(f"Using chunks: {chunks_file}")
    except FileNotFoundError as e:
        raise SystemExit(f"No enhanced chunks found for '{doc_name}'. Run the pipeline first.")

    # Load chunks
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except Exception as e:
        raise SystemExit(f"Error loading chunks: {e}")

    # Convert to LangChain Document format
    chunks = [
        Document(page_content=chunk_data["content"], metadata=chunk_data["metadata"])
        for chunk_data in chunks_data
    ]

    print(f"Uploading {len(chunks)} chunks to Pinecone...")
    print(f"   Index: {args.index}")
    print(f"   Namespace: {args.namespace}")
    print(f"   Profile: {os.getenv('PINECONE_METADATA_PROFILE', 'INFERENCE')}")

    # Upload to Pinecone
    try:
        uploader = OptimizedVectorUploader(
            api_key=api_key,
            index_name=args.index,
            namespace=args.namespace
        )

        stats = uploader.upload_chunks(chunks, show_progress=True)

        print(f"\nUpload complete:")
        print(f"   Uploaded: {stats['uploaded']} chunks")
        print(f"   Skipped: {stats['skipped']} chunks")
        print(f"   Errors: {stats['errors']} chunks")
        if stats.get("size_warnings", 0) > 0:
            print(f"   Size warnings: {stats['size_warnings']} chunks")

        # Save upload stats to manifest if available
        m = load_manifest(doc_name)
        if m:
            upload_entry = {
                "completed": True,
                "timestamp": datetime.now().isoformat(),
                "index": args.index,
                "namespace": args.namespace,
                "stats": stats
            }
            m["vector_upload"] = upload_entry
            save_manifest(doc_name, m)
            print(f"   Saved upload info to manifest")

    except Exception as e:
        raise SystemExit(f"Upload failed: {e}")


def cmd_verify_upload(args: argparse.Namespace) -> None:
    """Verify Pinecone upload and test retrieval"""
    import os
    from src.vector_uploader_optimized import OptimizedVectorUploader

    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY environment variable is required.")

    try:
        # Create uploader just for verification (no upload)
        uploader = OptimizedVectorUploader(
            api_key=api_key,
            index_name=args.index,
            namespace=args.namespace
        )

        # Run verification
        verification = uploader.verify_upload(
            sample_query=args.query,
            expected_chunks=args.expected_chunks or 0
        )

        print(f"\nUpload Verification Results:")
        print(f"   Namespace: {verification['namespace']}")
        print(f"   Total vectors: {verification['total_vectors']}")
        print(f"   Upload success: {verification['upload_success']}")

        query_test = verification['query_test']
        print(f"\nQuery Test ('{query_test['query']}'):")
        print(f"   Results found: {query_test['results_found']}")
        print(f"   Test passed: {query_test['test_passed']}")

        metadata_quality = verification['metadata_quality']
        print(f"\nMetadata Quality (Score: {metadata_quality['quality_score']}):")
        for check, percentage in metadata_quality['details'].items():
            print(f"   {check}: {percentage:.1f}%")

        sample_meta = verification['sample_metadata']
        if sample_meta:
            print(f"\nSample Metadata:")
            print(f"   Full text preserved: {sample_meta['has_full_text']}")
            print(f"   Citation info: {sample_meta['has_citations']}")
            print(f"   Location data: {sample_meta['has_location']}")
            print(f"   Metadata size: {sample_meta['metadata_size_kb']}KB")
            print(f"   Profile: {verification['metadata_profile']}")

        return verification

    except Exception as e:
        raise SystemExit(f"Verification failed: {e}")


def cmd_inspect(args: argparse.Namespace) -> None:
    doc_name = args.doc or Path(args.pdf).stem if args.pdf else None
    if not doc_name:
        raise SystemExit("Provide --doc or --pdf to locate outputs.")
    out_dir = Path("enhanced_output") / doc_name
    stage = getattr(args, "stage", None)

    if stage is None:
        enhanced_file = out_dir / f"enhanced_{doc_name}.json"
        if not enhanced_file.exists():
            print(f"Enhanced file not found at {enhanced_file}. Try running 'stage --stage 4' or 'run'.")
            return
        data = json.loads(enhanced_file.read_text(encoding="utf-8"))
        total = len(data)
        chunks_with_cit = sum(1 for c in data if (c.get("metadata", {}) or {}).get("citations"))
        payload = {"doc": doc_name, "total_chunks": total, "chunks_with_citations": chunks_with_cit}
        print(json.dumps(payload, indent=2) if getattr(args, "json", False) else json.dumps(payload, indent=2))
        return

    if stage == "1":
        stats_p = out_dir / "1_raw_text" / "extraction_stats.json"
        if stats_p.exists():
            stats = json.loads(stats_p.read_text(encoding="utf-8"))
            print(json.dumps({"extraction_stats": stats}, indent=2))
        else:
            print("Stage 1 artifacts not found")
    elif stage == "2":
        stats_p = out_dir / "2_chunks" / "chunking_stats.json"
        if stats_p.exists():
            stats = json.loads(stats_p.read_text(encoding="utf-8"))
            print(json.dumps({"chunking_stats": stats}, indent=2))
        else:
            print("Stage 2 artifacts not found")
    elif stage == "2.5":
        sum_p = out_dir / "2.5_classify" / "classification_summary.json"
        filt_p = out_dir / "2.5_classify" / "filtered_ids.json"
        if sum_p.exists():
            summary = json.loads(sum_p.read_text(encoding="utf-8"))
            filtered = json.loads(filt_p.read_text(encoding="utf-8")) if filt_p.exists() else []
            payload = {
                "classification_counts": summary.get("classification_counts", {}),
                "filtered_out": summary.get("filtered_out", 0),
                "params": summary.get("params", {}),
                "sample_filtered": filtered[:5],
            }
            print(json.dumps(payload, indent=2))
        else:
            print("Stage 2.5 artifacts not found")
    elif stage == "3":
        three_dir = out_dir / "3_enhanced_chunks"
        if three_dir.exists():
            count = len(list(three_dir.glob("enhanced_chunk_*.json")))
            print(json.dumps({"enhanced_chunks": count}, indent=2))
        else:
            print("Stage 3 artifacts not found")
    elif stage == "4":
        stats_p = out_dir / "4_citations" / "citation_stats.json"
        if stats_p.exists():
            stats = json.loads(stats_p.read_text(encoding="utf-8"))
            print(json.dumps({"citation_stats": stats}, indent=2))
        else:
            print("Stage 4 artifacts not found")
    elif stage == "5":
        five_dir = out_dir / "5_citation_index"
        if five_dir.exists():
            citation_index_p = five_dir / "citation_index.json"
            vector_ready_p = five_dir / f"enhanced_{doc_name}_vector_ready.json"

            payload = {"stage5_artifacts": {}}

            if citation_index_p.exists():
                try:
                    with open(citation_index_p, 'r', encoding='utf-8') as f:
                        citation_index = json.load(f)
                    payload["stage5_artifacts"]["citation_index_size"] = len(citation_index)
                    payload["stage5_artifacts"]["citation_index_file"] = str(citation_index_p)
                except Exception:
                    pass

            if vector_ready_p.exists():
                try:
                    with open(vector_ready_p, 'r', encoding='utf-8') as f:
                        vector_chunks = json.load(f)
                    payload["stage5_artifacts"]["vector_ready_chunks"] = len(vector_chunks)
                    payload["stage5_artifacts"]["vector_ready_file"] = str(vector_ready_p)

                    # Calculate size reduction
                    original_file = out_dir / f"enhanced_{doc_name}.json"
                    if original_file.exists():
                        original_size = original_file.stat().st_size / 1024  # KB
                        optimized_size = vector_ready_p.stat().st_size / 1024  # KB
                        index_size = citation_index_p.stat().st_size / 1024 if citation_index_p.exists() else 0

                        payload["stage5_artifacts"]["size_reduction"] = {
                            "original_size_kb": round(original_size, 1),
                            "optimized_size_kb": round(optimized_size, 1),
                            "citation_index_kb": round(index_size, 1),
                            "total_new_size_kb": round(optimized_size + index_size, 1),
                            "reduction_percent": round((1 - (optimized_size + index_size) / original_size) * 100, 1)
                        }
                except Exception:
                    pass

            print(json.dumps(payload, indent=2))
        else:
            print("Stage 5 artifacts not found")
    else:
        raise SystemExit(f"Unknown stage: {stage}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Thera-RAG Ingestion CLI")
    sub = p.add_subparsers(dest="cmd")

    # init
    p_init = sub.add_parser("init", help="Initialize manifest for a PDF")
    p_init.add_argument("--pdf", default=str(Path("data") / "vha-guideline.pdf"))
    p_init.add_argument("--text-only", action="store_true")

    # status
    p_status = sub.add_parser("status", help="Show manifest summary")
    p_status.add_argument("--doc", default=None)
    p_status.add_argument("--pdf", default=None)

    # stage (run a single stage)
    p_stage = sub.add_parser("stage", help="Run a single stage")
    p_stage.add_argument("--pdf", default=str(Path("data") / "vha-guideline.pdf"))
    p_stage.add_argument("--output", default="enhanced_output")
    p_stage.add_argument("--stage", required=True, choices=STAGE_ORDER)
    # Stage 1 params
    p_stage.add_argument("--text-only", action="store_true")
    # Stage 2.5 params
    p_stage.add_argument("--density-threshold", type=float, default=0.7)
    p_stage.add_argument("--keep", nargs="*", help="Labels to keep (e.g., references appendix)")

    # approve
    p_approve = sub.add_parser("approve", help="Approve a stage")
    p_approve.add_argument("--doc", default=None)
    p_approve.add_argument("--pdf", default=None)
    p_approve.add_argument("--stage", required=True, choices=STAGE_ORDER)
    p_approve.add_argument("--keep", nargs="*", help="Stage 2.5: keep labels override at approval time")

    # reject
    p_reject = sub.add_parser("reject", help="Reject a stage (removes outputs and manifest entry)")
    p_reject.add_argument("--doc", default=None)
    p_reject.add_argument("--pdf", default=None)
    p_reject.add_argument("--stage", required=True, choices=STAGE_ORDER)

    # run
    p_run = sub.add_parser("run", help="Run sequentially and stop at first unapproved stage")
    p_run.add_argument("--pdf", default=str(Path("data") / "vha-guideline.pdf"))
    p_run.add_argument("--output", default="enhanced_output")
    p_run.add_argument("--text-only", action="store_true")
    p_run.add_argument("--auto-continue", action="store_true", help="Run through stages without stopping for approval")
    p_run.add_argument("--force", action="store_true", help="Re-run stages even if already approved")
    p_run.add_argument("--from-stage", dest="from_stage", default=None, choices=STAGE_ORDER)
    p_run.add_argument("--until-stage", dest="until_stage", default=None, choices=STAGE_ORDER)

# inspect
    p_inspect = sub.add_parser("inspect", help="Inspect outputs (optionally per stage)")
    p_inspect.add_argument("--doc", default=None)
    p_inspect.add_argument("--pdf", default=None)
    p_inspect.add_argument("--stage", default=None, choices=STAGE_ORDER)
    p_inspect.add_argument("--json", action="store_true")

    # upload
    p_upload = sub.add_parser("upload", help="Upload optimized chunks to Pinecone")
    p_upload.add_argument("--doc", default=None)
    p_upload.add_argument("--pdf", default=str(Path("data") / "vha-guideline.pdf"))
    p_upload.add_argument("--index", default="medical-rag-index", help="Pinecone index name")
    p_upload.add_argument("--namespace", default="thera-rag", help="Pinecone namespace")
    p_upload.add_argument("--profile", choices=["MINIMAL", "INFERENCE", "FULL"], help="Metadata profile")

    # verify-upload
    p_verify = sub.add_parser("verify-upload", help="Verify Pinecone upload and test retrieval")
    p_verify.add_argument("--index", default="medical-rag-index", help="Pinecone index name")
    p_verify.add_argument("--namespace", default="thera-rag", help="Pinecone namespace")
    p_verify.add_argument("--query", default="medical treatment", help="Test query")
    p_verify.add_argument("--expected-chunks", type=int, help="Expected number of chunks")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    if args.cmd == "init":
        cmd_init(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "stage":
        cmd_stage(args)
    elif args.cmd == "approve":
        cmd_approve(args)
    elif args.cmd == "reject":
        cmd_reject(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "inspect":
        cmd_inspect(args)
    elif args.cmd == "upload":
        cmd_upload(args)
    elif args.cmd == "verify-upload":
        cmd_verify_upload(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()







