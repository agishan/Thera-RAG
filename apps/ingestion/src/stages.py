from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

from langchain_core.documents import Document

from .document_processor import DoclingBookLoader
from .chunk_processor import SmartChunker, ChunkingConfig
from .chunk_classifier import ChunkClassifier
from .metadata_enhancer import DocumentMetadataExtractor, ChunkMetadataEnhancer
from .citation_extractor import CitationExtractor
from .citation_index_builder import CitationIndexBuilder


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def stage1_extract(pdf_path: Path, out_dir: Path, text_only: bool = False) -> Dict[str, Any]:
    loader = DoclingBookLoader(str(pdf_path), text_only=text_only)
    structured = loader.extract_structured_content()
    full_text = structured.get("full_text") or structured.get("text", "")
    stats = {
        "characters": len(full_text),
        "words": len(full_text.split()),
        "lines": len(full_text.split("\n")),
        "paragraphs": len([p for p in full_text.split("\n\n") if p.strip()]),
    }
    stage_dir = out_dir / "1_raw_text"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "full_text.txt").write_text(full_text, encoding="utf-8")
    _write_json(stage_dir / "structured_content.json", structured)
    _write_json(stage_dir / "extraction_stats.json", stats)
    return {"stats": stats, "artifacts": [str(stage_dir / "full_text.txt"), str(stage_dir / "structured_content.json")]}


def _load_full_text(out_dir: Path) -> str:
    p = out_dir / "1_raw_text" / "full_text.txt"
    return p.read_text(encoding="utf-8")


def stage2_chunk(pdf_path: Path, out_dir: Path, cfg: ChunkingConfig | None = None) -> Dict[str, Any]:
    # Always re-extract with Docling to get structured document for chunking
    print("Extracting document with Docling for structure-aware chunking...")
    from .document_processor import DoclingBookLoader

    # Create loader with same settings as stage 1 (check if text_only was used)
    try:
        structured_path = out_dir / "1_raw_text" / "structured_content.json"
        text_only = False
        if structured_path.exists():
            with open(structured_path, 'r', encoding='utf-8') as f:
                import json
                structured_saved = json.load(f)
                text_only = structured_saved.get('metadata', {}).get('text_only_mode', False)
    except Exception:
        text_only = False

    loader = DoclingBookLoader(str(pdf_path), text_only=text_only)
    structured = loader.extract_structured_content()
    docling_doc = structured.get('raw_document')

    chunker = SmartChunker(cfg or ChunkingConfig())
    base_metadata = {"source_file": pdf_path.name, "processed_at": datetime.now().isoformat()}

    # Use Docling HierarchicalChunker
    if docling_doc is not None:
        chunks: List[Document] = chunker.create_chunks(docling_doc, base_metadata)
    else:
        raise RuntimeError("No Docling document available - stage 1 may have failed")
    stage_dir = out_dir / "2_chunks"
    stage_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for i, ch in enumerate(chunks, 1):
        path = stage_dir / f"chunk_{i:03d}.json"
        _write_json(path, {"content": ch.page_content, "metadata": ch.metadata})
        artifacts.append(str(path))
    stats = {
        "count": len(chunks),
        "size_min": min((len(c.page_content) for c in chunks), default=0),
        "size_max": max((len(c.page_content) for c in chunks), default=0),
        "size_mean": (sum(len(c.page_content) for c in chunks) / len(chunks)) if chunks else 0,
    }
    _write_json(stage_dir / "chunking_stats.json", stats)
    return {"stats": stats, "artifacts": artifacts}


def _iter_stage2_chunks(out_dir: Path) -> List[Dict[str, Any]]:
    stage_dir = out_dir / "2_chunks"
    items = []
    for p in sorted(stage_dir.glob("chunk_*.json")):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return items


def stage2_5_classify(out_dir: Path, density_threshold: float = 0.7, keep_labels: List[str] | None = None) -> Dict[str, Any]:
    keep_labels = set(keep_labels or [])
    items = _iter_stage2_chunks(out_dir)
    classifier = ChunkClassifier()
    classification_counts: Dict[str, int] = {}
    filtered_ids: List[str] = []

    for item in items:
        content = item.get("content", "")
        meta = item.get("metadata", {}) or {}
        res = classifier.classify(content)
        label = res.get("label", "medical_content")
        score = float(res.get("score", 0.0))
        meta["chunk_category"] = label
        meta["reference_likeness"] = score
        # Update back
        item["metadata"] = meta

        classification_counts[label] = classification_counts.get(label, 0) + 1
        # Filter logic
        if label in {"references", "appendix", "acknowledgments", "keywords"} and label not in keep_labels:
            filtered_ids.append(meta.get("chunk_id") or meta.get("id") or "")

    # Save classification summary
    stage_dir = out_dir / "2.5_classify"
    stage_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "classification_counts": classification_counts,
        "filtered_out": len(filtered_ids),
        "params": {"density_threshold": density_threshold, "keep": list(keep_labels)},
    }
    _write_json(stage_dir / "classification_summary.json", summary)
    _write_json(stage_dir / "filtered_ids.json", filtered_ids)
    # Also write back updated 2_chunks with enriched metadata
    # (No overwriting chunk content)
    two_dir = out_dir / "2_chunks"
    for i, item in enumerate(items, 1):
        _write_json(two_dir / f"chunk_{i:03d}.json", item)

    return {"stats": summary, "artifacts": [str(stage_dir / "classification_summary.json")]} 


def _load_structured(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "1_raw_text" / "structured_content.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def stage3_enhance(pdf_path: Path, out_dir: Path) -> Dict[str, Any]:
    full_text = _load_full_text(out_dir)
    structured = _load_structured(out_dir)
    # Load chunks from 2 and apply filter from 2.5
    items = _iter_stage2_chunks(out_dir)
    filtered_ids = set(json.loads((out_dir / "2.5_classify" / "filtered_ids.json").read_text(encoding="utf-8"))
                       if (out_dir / "2.5_classify" / "filtered_ids.json").exists() else [])

    # Convert to LangChain Documents
    chunks: List[Document] = []
    for item in items:
        meta = item.get("metadata", {}) or {}
        cid = meta.get("chunk_id") or meta.get("id")
        if cid and cid in filtered_ids:
            continue
        chunks.append(Document(page_content=item.get("content", ""), metadata=meta))

    doc_meta_extractor = DocumentMetadataExtractor()
    doc_ref = doc_meta_extractor.extract_document_metadata(full_text, pdf_path.name, structured)
    enhancer = ChunkMetadataEnhancer(doc_ref)

    stage_dir = out_dir / "3_enhanced_chunks"
    stage_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[str] = []

    enhanced: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        ench = enhancer.enhance_chunk_metadata(ch, full_text, i)
        path = stage_dir / f"enhanced_chunk_{i+1:03d}.json"
        _write_json(path, {"content": ench.page_content, "metadata": ench.metadata})
        artifacts.append(str(path))
        enhanced.append({"content": ench.page_content, "metadata": ench.metadata})

    # Sample metadata
    if enhanced:
        _write_json(stage_dir / "sample_metadata.json", enhanced[0]["metadata"])

    stats = {"count": len(enhanced)}
    return {"stats": stats, "artifacts": artifacts}


def stage4_citations(out_dir: Path, google_api_key: str | None) -> Dict[str, Any]:
    full_text = _load_full_text(out_dir)
    # Load enhanced chunks
    three_dir = out_dir / "3_enhanced_chunks"
    enhanced: List[Document] = []
    for p in sorted(three_dir.glob("enhanced_chunk_*.json")):
        item = json.loads(p.read_text(encoding="utf-8"))
        enhanced.append(Document(page_content=item.get("content", ""), metadata=item.get("metadata", {}) or {}))

    extractor = CitationExtractor(llm_api_key=google_api_key, use_llm=bool(google_api_key))
    citations, matches = extractor.extract_citations_from_document(full_text, enhanced)
    final_chunks = extractor.enhance_chunks_with_citations(enhanced, matches)

    stage_dir = out_dir / "4_citations"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Write artifacts
    # extracted citations
    _write_json(stage_dir / "extracted_citations.json", [
        {
            "inline_text": c.inline_text,
            "full_reference": c.full_reference,
            "authors": c.authors,
            "year": c.year,
            "title": c.title,
            "journal": c.journal,
            "doi": c.doi,
            "citation_type": c.citation_type,
            "confidence": c.confidence,
        }
        for c in citations
    ])

    # matches
    _write_json(stage_dir / "citation_matches.json", [
        {
            "inline_citation": m.inline_citation,
            "chunk_location": m.chunk_location,
            "context_before": m.context_before,
            "context_after": m.context_after,
            "full_reference": {
                "inline_text": m.full_reference.inline_text,
                "authors": m.full_reference.authors,
                "year": m.full_reference.year,
                "title": m.full_reference.title,
            },
        }
        for m in matches
    ])

    # final chunks with citations
    final_serialized = [{"content": d.page_content, "metadata": d.metadata} for d in final_chunks]
    _write_json(stage_dir / "final_chunks_with_citations.json", final_serialized)

    # consolidated enhanced file
    _write_json(out_dir / f"enhanced_{out_dir.name}.json", final_serialized)

    stats = {
        "citations_extracted": len(citations),
        "citation_matches": len(matches),
        "chunks_with_citations": sum(1 for d in final_chunks if d.metadata.get("citations")),
    }
    _write_json(stage_dir / "citation_stats.json", stats)
    return {"stats": stats, "artifacts": [str(stage_dir / "final_chunks_with_citations.json")]}


def stage5_citation_optimization(out_dir: Path) -> Dict[str, Any]:
    """
    Stage 5: Citation optimization - Create citation index and lightweight chunk references

    This stage processes the enhanced chunks from Stage 4, extracts all citations,
    deduplicates them into a citation index, and creates lightweight chunk references
    for optimized vector storage.
    """
    print("Stage 5: Citation optimization")

    # Input: Enhanced chunks from Stage 4
    enhanced_chunks_file = out_dir / f"enhanced_{out_dir.name}.json"
    if not enhanced_chunks_file.exists():
        raise FileNotFoundError(f"Enhanced chunks file not found: {enhanced_chunks_file}")

    # Output directory
    stage_dir = out_dir / "5_citation_index"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Build citation index and create lightweight chunks
    builder = CitationIndexBuilder()
    stats = builder.process_enhanced_chunks(enhanced_chunks_file, stage_dir)

    # Also create the optimized vector-ready chunks
    lightweight_file = stage_dir / f"enhanced_{out_dir.name}_lightweight.json"
    vector_ready_file = stage_dir / f"enhanced_{out_dir.name}_vector_ready.json"

    # Copy lightweight file to vector-ready name for clarity
    if lightweight_file.exists():
        import shutil
        shutil.copy2(lightweight_file, vector_ready_file)
        stats["artifacts"].append(str(vector_ready_file))

    print(f"Stage 5 complete:")
    print(f"   Citation index: {len(builder.citation_index)} unique citations")
    print(f"   Size reduction: {stats.get('size_reduction_percent', 0)}%")
    print(f"   Vector-ready chunks: {vector_ready_file}")

    return {"stats": stats, "artifacts": stats.get("artifacts", [])}


def get_vector_ready_chunks(out_dir: Path) -> Path:
    """
    Get path to vector-ready chunks file (optimized for Pinecone upload)

    Returns path to lightweight chunks file if Stage 5 has been run,
    otherwise returns original enhanced chunks file.
    """
    # Check for Stage 5 optimized chunks first
    vector_ready_file = out_dir / "5_citation_index" / f"enhanced_{out_dir.name}_vector_ready.json"
    if vector_ready_file.exists():
        return vector_ready_file

    # Fallback to Stage 4 enhanced chunks
    enhanced_chunks_file = out_dir / f"enhanced_{out_dir.name}.json"
    if enhanced_chunks_file.exists():
        return enhanced_chunks_file

    raise FileNotFoundError(f"No enhanced chunks found in {out_dir}")

