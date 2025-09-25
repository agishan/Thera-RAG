from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Robust imports to work whether run as package or script
try:
    from ..src.document_processor import DoclingBookLoader
    from ..src.medical_document_processor import MedicalDocumentProcessor
except Exception:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from document_processor import DoclingBookLoader
    from medical_document_processor import MedicalDocumentProcessor


def _calc_text_stats(text: str) -> Dict[str, Any]:
    return {
        "characters": len(text or ""),
        "words": len((text or "").split()),
        "lines": len((text or "").split("\n")),
        "paragraphs": len([p for p in (text or "").split("\n\n") if p.strip()]),
    }


def _stage_dir_for(pdf_path: Path, base_dir: Path = Path("enhanced_output")) -> Path:
    doc_name = pdf_path.stem
    stage_dir = base_dir / doc_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def run_stage1(pdf_path: str, text_only: bool = False) -> Dict[str, Any]:
    """Run Stage 1: Docling extraction.

    Returns: dict with full_text, structured_content, stats, stage_dir, doc_name.
    Writes artifacts under 1_raw_text/.
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    loader = DoclingBookLoader(str(pdf), text_only=text_only)
    structured = loader.extract_structured_content()
    full_text = structured.get("full_text") or structured.get("text", "")
    stats = _calc_text_stats(full_text)

    stage_dir = _stage_dir_for(pdf)
    out_dir = stage_dir / "1_raw_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    (out_dir / "full_text.txt").write_text(full_text, encoding="utf-8")
    (out_dir / "structured_content.json").write_text(
        json.dumps(structured, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out_dir / "extraction_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    return {
        "full_text": full_text,
        "structured_content": structured,
        "stats": stats,
        "stage_dir": str(stage_dir),
        "doc_name": pdf.stem,
        "text_only": text_only,
        "timestamp": datetime.now().isoformat(),
    }


def run_stage1_5(full_text: str, stage_dir: str, exclude_references: bool = True) -> Dict[str, Any]:
    """Run Stage 1.5: Medical filtering on text.

    Returns: dict with filtered_text, sections, analysis, stats.
    Writes artifacts under 1.5_medical_filtering/.
    """
    med = MedicalDocumentProcessor(exclude_references=exclude_references)
    sections = med.detect_sections(full_text)
    filtered_text = med.filter_for_rag(sections)
    analysis = med.analyze_document_structure(full_text)

    stats = {
        "original_length": len(full_text or ""),
        "filtered_length": len(filtered_text or ""),
        "reduction_percent": (1 - len(filtered_text or "") / max(1, len(full_text or ""))) * 100,
        "sections_detected": len(sections),
        "references_filtered": "references" in sections,
    }

    out_dir = Path(stage_dir) / "1.5_medical_filtering"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    (out_dir / "filtered_medical_content.txt").write_text(filtered_text, encoding="utf-8")

    sections_data = {}
    for name, section in sections.items():
        sections_data[name] = {
            "content_length": len(section.content),
            "start_pos": section.start_pos,
            "end_pos": section.end_pos,
            "confidence": section.confidence,
            "content_preview": section.content[:200] + ("..." if len(section.content) > 200 else ""),
        }
    (out_dir / "detected_sections.json").write_text(
        json.dumps(sections_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (out_dir / "filtering_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    analysis_json = dict(analysis)
    analysis_json.pop("sections", None)
    (out_dir / "document_analysis.json").write_text(
        json.dumps(analysis_json, indent=2, default=str), encoding="utf-8"
    )

    return {
        "filtered_text": filtered_text,
        "sections": sections,
        "analysis": analysis,
        "stats": stats,
        "stage_dir": stage_dir,
        "timestamp": datetime.now().isoformat(),
    }
