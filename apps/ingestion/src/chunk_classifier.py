from __future__ import annotations

import re
from typing import Dict, Any


class ChunkClassifier:
    """
    Lightweight classifier to label chunks as 'references', 'appendix', 'acknowledgments', 'keywords', or 'medical_content'.
    Uses header cues and reference-density heuristics similar to the citation extractor.
    """

    REF_HEADER_PATTERNS = [
        r"(?i)^#+\s*references?\s*$",
        r"(?i)^#+\s*bibliography\s*$",
        r"(?i)^#+\s*works?\s+cited\s*$",
        r"(?i)^\s*references?\s*$",
        r"(?i)^\s*bibliography\s*$",
        r"(?i)^\s*works?\s+cited\s*$",
    ]

    APP_HEADER_PATTERNS = [
        r"(?i)^#+\s*appendi(?:x|ces)\b",
        r"(?i)^\s*appendi(?:x|ces)\b",
        r"(?i)^\s*supplementary\s+(material|data|information)\b",
    ]

    ACK_HEADER_PATTERNS = [
        r"(?i)^#+\s*acknowled?g(e)?ments?\b",
        r"(?i)^\s*acknowled?g(e)?ments?\b",
        r"(?i)^\s*funding\b",
        r"(?i)^\s*conflicts?\s+of\s+interest\b",
        r"(?i)^\s*disclosures?\b",
    ]

    KW_HEADER_PATTERNS = [
        r"(?i)^\s*keywords?\s*:",
        r"(?i)^\s*key\s+words?\s*:",
    ]

    def classify(self, text: str) -> Dict[str, Any]:
        text = text or ""
        header_line = (text.strip().split("\n") or [""])[0].strip()

        # Header-based shortcut
        for pat in self.REF_HEADER_PATTERNS:
            if re.match(pat, header_line):
                return {"label": "references", "score": 1.0}
        for pat in self.APP_HEADER_PATTERNS:
            if re.match(pat, header_line):
                return {"label": "appendix", "score": 0.9}
        for pat in self.ACK_HEADER_PATTERNS:
            if re.match(pat, header_line):
                return {"label": "acknowledgments", "score": 0.9}
        for pat in self.KW_HEADER_PATTERNS:
            if re.match(pat, header_line):
                return {"label": "keywords", "score": 0.8}

        # Density heuristic for references-like content
        density = self._reference_density(text)
        if density >= 0.7:
            return {"label": "references", "score": float(density)}

        return {"label": "medical_content", "score": 1.0 - float(density)}

    def _reference_density(self, text_block: str) -> float:
        if not text_block.strip():
            return 0.0

        lines = [ln.strip() for ln in text_block.split("\n") if ln.strip()]
        if len(lines) < 3:
            return 0.0

        score = 0.0
        indicators = 0
        for line in lines:
            line_score = 0.0
            if re.match(r"^\d+[\.|\)]\s+", line):
                line_score += 0.3
            elif re.match(r"^\[\d+\]\s+", line):
                line_score += 0.3
            if re.search(r"\(\d{4}[a-z]?\)", line) or re.search(r"\b\d{4}[a-z]?\b", line):
                line_score += 0.2
            if re.search(r"\bvol\.?\s*\d+|\bvolume\s*\d+|\bissue\s*\d+", line, re.I):
                line_score += 0.15
            if re.search(r"doi:|https?://|www\.", line, re.I):
                line_score += 0.15
            if re.search(r"\bpp?\.\s*\d+|\b\d+[--]\d+\b", line):
                line_score += 0.1
            if len(line) > 50 and line.count(',') >= 2:
                line_score += 0.1
            if line_score > 0.1:
                indicators += 1
                score += line_score

        density = indicators / len(lines)
        normalized = (score / len(lines)) * (1 + density)
        return min(1.0, normalized)

