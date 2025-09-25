"""
Medical Document Structure Detection and Processing

Specialized processor for medical papers that can:
1. Detect standard medical paper sections (Abstract, Methods, Results, etc.)
2. Filter out references and non-content sections
3. Optimize content for RAG applications
"""

import re
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextSpan:
    """Represents a span of text with start/end positions"""
    start: int
    end: int
    content: str
    section_type: str
    confidence: float = 1.0


class DocumentSection(NamedTuple):
    """Named tuple for document sections"""
    name: str
    content: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0


class MedicalDocumentProcessor:
    """
    Processor specialized for medical/scientific documents

    Detects common sections and filters content for RAG optimization
    """

    # Comprehensive medical paper section patterns
    SECTION_PATTERNS = {
        'title': [
            r'(?i)^\s*(?:title\s*:?\s*)?(.{10,200}?)(?:\n|$)',
        ],
        'guideline_section': [
            r'^##\s+(.+)$',  # Markdown-style headers like "## Methodology"
        ],
        'abstract': [
            r'(?i)^\s*abstract\s*:?\s*$',
            r'(?i)^\s*summary\s*:?\s*$',
            r'(?i)^\s*executive\s+summary\s*:?\s*$',
            r'(?i)^\s*overview\s*:?\s*$'
        ],
        'keywords': [
            r'(?i)^\s*keywords?\s*:?\s*',
            r'(?i)^\s*key\s+words?\s*:?\s*',
            r'(?i)^\s*index\s+terms?\s*:?\s*'
        ],
        'introduction': [
            r'(?i)^\s*(?:1\.?\s*)?introduction\s*:?\s*$',
            r'(?i)^\s*(?:1\.?\s*)?background\s*:?\s*$',
            r'(?i)^\s*(?:1\.?\s*)?objective\s*:?\s*$',
            r'(?i)^\s*(?:1\.?\s*)?rationale\s*:?\s*$'
        ],
        'methods': [
            r'(?i)^\s*(?:2\.?\s*)?methods?\s*:?\s*$',
            r'(?i)^\s*(?:2\.?\s*)?methodology\s*:?\s*$',
            r'(?i)^\s*materials?\s+and\s+methods?\s*:?\s*$',
            r'(?i)^\s*(?:2\.?\s*)?experimental\s+(?:design|procedure|methods?)\s*:?\s*$',
            r'(?i)^\s*(?:2\.?\s*)?study\s+design\s*:?\s*$',
            r'(?i)^\s*(?:2\.?\s*)?participants?\s*:?\s*$',
            r'(?i)^\s*(?:2\.?\s*)?procedures?\s*:?\s*$'
        ],
        'results': [
            r'(?i)^\s*(?:3\.?\s*)?results?\s*:?\s*$',
            r'(?i)^\s*(?:3\.?\s*)?findings?\s*:?\s*$',
            r'(?i)^\s*(?:3\.?\s*)?outcomes?\s*:?\s*$',
            r'(?i)^\s*(?:3\.?\s*)?observations?\s*:?\s*$'
        ],
        'discussion': [
            r'(?i)^\s*(?:4\.?\s*)?discussion\s*:?\s*$',
            r'(?i)^\s*(?:4\.?\s*)?analysis\s*:?\s*$',
            r'(?i)^\s*(?:4\.?\s*)?interpretation\s*:?\s*$',
            r'(?i)^\s*(?:4\.?\s*)?clinical\s+implications?\s*:?\s*$'
        ],
        'conclusion': [
            r'(?i)^\s*(?:5\.?\s*)?conclusions?\s*:?\s*$',
            r'(?i)^\s*(?:5\.?\s*)?summary\s*:?\s*$',
            r'(?i)^\s*(?:5\.?\s*)?implications?\s*:?\s*$',
            r'(?i)^\s*(?:5\.?\s*)?future\s+(?:work|research|directions?)\s*:?\s*$'
        ],
        'limitations': [
            r'(?i)^\s*limitations?\s*:?\s*$',
            r'(?i)^\s*study\s+limitations?\s*:?\s*$'
        ],
        'references': [
            r'(?i)^\s*references?\s*:?\s*$',
            r'(?i)^\s*bibliography\s*:?\s*$',
            r'(?i)^\s*citations?\s*:?\s*$',
            r'(?i)^\s*(?:\d+\.?\s*)?references?\s*$',
            r'(?i)^\s*literature\s+cited\s*:?\s*$',
            r'(?i)^\s*works?\s+cited\s*:?\s*$'
        ],
        'appendices': [
            r'(?i)^\s*appendi(?:x|ces)\s*(?:[a-z]|\d+)?\s*:?\s*$',
            r'(?i)^\s*supplementary\s+(?:material|data|information)\s*:?\s*$',
            r'(?i)^\s*supporting\s+information\s*:?\s*$',
            r'(?i)^\s*additional\s+(?:material|data)\s*:?\s*$'
        ],
        'acknowledgments': [
            r'(?i)^\s*acknowledgments?\s*:?\s*$',
            r'(?i)^\s*acknowledgements?\s*:?\s*$',
            r'(?i)^\s*funding\s*:?\s*$',
            r'(?i)^\s*financial\s+support\s*:?\s*$',
            r'(?i)^\s*conflicts?\s+of\s+interest\s*:?\s*$',
            r'(?i)^\s*disclosures?\s*:?\s*$'
        ],
        'ethics': [
            r'(?i)^\s*ethics?\s*:?\s*$',
            r'(?i)^\s*ethical?\s+(?:approval|considerations?)\s*:?\s*$',
            r'(?i)^\s*institutional\s+review\s+board\s*:?\s*$',
            r'(?i)^\s*irb\s+approval\s*:?\s*$'
        ]
    }

    # Sections to include in RAG content (medical knowledge)
    RAG_RELEVANT_SECTIONS = {
        'abstract', 'introduction', 'methods', 'results', 'discussion',
        'conclusion', 'limitations', 'ethics', 'practice_points', 'quality_control',
        # Guideline-specific sections that contain medical knowledge
        'machine_methodology_quality_assurance_and_test', 'sample_type_and_pre_analytical_issues',
        'precision_and_accuracy_of_testing', 'internal_quality_control', 'external_quality_assurance',
        'reference_ranges', 'vha_traces', 'obstetric_and_postpartum_bleeding',
        'prediction_of_bleeding_coagulopathy', 'diagnosis_of_bleeding_coagulopathy',
        'use_of_rotem_teg_sonoclot_for_guiding_transfusion', 'liver_disease_and_liver_surgery',
        'cardiac_surgery', 'trauma_haemorrhage'
    }

    # Sections to exclude from RAG (noise for medical context)
    RAG_EXCLUDE_SECTIONS = {
        'references', 'appendices', 'acknowledgments', 'keywords', 'title',
        'review_of_the_manuscript'  # Administrative content
    }

    def __init__(self, exclude_references: bool = True, include_abstract: bool = True):
        """
        Initialize medical document processor

        Args:
            exclude_references: Whether to filter out reference sections
            include_abstract: Whether to include abstract in RAG content
        """
        self.exclude_references = exclude_references
        self.include_abstract = include_abstract

    def detect_sections(self, text: str) -> Dict[str, DocumentSection]:
        """
        Detect document sections using pattern matching

        Args:
            text: Full document text

        Returns:
            Dictionary mapping section names to DocumentSection objects
        """
        # First try to detect markdown-style sections (like guidelines)
        markdown_sections = self._detect_markdown_sections(text)
        if len(markdown_sections) > 3:  # If we found many markdown sections, use that
            return markdown_sections

        # Otherwise use traditional academic paper detection
        return self._detect_traditional_sections(text)

    def _detect_markdown_sections(self, text: str) -> Dict[str, DocumentSection]:
        """Detect sections using markdown-style headers (## Section Name)"""
        sections = {}
        lines = text.split('\n')

        current_section = None
        section_start = 0
        section_content = []
        char_position = 0

        for i, line in enumerate(lines):
            line_start_pos = char_position
            char_position += len(line) + 1  # +1 for newline

            # Check for markdown headers
            if re.match(r'^##\s+(.+)$', line.strip()):
                # Save previous section if exists
                if current_section and section_content:
                    content = '\n'.join(section_content).strip()
                    if content:
                        sections[current_section] = DocumentSection(
                            name=current_section,
                            content=content,
                            start_pos=section_start,
                            end_pos=line_start_pos - 1,
                            confidence=0.95  # High confidence for markdown headers
                        )

                # Extract section name and classify it
                section_title = re.match(r'^##\s+(.+)$', line.strip()).group(1)
                current_section = self._classify_guideline_section(section_title)
                section_start = line_start_pos
                section_content = [line]  # Include the header
            else:
                # Add to current section
                if current_section:
                    section_content.append(line)
                elif not sections:  # Before any section detected
                    if 'preamble' not in sections:
                        sections['preamble'] = DocumentSection(
                            name='preamble',
                            content='',
                            start_pos=0,
                            end_pos=0,
                            confidence=0.5
                        )
                    current_content = sections['preamble'].content
                    sections['preamble'] = sections['preamble']._replace(
                        content=current_content + line + '\n',
                        end_pos=char_position
                    )

        # Don't forget the last section
        if current_section and section_content:
            content = '\n'.join(section_content).strip()
            if content:
                sections[current_section] = DocumentSection(
                    name=current_section,
                    content=content,
                    start_pos=section_start,
                    end_pos=len(text),
                    confidence=0.95
                )

        return sections

    def _detect_traditional_sections(self, text: str) -> Dict[str, DocumentSection]:
        """Detect sections using traditional academic paper patterns"""
        sections = {}
        lines = text.split('\n')

        current_section = None
        section_start = 0
        section_content = []
        char_position = 0

        for i, line in enumerate(lines):
            line_start_pos = char_position
            char_position += len(line) + 1  # +1 for newline

            # Check if this line matches any section pattern
            detected_section = self._match_section_pattern(line)

            if detected_section:
                # Save previous section if exists
                if current_section and section_content:
                    content = '\n'.join(section_content).strip()
                    if content:  # Only save non-empty sections
                        sections[current_section] = DocumentSection(
                            name=current_section,
                            content=content,
                            start_pos=section_start,
                            end_pos=line_start_pos - 1,
                            confidence=0.9  # High confidence for pattern matches
                        )

                # Start new section
                current_section = detected_section
                section_start = line_start_pos
                section_content = [line]  # Include the header
            else:
                # Add to current section
                if current_section:
                    section_content.append(line)
                elif not sections:  # Before any section detected - treat as preamble
                    if 'preamble' not in sections:
                        sections['preamble'] = DocumentSection(
                            name='preamble',
                            content='',
                            start_pos=0,
                            end_pos=0,
                            confidence=0.5
                        )
                    current_content = sections['preamble'].content
                    sections['preamble'] = sections['preamble']._replace(
                        content=current_content + line + '\n',
                        end_pos=char_position
                    )

        # Don't forget the last section
        if current_section and section_content:
            content = '\n'.join(section_content).strip()
            if content:
                sections[current_section] = DocumentSection(
                    name=current_section,
                    content=content,
                    start_pos=section_start,
                    end_pos=len(text),
                    confidence=0.9
                )

        return sections

    def _match_section_pattern(self, line: str) -> Optional[str]:
        """
        Check if a line matches any section pattern

        Args:
            line: Text line to check

        Returns:
            Section name if match found, None otherwise
        """
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 3:
            return None

        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, line_stripped):
                    return section_name

        return None

    def _classify_guideline_section(self, section_title: str) -> str:
        """
        Classify guideline section based on its title

        Args:
            section_title: The section title from markdown header

        Returns:
            Classified section type
        """
        title_lower = section_title.lower().strip()

        # Map guideline sections to our standard categories
        if any(word in title_lower for word in ['methodology', 'methods', 'method']):
            return 'methods'
        elif any(word in title_lower for word in ['results', 'findings', 'outcomes']):
            return 'results'
        elif any(word in title_lower for word in ['discussion', 'analysis', 'interpretation']):
            return 'discussion'
        elif any(word in title_lower for word in ['conclusion', 'summary', 'recommendations']):
            return 'conclusion'
        elif any(word in title_lower for word in ['introduction', 'background', 'overview']):
            return 'introduction'
        elif any(word in title_lower for word in ['references', 'bibliography', 'citations']):
            return 'references'
        elif any(word in title_lower for word in ['appendix', 'appendices', 'supplementary']):
            return 'appendices'
        elif any(word in title_lower for word in ['acknowledgment', 'funding', 'conflicts']):
            return 'acknowledgments'
        elif 'practice points' in title_lower or 'key points' in title_lower:
            return 'practice_points'  # Special category for guidelines
        elif any(word in title_lower for word in ['quality', 'assurance', 'control']):
            return 'quality_control'  # Special category for guidelines
        else:
            # Use a simplified version of the title as section name
            simplified = re.sub(r'[^a-zA-Z0-9\s]', '', title_lower)
            simplified = re.sub(r'\s+', '_', simplified.strip())
            return simplified[:50]  # Limit length

    def filter_for_rag(self, sections: Dict[str, DocumentSection]) -> str:
        """
        Filter document sections to keep only RAG-relevant content

        Args:
            sections: Dictionary of detected sections

        Returns:
            Filtered text containing only medical content relevant for RAG
        """
        rag_content = []

        # Include relevant sections in logical order
        section_order = [
            'abstract', 'introduction', 'methods', 'results',
            'discussion', 'conclusion', 'limitations', 'ethics'
        ]

        for section_name in section_order:
            if section_name in sections:
                # Apply inclusion/exclusion rules
                if section_name in self.RAG_RELEVANT_SECTIONS:
                    if section_name == 'abstract' and not self.include_abstract:
                        continue

                    section = sections[section_name]
                    # Clean the section content
                    clean_content = self._clean_section_content(section.content, section_name)
                    if clean_content.strip():
                        rag_content.append(f"## {section.name.title()}\n{clean_content}")

        # Add any other detected relevant sections not in standard order
        for section_name, section in sections.items():
            if (section_name not in section_order and
                section_name in self.RAG_RELEVANT_SECTIONS and
                section_name not in self.RAG_EXCLUDE_SECTIONS):
                clean_content = self._clean_section_content(section.content, section_name)
                if clean_content.strip():
                    rag_content.append(f"## {section.name.title()}\n{clean_content}")

        return '\n\n'.join(rag_content)

    def _clean_section_content(self, content: str, section_name: str) -> str:
        """
        Clean section content by removing headers and noise

        Args:
            content: Raw section content
            section_name: Name of the section

        Returns:
            Cleaned content
        """
        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                cleaned_lines.append('')
                continue

            # Skip section headers that we already detected
            if self._match_section_pattern(line):
                continue

            # Skip obvious formatting artifacts
            if line in ['<!-- image -->', '---', '***', '===']:
                continue

            # Skip very short lines that are likely artifacts
            if len(line) < 3:
                continue

            cleaned_lines.append(line)

        # Remove empty lines at start/end
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()

        return '\n'.join(cleaned_lines)

    def get_references_section(self, sections: Dict[str, DocumentSection]) -> Optional[str]:
        """
        Extract references section for separate processing

        Args:
            sections: Dictionary of detected sections

        Returns:
            References content if found, None otherwise
        """
        if 'references' in sections:
            return sections['references'].content
        return None

    def analyze_document_structure(self, text: str) -> Dict[str, any]:
        """
        Comprehensive analysis of document structure

        Args:
            text: Full document text

        Returns:
            Analysis results including sections, statistics, and quality metrics
        """
        sections = self.detect_sections(text)
        rag_content = self.filter_for_rag(sections)

        analysis = {
            'total_sections': len(sections),
            'detected_sections': list(sections.keys()),
            'has_standard_structure': self._has_standard_medical_structure(sections),
            'rag_content_ratio': self._calculate_rag_content_ratio(rag_content, text),
            'reference_section_size': len(sections.get('references', DocumentSection('', '', 0, 0)).content),
            'quality_score': self._calculate_quality_score(sections),
            'sections': sections,
            'filtered_content_length': len(rag_content),
            'original_content_length': len(text)
        }

        return analysis

    def _has_standard_medical_structure(self, sections: Dict[str, DocumentSection]) -> bool:
        """Check if document has standard medical paper structure"""
        required_sections = {'abstract', 'introduction', 'methods', 'results'}
        detected = set(sections.keys())
        return len(required_sections.intersection(detected)) >= 3

    def _calculate_rag_content_ratio(self, rag_content: str, full_text: str) -> float:
        """Calculate ratio of RAG-relevant content to total content"""
        if not full_text.strip():
            return 0.0
        return len(rag_content) / len(full_text)

    def _calculate_quality_score(self, sections: Dict[str, DocumentSection]) -> float:
        """Calculate overall document quality score for RAG processing"""
        score = 0.0

        # Base score for having sections
        if sections:
            score += 0.3

        # Bonus for standard medical structure
        if self._has_standard_medical_structure(sections):
            score += 0.4

        # Bonus for having references (indicates academic paper)
        if 'references' in sections:
            score += 0.2

        # Penalty for too many unknown sections
        known_sections = set(self.SECTION_PATTERNS.keys())
        unknown_sections = set(sections.keys()) - known_sections
        if len(unknown_sections) > 3:
            score -= 0.1

        return max(0.0, min(1.0, score))


# Convenience functions for integration
def detect_medical_sections(text: str, exclude_references: bool = True) -> Dict[str, DocumentSection]:
    """Convenience function to detect medical document sections"""
    processor = MedicalDocumentProcessor(exclude_references=exclude_references)
    return processor.detect_sections(text)


def filter_medical_content_for_rag(text: str, exclude_references: bool = True) -> str:
    """Convenience function to filter medical content for RAG"""
    processor = MedicalDocumentProcessor(exclude_references=exclude_references)
    sections = processor.detect_sections(text)
    return processor.filter_for_rag(sections)


def analyze_medical_document(text: str) -> Dict[str, any]:
    """Convenience function for comprehensive medical document analysis"""
    processor = MedicalDocumentProcessor()
    return processor.analyze_document_structure(text)


if __name__ == "__main__":
    # Test with sample medical text
    sample_text = """
    Title: A Medical Study on Treatment Efficacy

    Abstract
    This study examines the efficacy of treatment X in patients with condition Y.
    We conducted a randomized controlled trial with 500 participants.

    Introduction
    Condition Y affects millions of patients worldwide and represents a significant
    healthcare challenge. Previous studies have shown limited effectiveness.

    Methods
    We conducted a randomized controlled trial over 12 months.
    Participants were divided into treatment and control groups.

    Results
    Treatment X showed significant improvement in 85% of patients.
    Side effects were minimal and manageable.

    Discussion
    Our findings suggest that treatment X is effective for condition Y.
    These results have important clinical implications.

    Conclusion
    Treatment X represents a promising therapeutic option for patients with condition Y.

    References
    1. Smith et al. (2020). Previous study on condition Y. Nature Medicine, 15(3), 245-250.
    2. Jones et al. (2021). Treatment approaches for Y. NEJM, 384(12), 1123-1130.
    3. Brown et al. (2019). Meta-analysis of treatments. Lancet, 393(10181), 1234-1240.
    """

    processor = MedicalDocumentProcessor()
    sections = processor.detect_sections(sample_text)
    filtered_content = processor.filter_for_rag(sections)
    analysis = processor.analyze_document_structure(sample_text)

    print("=== MEDICAL DOCUMENT ANALYSIS ===")
    print(f"Detected sections: {list(sections.keys())}")
    print(f"Has standard structure: {analysis['has_standard_structure']}")
    print(f"Quality score: {analysis['quality_score']:.2f}")
    print(f"RAG content ratio: {analysis['rag_content_ratio']:.2f}")
    print(f"\nFiltered content length: {len(filtered_content)} chars")
    print(f"Original content length: {len(sample_text)} chars")
    print(f"Reference section found: {'references' in sections}")

    print(f"\n=== FILTERED RAG CONTENT ===")
    print(filtered_content[:500] + "..." if len(filtered_content) > 500 else filtered_content)