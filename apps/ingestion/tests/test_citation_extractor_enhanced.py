"""
Comprehensive tests for enhanced citation extractor features

Tests the upgraded citation extraction system with:
- Robust references discovery
- Numbered references parsing
- Enhanced matching with title similarity
- DOI/URL extraction and normalization
- LLM guardrails with JSON validation
- Configurable context windows
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from Ingestion.citation_extractor import (
    CitationExtractor, Citation, CitationMatch, SpanCandidate, ExtractorConfig
)


class TestReferenceDiscovery:
    """Test robust references discovery methods"""

    def setup_method(self):
        self.extractor = CitationExtractor(use_llm=False)

    def test_find_references_with_markdown_header(self):
        """Test finding references with markdown headers"""
        text = """
# Introduction
Some intro text here.

## Methods
Method details.

# References
1. Smith, J. (2023). Test paper. Journal, 1, 1-10.
2. Brown, A. (2022). Another paper. Science, 2, 20-30.

# Appendix
Additional content.
"""
        candidates = self.extractor.find_references_candidates(text)
        assert len(candidates) > 0

        best = self.extractor.select_best_references_span(candidates)
        assert best is not None
        assert "references" in best.header_text.lower()
        assert best.confidence > 0.4

    def test_find_references_without_header(self):
        """Test finding references without markdown headers"""
        text = """
Introduction text here.

REFERENCES

1. Smith, J. (2023). Test paper. Journal of Testing, 45(2), 123-145. doi:10.1234/test
2. Brown, A. et al. (2022). Another study. Science Today, 12, 200-215.
3. Wilson, K. (2021). Third paper. Medical Review, 8(3), 45-60.

Appendix content here.
"""
        candidates = self.extractor.find_references_candidates(text)
        assert len(candidates) > 0

        best = self.extractor.select_best_references_span(candidates)
        assert best is not None
        assert best.confidence > 0.4

    def test_find_references_multiple_candidates(self):
        """Test handling multiple reference section candidates"""
        text = """
Early references mention:
1. Old, A. (2000). Historical paper.

Main content here.

REFERENCES
1. Smith, J. (2023). Current paper. Journal, 1, 1-10.
2. Brown, A. (2022). Recent work. Science, 2, 20-30.
"""
        candidates = self.extractor.find_references_candidates(text)
        assert len(candidates) >= 1

        # Should prefer the section with header and higher density
        best = self.extractor.select_best_references_span(candidates)
        assert best is not None
        # Should pick the real references section, not the early mention
        section_text = text[best.start:best.end]
        assert "Smith, J." in section_text

    def test_reference_density_calculation(self):
        """Test reference density scoring"""
        # High density reference text
        ref_text = """
1. Smith, J. (2023). Title here. Journal Name, 45(2), 123-145. doi:10.1234/example
2. Brown, A. et al. (2022). Another title. Science Today, 12, 200-215.
3. Wilson, K. (2021). Third paper. Medical Review, 8(3), 45-60.
"""
        density = self.extractor._calculate_reference_density(ref_text)
        assert density > 0.7

        # Low density regular text
        regular_text = """
This is just regular paragraph text without any citations or references.
It talks about various topics but doesn't have the structured format.
There are no years in parentheses or numbered lists.
"""
        density = self.extractor._calculate_reference_density(regular_text)
        assert density < 0.3

    @patch('Ingestion.citation_extractor.ChatGoogleGenerativeAI')
    def test_llm_references_discovery(self, mock_llm_class):
        """Test LLM-based references discovery"""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "candidates": [
                {
                    "start_char": 100,
                    "end_char": 300,
                    "confidence": 0.9,
                    "header_text": "REFERENCES"
                }
            ]
        })
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        # Create extractor with LLM
        config = ExtractorConfig(use_llm_for_refs=True)
        extractor = CitationExtractor(llm_api_key="test_key", config=config)

        text = "Some text here. REFERENCES section content."
        candidates = extractor._llm_find_references(text)

        assert len(candidates) == 1
        assert candidates[0].confidence == 0.9
        assert candidates[0].header_text == "REFERENCES"


class TestNumberedReferences:
    """Test numbered references parsing"""

    def setup_method(self):
        self.extractor = CitationExtractor(use_llm=False)

    def test_parse_numbered_list_dot_format(self):
        """Test parsing numbered list with dot format"""
        ref_text = """
1. Smith, J. (2023). First paper. Journal A, 1, 1-10.
2. Brown, A. (2022). Second paper. Journal B, 2, 20-30.
3. Wilson, K. (2021). Third paper. Journal C, 3, 30-40.
"""
        index_map, citations = self.extractor.parse_numbered_references(ref_text)

        assert len(index_map) == 3
        assert len(citations) == 3
        assert 1 in index_map
        assert 2 in index_map
        assert 3 in index_map
        assert "Smith" in index_map[1].authors[0]

    def test_parse_numbered_list_paren_format(self):
        """Test parsing numbered list with parentheses format"""
        ref_text = """
1) Smith, J. (2023). First paper. Journal A, 1, 1-10.
2) Brown, A. (2022). Second paper. Journal B, 2, 20-30.
"""
        index_map, citations = self.extractor.parse_numbered_references(ref_text)

        assert len(index_map) == 2
        assert 1 in index_map
        assert 2 in index_map

    def test_parse_numbered_list_bracket_format(self):
        """Test parsing numbered list with bracket format"""
        ref_text = """
[1] Smith, J. (2023). First paper. Journal A, 1, 1-10.
[2] Brown, A. (2022). Second paper. Journal B, 2, 20-30.
"""
        index_map, citations = self.extractor.parse_numbered_references(ref_text)

        assert len(index_map) == 2
        assert 1 in index_map
        assert 2 in index_map

    def test_resolve_single_numeric_citation(self):
        """Test resolving single numeric citation"""
        index_map = {
            1: Citation("", "Smith et al paper", ["Smith, J."], "2023", "Title"),
            2: Citation("", "Brown et al paper", ["Brown, A."], "2022", "Title 2")
        }

        citations = self.extractor._resolve_numeric_citations("[1]", index_map)
        assert len(citations) == 1
        assert citations[0].authors[0] == "Smith, J."

    def test_resolve_citation_ranges(self):
        """Test resolving citation ranges like [3-5]"""
        numbers = self.extractor._parse_citation_ranges("3-5")
        assert numbers == [3, 4, 5]

        numbers = self.extractor._parse_citation_ranges("3–5")  # en-dash
        assert numbers == [3, 4, 5]

    def test_resolve_citation_list(self):
        """Test resolving citation lists like [1,3,5]"""
        numbers = self.extractor._parse_citation_ranges("1,3,5")
        assert numbers == [1, 3, 5]

        numbers = self.extractor._parse_citation_ranges("1, 3, 5")
        assert numbers == [1, 3, 5]

    def test_duplicate_numbers_fallback(self):
        """Test handling duplicate reference numbers"""
        ref_text = """
1. Smith, J. (2023). First paper.
1. Brown, A. (2022). Second paper with same number.
2. Wilson, K. (2021). Third paper.
"""
        index_map, citations = self.extractor.parse_numbered_references(ref_text)

        # Should handle gracefully - last occurrence wins
        assert len(citations) >= 2
        assert 1 in index_map
        assert 2 in index_map


class TestEnhancedMatching:
    """Test enhanced matching with title similarity"""

    def setup_method(self):
        self.extractor = CitationExtractor(use_llm=False)

    def test_title_similarity_high_match(self):
        """Test high title similarity matching"""
        title1 = "Machine Learning Applications in Medical Diagnosis"
        title2 = "Machine Learning Applications in Medical Diagnosis"

        similarity = self.extractor.title_similarity(title1, title2)
        assert similarity > 0.9

    def test_title_similarity_low_match(self):
        """Test low title similarity matching"""
        title1 = "Machine Learning Applications in Medical Diagnosis"
        title2 = "Quantum Computing for Financial Markets"

        similarity = self.extractor.title_similarity(title1, title2)
        assert similarity < 0.3

    def test_normalize_title_punctuation(self):
        """Test title normalization removes punctuation"""
        title = "Machine Learning: Applications in Medical Diagnosis!"
        normalized = self.extractor.normalize_title(title)

        assert ":" not in normalized
        assert "!" not in normalized
        assert "machine learning applications medical diagnosis" in normalized

    def test_normalize_title_unicode(self):
        """Test title normalization handles unicode"""
        title = "Café-based Learning: A Novel Approach"
        normalized = self.extractor.normalize_title(title)

        # Should handle unicode normalization
        assert "cafe" in normalized.lower() or "café" in normalized.lower()

    def test_doi_exact_match_bonus(self):
        """Test DOI exact match gives bonus score"""
        citation = Citation(
            "", "Test paper", ["Smith, J."], "2023", "Title",
            doi="10.1234/example"
        )

        identifiers = {"doi": "10.1234/example", "year": "2023"}
        score = self.extractor._calculate_match_score(identifiers, citation)

        # Should get high score for DOI match
        assert score > 0.8

    def test_match_score_calculation(self):
        """Test comprehensive match score calculation"""
        citation = Citation(
            "", "Test paper", ["Smith, John", "Brown, Alice"], "2023",
            "Machine Learning in Medicine"
        )

        identifiers = {
            "authors": ["Smith", "Brown"],
            "year": "2023",
            "title": "Machine Learning in Medicine"
        }

        score = self.extractor._calculate_match_score(identifiers, citation)
        assert score > 0.7

    def test_match_threshold_filtering(self):
        """Test match threshold filtering"""
        # This would be tested in integration with the full matching pipeline
        config = ExtractorConfig(match_threshold=0.8)
        extractor = CitationExtractor(use_llm=False, config=config)

        assert extractor.config.match_threshold == 0.8


class TestDoiUrlExtraction:
    """Test DOI/URL extraction and normalization"""

    def setup_method(self):
        self.extractor = CitationExtractor(use_llm=False)

    def test_extract_doi_from_reference(self):
        """Test DOI extraction from reference text"""
        text = "Smith, J. (2023). Paper title. Journal, 1, 1-10. doi:10.1234/example"
        doi = self.extractor._extract_doi_from_text(text)
        assert doi == "10.1234/example"

    def test_extract_url_from_reference(self):
        """Test URL extraction from reference text"""
        text = "Smith, J. (2023). Paper title. Available at: https://example.com/paper"
        url = self.extractor._extract_url_from_text(text)
        assert url == "https://example.com/paper"

    def test_normalize_doi_format(self):
        """Test DOI normalization to standard format"""
        doi = "10.1234/example"
        normalized = self.extractor._normalize_doi(doi)
        assert normalized == "https://doi.org/10.1234/example"

        # Test with doi: prefix
        doi_with_prefix = "doi:10.1234/example"
        normalized = self.extractor._normalize_doi(doi_with_prefix)
        assert normalized == "https://doi.org/10.1234/example"

    def test_prefer_doi_over_url(self):
        """Test that DOI is preferred over URL"""
        citation = Citation(
            "", "Paper with both", ["Smith, J."], "2023", "Title"
        )
        citation.full_reference = "Smith, J. (2023). Title. Journal. doi:10.1234/example https://other.com"

        enhanced = self.extractor.fill_missing_doi_and_url(citation)
        assert enhanced.doi == "10.1234/example"
        assert enhanced.doi_url == "https://doi.org/10.1234/example"


class TestLlmGuardrails:
    """Test LLM guardrails with JSON validation"""

    def setup_method(self):
        self.extractor = CitationExtractor(use_llm=False)

    def test_valid_llm_json_parsing(self):
        """Test valid LLM JSON parsing"""
        valid_data = {
            "authors": ["Smith, J.", "Brown, A."],
            "year": "2023",
            "title": "Test Paper"
        }

        is_valid = self.extractor._validate_citation_json(valid_data)
        assert is_valid

    def test_invalid_llm_json_fallback(self):
        """Test fallback when LLM JSON is invalid"""
        invalid_data = {
            "authors": ["Smith, J."],
            # Missing required 'year' and 'title'
        }

        is_valid = self.extractor._validate_citation_json(invalid_data)
        assert not is_valid

    def test_missing_required_fields_fallback(self):
        """Test fallback when required fields are missing"""
        incomplete_data = {
            "authors": ["Smith, J."],
            "year": "2023"
            # Missing required 'title'
        }

        is_valid = self.extractor._validate_citation_json(incomplete_data)
        assert not is_valid

    def test_fallback_regex_parse(self):
        """Test regex fallback parsing"""
        raw_line = "Smith, J. (2023). Test title. Journal Name, 45(2), 123-145."
        citation = self.extractor._fallback_regex_parse(raw_line)

        assert citation is not None
        assert "Smith" in citation.authors[0]
        assert citation.year == "2023"


class TestContextWindows:
    """Test configurable context windows"""

    def setup_method(self):
        config = ExtractorConfig(
            sentence_aware=True,
            context_before_chars=50,
            context_after_chars=50
        )
        self.extractor = CitationExtractor(use_llm=False, config=config)

    def test_sentence_aware_context(self):
        """Test sentence-aware context extraction"""
        text = "First sentence here. Second sentence with citation [1] in the middle. Third sentence after."
        match_pos = text.find("[1]")

        before, after = self.extractor._get_sentence_context(text, match_pos)

        assert "Second sentence" in before or "First sentence" in before
        assert "Third sentence" in after

    def test_char_context_fallback(self):
        """Test character context fallback"""
        text = "Some text with citation [1] embedded here."
        match_pos = text.find("[1]")

        before, after = self.extractor._get_char_context(text, match_pos)

        assert len(before) <= self.extractor.config.context_before_chars
        assert len(after) <= self.extractor.config.context_after_chars

    def test_configurable_context_sizes(self):
        """Test different context sizes"""
        config_small = ExtractorConfig(context_before_chars=10, context_after_chars=10)
        extractor_small = CitationExtractor(use_llm=False, config=config_small)

        text = "This is a longer text with citation [1] and more content after it."
        match_pos = text.find("[1]")

        before, after = extractor_small._get_char_context(text, match_pos)

        assert len(before) <= 10
        assert len(after) <= 10


class TestIntegration:
    """Integration tests for the complete pipeline"""

    def setup_method(self):
        self.extractor = CitationExtractor(use_llm=False)

    def test_end_to_end_processing(self):
        """Test complete end-to-end citation extraction"""
        document_text = """
# Introduction
This paper discusses machine learning applications.

# Methods
We used various algorithms as described by Smith (2023) and Brown et al. (2022).

# References
1. Smith, J. (2023). Machine Learning Basics. Journal of AI, 45(2), 123-145. doi:10.1234/ml2023
2. Brown, A., Wilson, K., & Davis, M. (2022). Advanced Algorithms. Computer Science Review, 12, 200-215.
"""

        from langchain_core.documents import Document
        chunks = [
            Document(
                page_content="We used various algorithms as described by Smith (2023) and Brown et al. (2022).",
                metadata={"chunk_id": "chunk_0"}
            )
        ]

        # Run full extraction
        citations, matches = self.extractor.extract_citations_from_document(document_text, chunks)

        # Verify we found citations and matches
        assert len(citations) >= 1
        assert any("Smith" in str(c.authors) for c in citations)

    def test_backward_compatibility(self):
        """Test that new features don't break existing API"""
        # Should work with old-style initialization
        extractor = CitationExtractor(llm_api_key=None, use_llm=False)

        # Should have default config
        assert extractor.config is not None
        assert extractor.config.match_threshold == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])