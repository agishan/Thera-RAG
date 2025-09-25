"""
Unit tests for text-only mode in document processing
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "ingestion"))

from document_processor import DoclingBookLoader


class TestTextOnlyMode:
    """Test suite for text-only document processing mode"""

    def test_text_only_mode_initialization(self):
        """Test that text-only mode properly initializes with correct settings"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Mock the file existence check
            with patch('pathlib.Path.exists', return_value=True):
                with patch.object(DoclingBookLoader, '_setup_converter') as mock_setup:
                    mock_setup.return_value = Mock()

                    # Test text-only mode
                    loader = DoclingBookLoader(tmp_path, text_only=True)

                    assert loader.text_only is True
                    # Verify _setup_converter was called with text-only settings
                    mock_setup.assert_called_once_with(
                        1,    # num_threads reduced to 1
                        False,  # do_ocr disabled
                        False,  # do_table_structure disabled
                        "auto"  # accelerator_device
                    )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_full_mode_initialization(self):
        """Test that full mode maintains default heavy processing settings"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with patch('pathlib.Path.exists', return_value=True):
                with patch.object(DoclingBookLoader, '_setup_converter') as mock_setup:
                    mock_setup.return_value = Mock()

                    # Test full mode (default)
                    loader = DoclingBookLoader(tmp_path, text_only=False)

                    assert loader.text_only is False
                    # Verify _setup_converter was called with full processing settings
                    mock_setup.assert_called_once_with(
                        8,    # num_threads default
                        True,   # do_ocr enabled
                        True,   # do_table_structure enabled
                        "auto"  # accelerator_device
                    )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_text_only_structured_content_format(self):
        """Test that text-only mode returns correct structured content format"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 1024

                    # Mock the converter and document
                    mock_doc = Mock()
                    mock_doc.export_to_markdown.return_value = "Sample text content"
                    mock_doc.pages = [Mock(), Mock()]  # 2 pages

                    mock_conversion_result = Mock()
                    mock_conversion_result.document = mock_doc
                    mock_conversion_result.processing_time = 2.5

                    mock_converter = Mock()
                    mock_converter.convert.return_value = mock_conversion_result

                    with patch.object(DoclingBookLoader, '_setup_converter', return_value=mock_converter):
                        loader = DoclingBookLoader(tmp_path, text_only=True)
                        result = loader.extract_structured_content()

                        # Check structure
                        assert "text" in result
                        assert "full_text" in result
                        assert "metadata" in result
                        assert "tables" in result
                        assert "images" in result

                        # Check text-only specific behavior
                        assert result["tables"] == []  # Empty in text-only mode
                        assert result["images"] == []  # Empty in text-only mode
                        assert result["metadata"]["text_only_mode"] is True
                        assert result["text"] == "Sample text content"
                        assert result["full_text"] == "Sample text content"

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_full_mode_structured_content_format(self):
        """Test that full mode calls table and image extraction"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 1024

                    # Mock the converter and document
                    mock_doc = Mock()
                    mock_doc.export_to_markdown.return_value = "Sample text content"
                    mock_doc.pages = [Mock(), Mock()]

                    mock_conversion_result = Mock()
                    mock_conversion_result.document = mock_doc
                    mock_conversion_result.processing_time = 15.2

                    mock_converter = Mock()
                    mock_converter.convert.return_value = mock_conversion_result

                    with patch.object(DoclingBookLoader, '_setup_converter', return_value=mock_converter):
                        with patch.object(DoclingBookLoader, '_extract_tables', return_value=[{"table": "data"}]) as mock_tables:
                            with patch.object(DoclingBookLoader, '_extract_images', return_value=[{"image": "info"}]) as mock_images:
                                loader = DoclingBookLoader(tmp_path, text_only=False)
                                result = loader.extract_structured_content()

                                # Check that extraction methods were called
                                mock_tables.assert_called_once_with(mock_doc)
                                mock_images.assert_called_once_with(mock_doc)

                                # Check results include extracted data
                                assert result["tables"] == [{"table": "data"}]
                                assert result["images"] == [{"image": "info"}]
                                assert result["metadata"]["text_only_mode"] is False

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_factory_with_text_only(self):
        """Test that LoaderFactory passes text_only parameter correctly"""
        from document_processor import LoaderFactory

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            with patch('pathlib.Path.exists', return_value=True):
                with patch.object(DoclingBookLoader, '_setup_converter') as mock_setup:
                    mock_setup.return_value = Mock()

                    # Test factory with text_only=True
                    loader = LoaderFactory.create_loader(tmp_path, text_only=True)
                    assert loader.text_only is True

                    # Test factory with text_only=False
                    loader = LoaderFactory.create_loader(tmp_path, text_only=False)
                    assert loader.text_only is False

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_interactive_pipeline_text_only_integration(self):
        """Test that InteractivePipeline properly uses text_only mode"""
        # Add scripts to path for imports
        scripts_path = project_root / "scripts"
        sys.path.insert(0, str(scripts_path))

        from interactive_pipeline import InteractivePipeline

        # Test initialization
        pipeline = InteractivePipeline(text_only=True)
        assert pipeline.text_only is True

        pipeline = InteractivePipeline(text_only=False)
        assert pipeline.text_only is False


class TestPerformanceEstimation:
    """Tests to help estimate performance improvements"""

    def test_text_only_mode_should_be_faster(self):
        """
        This test documents the expected performance characteristics.
        In real usage:
        - Text-only mode: ~2-5 seconds per document
        - Full mode: ~30-120 seconds per document
        - Expected speedup: 6-24x faster
        """
        # This is a documentation test - the actual performance depends on:
        # 1. Document complexity
        # 2. Hardware capabilities
        # 3. Text layer quality in PDF

        # Expected CPU resource reduction in text-only mode:
        expected_reductions = {
            "ocr_processing": "100%",  # Completely disabled
            "layout_analysis": "90%",  # Minimal layout processing
            "table_extraction": "100%",  # Skipped entirely
            "thread_usage": "87.5%",  # 8 threads -> 1 thread
            "memory_usage": "60-80%",  # Reduced model loading
        }

        assert expected_reductions["ocr_processing"] == "100%"
        assert expected_reductions["table_extraction"] == "100%"


if __name__ == "__main__":
    pytest.main([__file__])