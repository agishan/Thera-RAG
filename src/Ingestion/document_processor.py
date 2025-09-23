"""
Document loading utilities using Docling for PDF processing
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
    except ImportError:
        # Fallback for older versions
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
        from docling.datamodel.acceleration_options import AcceleratorDevice
    except ImportError:
        # Fallback for even older versions
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        AcceleratorOptions = None
        AcceleratorDevice = None
except ImportError:
    raise ImportError("Docling is required. Install with: pip install docling")


class DoclingBookLoader:
    """
    Advanced PDF loader using Docling with configurable processing options
    """

    def __init__(
        self,
        file_path: str,
        num_threads: int = 8,
        do_ocr: bool = True,
        do_table_structure: bool = True,
        accelerator_device: str = "auto"
    ):
        """
        Initialize the Docling loader

        Args:
            file_path: Path to the PDF file
            num_threads: Number of threads for processing
            do_ocr: Whether to perform OCR on images
            do_table_structure: Whether to extract table structure
            accelerator_device: Device for acceleration ("auto", "cpu", "gpu")
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        self.converter = self._setup_converter(
            num_threads, do_ocr, do_table_structure, accelerator_device
        )

    def _setup_converter(
        self,
        num_threads: int,
        do_ocr: bool,
        do_table_structure: bool,
        accelerator_device: str
    ) -> DocumentConverter:
        """Setup the Docling document converter with appropriate options"""

        # Handle different Docling versions
        if AcceleratorOptions and AcceleratorDevice:
            # Get accelerator device enum
            if accelerator_device.lower() == "auto":
                device = AcceleratorDevice.AUTO
            elif accelerator_device.lower() == "cpu":
                device = AcceleratorDevice.CPU
            elif accelerator_device.lower() == "gpu":
                device = AcceleratorDevice.GPU
            else:
                device = AcceleratorDevice.AUTO

            accelerator_options = AcceleratorOptions(
                num_threads=num_threads,
                device=device
            )
            pipeline_options = PdfPipelineOptions(
                accelerator_options=accelerator_options,
                do_ocr=do_ocr,
                do_table_structure=do_table_structure,
            )
        else:
            # Simplified options for older versions
            pipeline_options = PdfPipelineOptions(
                do_ocr=do_ocr,
                do_table_structure=do_table_structure,
            )

        # Enable cell matching if available
        if hasattr(pipeline_options, 'table_structure_options'):
            pipeline_options.table_structure_options.do_cell_matching = True

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract_text(self) -> str:
        """
        Extract text from PDF as markdown

        Returns:
            Markdown-formatted text content
        """
        try:
            docling_doc = self.converter.convert(self.file_path).document
            return docling_doc.export_to_markdown()
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {self.file_path}: {e}")

    def extract_structured_content(self) -> Dict[str, Any]:
        """
        Extract structured content including metadata

        Returns:
            Dictionary with text, metadata, and structural information
        """
        try:
            conversion_result = self.converter.convert(self.file_path)
            doc = conversion_result.document

            return {
                "text": doc.export_to_markdown(),
                "raw_document": doc,
                "metadata": {
                    "file_path": str(self.file_path),
                    "file_size": self.file_path.stat().st_size,
                    "pages": len(doc.pages) if hasattr(doc, 'pages') else None,
                    "processing_time": getattr(conversion_result, 'processing_time', None),
                },
                "tables": self._extract_tables(doc),
                "images": self._extract_images(doc)
            }
        except Exception as e:
            raise RuntimeError(f"Failed to extract structured content from {self.file_path}: {e}")

    def _extract_tables(self, doc) -> list:
        """Extract table information if available"""
        tables = []
        if hasattr(doc, 'tables'):
            for table in doc.tables:
                tables.append({
                    "content": str(table),
                    "location": getattr(table, 'bbox', None),
                    "page": getattr(table, 'page', None)
                })
        return tables

    def _extract_images(self, doc) -> list:
        """Extract image information if available"""
        images = []
        if hasattr(doc, 'images'):
            for image in doc.images:
                images.append({
                    "location": getattr(image, 'bbox', None),
                    "page": getattr(image, 'page', None),
                    "type": getattr(image, 'type', None)
                })
        return images


class LoaderFactory:
    """Factory for creating document loaders based on file type"""

    @staticmethod
    def create_loader(file_path: str, **kwargs) -> DoclingBookLoader:
        """
        Create appropriate loader for file type

        Args:
            file_path: Path to document
            **kwargs: Additional arguments for loader

        Returns:
            Document loader instance
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.pdf':
            return DoclingBookLoader(str(file_path), **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    @staticmethod
    def get_supported_formats() -> list:
        """Get list of supported file formats"""
        return ['.pdf']