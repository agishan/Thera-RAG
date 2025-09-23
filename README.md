# Citation-Aware Medical RAG

> **The Problem**: Standard RAG systems lose citation context in medical documents
> **Our Solution**: LLM-powered citation extraction that links inline citations to full bibliography entries

## What Makes This Special

Medical and academic documents have a **critical citation attribution problem** in RAG systems:

❌ **Before**: Inline citations `(Smith et al., 2023)` get separated from their full references
❌ **Before**: Content gets truncated (500-char limits)
❌ **Before**: Source attribution becomes vague

✅ **After**: Complete citation linking with LLM accuracy
✅ **After**: Full content preservation in vectors
✅ **After**: Rich source metadata for every chunk

## Core Innovation: LLM Citation Extraction

```
PDF → LLM Parser → "Smith et al. (2023) found that..." + Bibliography Entry
                                ↓
                    Enhanced Vector Chunk with Full Citation Metadata
```

**Key Features:**
- 🧠 **Google Gemini + Regex Hybrid**: Accurate parsing of complex citation formats
- 🔗 **Inline-to-Bibliography Linking**: Connects `(Smith, 2023)` to full references
- 📚 **Complete Citation Metadata**: Authors, year, title, journal, DOI preserved
- 💾 **No Content Truncation**: Full text stored in vectors
- ⚡ **Clean Architecture**: Focused on citation quality, no over-engineering

**Perfect For:**
- Medical research requiring source verification
- Clinical decision support with traceable evidence
- Academic literature review with proper attribution
- Regulatory environments requiring citation tracking

---

## Clinical Context: Viscoelastic Haemostatic Assays (VHA)
This project was initially developed to support the interpretation and application of clinical guidelines on viscoelastic haemostatic assays (VHA) such as TEG, ROTEM, and Sonoclot, which are used in the management of major bleeding in trauma, surgery, and obstetrics. These assays provide rapid, point-of-care assessment of coagulation and are increasingly used to guide transfusion and haemostatic therapy.

The system can ingest and answer questions about guidelines, such as the British Society for Haematology's recommendations on VHA use in major haemorrhage, liver transplantation, cardiac surgery, and trauma.

---

## Features
- **Chat-based QA**: Ask questions about ingested documents and receive answers with source references.
- **Retrieval-Augmented Generation (RAG)**: Combines vector search (Pinecone) with LLM (Gemini 1.5 Pro) for accurate, context-aware answers.
- **Document Ingestion Pipeline**: Process PDFs into structured, chunked, and embedded content for efficient retrieval.
- **Enhanced Content Rendering**: Tables and structured data are rendered interactively in the UI.
- **Session Management**: Track chat history and session IDs.
- **Optional Google Sheets Logging**: Log Q&A sessions for audit or research purposes.

---

## Citation-Focused Architecture

```
PDFs → LLM Citation Extraction → Enhanced Chunks → Pinecone Vector DB
         ↓                           ↓
   (Gemini parses citations)    (Full content + citations)
         ↓                           ↓
   Streamlit Chat UI  ←  Citation-Aware RAG Service
```

**Key Components:**
- **Citation Extractor**: LLM + regex hybrid for parsing inline citations and bibliography
- **Enhanced Chunker**: Preserves full content without character limits
- **Simple Pipeline**: Clean ingestion focused on citation quality
- **Citation-Aware RAG**: Retrieval with rich source metadata

---

## Setup & Installation
1. **Clone the repository**
2. **Install Python 3.9+ and [pip](https://pip.pypa.io/en/stable/)**
3. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Set up environment variables:**
   - Create a `.env` file or use Streamlit secrets for sensitive keys.
   - Required:
     - `PINECONE_API_KEY` (for Pinecone vector DB)
     - `GOOGLE_API_KEY` (for Gemini LLM)
   - Optional (for Google Sheets logging):
     - `GOOGLE_SHEETS_SPREADSHEET_ID`
     - `GOOGLE_SHEETS_CREDS_JSON` (service account JSON)
   - Other config variables can be set as needed (see `src/app/config.py`).

---

## Quick Start (3 Steps)

### 1. Environment Setup
```bash
# Required: LLM citation extraction (the core feature)
export GOOGLE_API_KEY="your-google-api-key"

# Optional: Vector storage
export PINECONE_API_KEY="your-pinecone-key"

# Install dependencies
pip install -r requirements.txt
```

### 2. Ingest Medical Documents
```bash
# Add your PDFs
mkdir data && cp your-papers.pdf data/

# Run citation extraction pipeline
python run_ingestion.py
```

**Processing Pipeline:**
1. 📄 Extract text from PDFs (Docling)
2. ✂️ Create intelligent chunks with overlap
3. 🧠 **LLM citation extraction** (inline + bibliography parsing)
4. 🔗 Link inline citations to full references
5. 💾 Save enhanced chunks with complete metadata
6. ⬆️ Upload to Pinecone (optional)

### 3. Query with Citation Context
```bash
# Start citation-aware RAG interface
cd src/app && streamlit run main.py
```

**Enhanced Retrieval:**
- Rich citation metadata in every response
- Full source attribution with clickable references
- Complete content context (no truncation)
- Traceable evidence chains for medical decisions

---

## Example: Citation Extraction in Action

### Input Document
```
Recent studies have shown promising results (Smith et al., 2023; Jones, 2022).
According to Wilson and Brown (2024), the treatment approach...

References:
1. Smith, J., Davis, M., Lee, K. (2023). "Advanced Treatment Protocols in
   Emergency Medicine." Journal of Emergency Medicine, 45(3), 234-245.
   DOI: 10.1016/j.jemermed.2023.03.001

2. Jones, A. (2022). "Clinical Guidelines for Rapid Response."
   Medical Procedures Quarterly, 12(4), 89-102.
```

### Enhanced Vector Chunk Output
```json
{
  "content": "Recent studies have shown promising results (Smith et al., 2023; Jones, 2022). According to Wilson and Brown (2024), the treatment approach...",
  "metadata": {
    "citations": [
      {
        "inline_text": "(Smith et al., 2023)",
        "full_reference": "Smith, J., Davis, M., Lee, K. (2023). Advanced Treatment Protocols in Emergency Medicine. Journal of Emergency Medicine, 45(3), 234-245.",
        "authors": ["Smith, J.", "Davis, M.", "Lee, K."],
        "year": "2023",
        "title": "Advanced Treatment Protocols in Emergency Medicine",
        "journal": "Journal of Emergency Medicine",
        "doi": "10.1016/j.jemermed.2023.03.001"
      }
    ],
    "text": "FULL CONTENT PRESERVED - NO TRUNCATION",
    "chunk_size": 2847,
    "citation_count": 2
  }
}
```

## 📚 Complete Documentation

### Core Documentation
- **[`README.md`](README.md)** - This file (overview & quick start)
- **[`SYSTEM_ANALYSIS.md`](SYSTEM_ANALYSIS.md)** - Ultra-deep technical analysis
- **[`PROJECT_CLEANUP_PLAN.md`](PROJECT_CLEANUP_PLAN.md)** - Comprehensive cleanup strategy

### Technical Guides
- **[`run_ingestion.py`](run_ingestion.py)** - Citation-focused ingestion pipeline
- **[`verify_output_quality.py`](verify_output_quality.py)** - Quality verification & metrics
- **[`.env.example`](.env.example)** - Environment configuration template

### Examples & Output
- **[`enhanced_output/`](enhanced_output/)** - Sample enhanced chunks with citation metadata
- **[`data/`](data/)** - Directory for your PDF files

### Architecture Deep Dive
- **[`src/ingestion/citation_extractor.py`](src/ingestion/citation_extractor.py)** - The core LLM citation system
- **[`src/ingestion/pipeline.py`](src/ingestion/pipeline.py)** - Clean ingestion pipeline
- **[`src/app/enhanced_retriever.py`](src/app/enhanced_retriever.py)** - Citation-aware retrieval
- **[`src/app/rag_service.py`](src/app/rag_service.py)** - RAG orchestration

## System Requirements

- **Python 3.9+**
- **Google API Key** (for LLM citation extraction)
- **Pinecone API Key** (optional, for vector storage)

**Key Dependencies:**
- `google-generativeai` - LLM citation parsing
- `langchain` - RAG orchestration
- `pinecone-client` - Vector storage
- `docling` - PDF text extraction
- `streamlit` - Web interface

## Use Cases

✅ **Medical Research**: Verify source claims with full citation trails
✅ **Clinical Guidelines**: Track evidence sources for treatment protocols
✅ **Literature Review**: Maintain proper academic attribution
✅ **Regulatory Compliance**: Auditable source documentation
✅ **Educational Content**: Teach with verifiable medical sources

## License & Usage

For research and educational use. Medical/clinical deployment requires compliance review.

**🎯 The Bottom Line**: This system makes medical document RAG **citation-aware**, solving the critical source attribution problem in healthcare AI applications.