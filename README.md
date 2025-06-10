# Thera-RAG: Therapy Assistant with RAG and Clinical Document QA

## Overview
Thera-RAG is an interactive, document-grounded question-answering (QA) and retrieval-augmented generation (RAG) system for clinical and medical research. It enables users to query ingested clinical guidelines, research papers, and other documents using natural language, leveraging state-of-the-art large language models (LLMs) and vector search.

The project is designed for use cases such as:
- Clinical decision support
- Medical research and literature review
- Rapid Q&A over guidelines and protocols
- Educational and training purposes

Thera-RAG is built with Streamlit for a modern, chat-based UI, and integrates Google Gemini 1.5 Pro, Pinecone vector database, and Google Sheets for optional logging.

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

## Architecture
```
PDFs/Guidelines → Ingestion Pipeline (Docling, SentenceTransformers) → Pinecone Vector DB
         ↓
   Streamlit App (src/app/main.py)
         ↓
   RAG Service (LangChain, Gemini 1.5 Pro, Pinecone)
         ↓
   User Chat Interface + (Optional) Google Sheets Logging
```
- **Ingestion**: `src/Ingestion/chunking_embedding.ipynb` processes PDFs, chunks text, and uploads embeddings to Pinecone.
- **App**: `src/app/main.py` runs the Streamlit chat UI and orchestrates RAG, config, and logging.
- **RAG**: `src/app/rag_service.py` handles retrieval and LLM-based answer generation.

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

## Usage
1. **Ingest documents (first time or when adding new PDFs):**
   - Place PDFs in `src/Ingestion/pdfs/`
   - Run the ingestion pipeline:
     ```bash
     cd src/Ingestion
     jupyter notebook chunking_embedding.ipynb
     # or run as a script if converted
     ```
   - This will extract, chunk, and embed content, uploading vectors to Pinecone.

2. **Run the Streamlit app:**
   ```bash
   cd src/app
   streamlit run main.py
   ```
   - Open the provided local URL in your browser.
   - Enter your API keys if prompted.
   - Start chatting! Ask questions about the ingested documents.

---

## Configuration
- All configuration is handled via environment variables or Streamlit secrets.
- See `src/app/config.py` for all options (model, Pinecone index, namespace, retrieval parameters, etc).
- Google Sheets logging is optional and will be enabled if credentials are provided.

---

## Dependencies
- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [Pandas, Numpy](https://pandas.pydata.org/)
- [LangChain](https://python.langchain.com/)
- [Google Generative AI](https://ai.google.dev/)
- [Pinecone](https://www.pinecone.io/)
- [SentenceTransformers](https://www.sbert.net/) (for ingestion)
- [Docling](https://github.com/docling-ai/docling) (for PDF parsing)
- [Google Sheets API](https://developers.google.com/sheets/api) (optional)

See `requirements.txt` for the full list.

---

## Optional: Google Sheets Logging
- If enabled, all Q&A interactions are logged to a specified Google Sheet for later analysis or audit.
- Requires a Google service account and spreadsheet setup.
- See `src/app/sheets_service.py` for details.

---

## License
This project is for research and educational use. For clinical or production deployment, review all dependencies, data privacy, and regulatory requirements.

---

## Acknowledgements
- British Society for Haematology for clinical guideline content.
- Open-source contributors to Streamlit, LangChain, Pinecone, and Docling. 