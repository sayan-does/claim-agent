# Medical Claim Processing Pipeline



- Multi-agent orchestration for document classification, extraction, validation, and decision-making

- Modular, testable, and async FastAPI backend
- Real-world PDF extraction (including OCR)
- LLM-driven data structuring and validation
- Vector store (FAISS) for semantic document chunking and retrieval

## Technology Choices & Rationale

### 1. OpenRouter GPT-4o (LLM)

- **Why:** Chosen for its state-of-the-art performance in text classification, extraction, and reasoning. OpenRouter provides reliable API access to GPT-4o, enabling advanced prompt engineering and robust handling of complex, real-world medical documents.
- **Benefit:** Maximizes accuracy and flexibility for all LLM-driven agent steps (classification, extraction, validation, decision).

### 2. PyPDF2 & OCR (pytesseract + pdf2image)

- **Why:** Medical PDFs are often a mix of digital and scanned content. PyPDF2 extracts digital text, while pytesseract (with pdf2image and Poppler) enables OCR for scanned/image-based pages.
- **Benefit:** Ensures comprehensive text extraction, so no critical information is missed regardless of PDF format.

### 3. LangChain & FAISS

- **Why:** LangChain provides robust document chunking and integration with vector databases. FAISS is a high-performance vector search library for semantic retrieval.
- **Benefit:** Enables efficient, context-aware retrieval of relevant document sections for LLM prompting, improving extraction and classification quality.

### 4. FastAPI

- **Why:** Modern, async-ready Python web framework with automatic OpenAPI docs and strong type support.
- **Benefit:** Ensures scalable, maintainable, and well-documented API endpoints for claim processing.

### 5. Pydantic

- **Why:** Type-safe, robust data validation and serialization for Python models.
- **Benefit:** Guarantees structured, validated data throughout the pipeline, reducing errors and improving reliability.

### 6. Modular Python Architecture

- **Why:** Separating PDF extraction, vector storage, and agent logic into distinct modules improves maintainability, testability, and extensibility.
- **Benefit:** Makes it easy to update, test, or swap out components as technology evolves or requirements change.

---

Each technology was chosen to maximize reliability, accuracy, and extensibility for real-world medical claim automation, ensuring the pipeline is robust for both current and future needs.

## Overview

This project is a modular, production-ready pipeline for automated medical insurance claim document processing. It leverages state-of-the-art AI (OpenRouter GPT-4o), robust PDF text extraction (including OCR), and vector search (FAISS) to classify, extract, validate, and make decisions on medical claim documents. The system is designed for extensibility, transparency, and high accuracy in real-world insurance workflows.

## Features

- **LLM-Driven Processing:** Uses OpenRouter GPT-4o for all classification, extraction, and validation tasks.
- **Advanced PDF Text Extraction:** Combines PyPDF2 and OCR (pytesseract/pdf2image) to extract text from both digital and scanned PDFs.
- **Vector Search with FAISS:** Chunks and stores document text for efficient semantic retrieval and context-aware LLM prompting.
- **Modular Architecture:**
  - `pdf_text_extractor.py`: Handles all PDF text extraction logic.
  - `faiss_store.py`: Manages FAISS vectorstore creation and retrieval.
  - `agents.py`: Contains all LLM agent logic (classification, extraction, validation, decision).
- **Structured Data Extraction:** All structured data is extracted by the LLM—no hardcoded or fake data.
- **Per-Claim JSON Output:** Each processed claim is saved as a JSON file in `claim_jsons/` for traceability and audit.
- **Comprehensive Testing:** Includes tests for each module, with LLM responses printed for transparency.

## Project Structure

```
claim-agent/
├── agents.py                # LLM agent logic (classification, extraction, validation, decision)
├── faiss_store.py           # FAISS vectorstore build/search logic
├── main.py                  # FastAPI app for claim processing
├── models.py                # Pydantic models for structured data
├── pdf_text_extractor.py    # PDF text extraction (PyPDF2 + OCR)
├── requirements.txt         # All dependencies
├── test.py                  # Expanded test suite
├── claim_jsons/             # Output folder for per-claim JSONs
├── documents/               # Input folder for test PDFs
└── ...
```

## How It Works

1. **Upload PDFs:** User uploads one or more PDF files (bills, discharge summaries, ID cards, etc.).
2. **Text Extraction:** Each PDF is processed with PyPDF2 and OCR to extract all possible text.
3. **Classification:** The LLM classifies each document (bill, discharge summary, id card, other).
4. **Data Extraction:** The LLM extracts all relevant structured data from the text.
5. **Validation:** Extracted data is validated for completeness and correctness.
6. **Decision:** The LLM (or rule-based agent) makes a claim decision (approve, reject, review).
7. **Output:** Results are returned via API and saved as JSON files for each claim.

## Setup Instructions

1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Install Poppler** (required for OCR):
   - **Windows:** Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/), extract, and add the `bin` folder to your PATH.
   - **Linux/macOS:** Install via your package manager (e.g., `sudo apt install poppler-utils`).
4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and add your OpenRouter API key and model info.
5. **Add sample PDFs** to the `documents/` folder for testing.

## Running the API

Start the FastAPI server:

```sh
uvicorn main:app --reload
```

Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API documentation.

## Testing

Run the test suite (includes PDF extraction, FAISS, and LLM agent tests):

```sh
pytest -v -s
```

You can also run `pdf_text_extractor.py` and `faiss_store.py` directly for module-level tests.

## Troubleshooting

- **Poppler errors:** Ensure Poppler is installed and its `bin` directory is in your system PATH.
- **OCR quality:** The pipeline applies image preprocessing for best OCR results, but quality depends on scan clarity.
- **LLM API issues:** Ensure your OpenRouter API key is valid and you have internet access.

## Extending the Pipeline

- Replace `FakeEmbeddings` with real embeddings for production-grade semantic search.
- Add more document types or extraction fields by updating the LLM prompts and models.
- Integrate with downstream claim management systems as needed.

## License

This project is provided for demonstration and research purposes. For production use, review all dependencies and ensure compliance with your organization's data privacy and security policies.
