import os
import httpx
import random
import PyPDF2
from models import BillData, DischargeSummaryData, IDCardData, OtherDocumentData, ValidationResult, ClaimDecision
from datetime import datetime
import io
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from pdf_text_extractor import extract_text_from_pdf
from faiss_store import store_text_in_faiss, retrieve_relevant_chunk

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

SYSTEM_PROMPT = (
    "You are a highly reliable, detail-oriented assistant for medical insurance claim document processing. "
    "Your job is to help classify, extract, and validate information from uploaded medical documents. "
    "You must always follow these rules:\n"
    "- Only respond with the requested information, in the format specified.\n"
    "- If asked to classify, choose only from: bill, discharge_summary, id_card, other.\n"
    "- If extracting data, be precise and never fabricate information.\n"
    "- If information is missing or unclear, state so explicitly.\n"
    "- Never include any patient data in your response except as found in the document.\n"
    "- If you are unsure, respond with 'other' or 'unknown'.\n"
    "- Be concise, accurate, and avoid speculation.\n"
    "- Do not provide explanations unless explicitly asked.\n"
    "- Always use the most up-to-date medical and insurance terminology.\n"
    "- When extracting or structuring data, always use the following JSON format as a reference for your output, including only the fields relevant to the document type:\n"
    "  For a bill: {\\\"type\\\": \\\"bill\\\", \\\"hospital_name\\\": ..., \\\"total_amount\\\": ..., \\\"date_of_service\\\": ...}\n"
    "  For a discharge summary: {\\\"type\\\": \\\"discharge_summary\\\", \\\"patient_name\\\": ..., \\\"diagnosis\\\": ..., \\\"admission_date\\\": ..., \\\"discharge_date\\\": ...}\n"
    "  For an id_card: {\\\"type\\\": \\\"id_card\\\", \\\"patient_name\\\": ..., \\\"patient_id\\\": ..., \\\"insurance_provider\\\": ..., \\\"policy_number\\\": ..., \\\"validity_date\\\": ...}\n"
    "  For other: {\\\"type\\\": \\\"other\\\", \\\"content_summary\\\": ...}\n"
    "- Do not include extra fields or explanations in your JSON output.\n"
)


async def classify_document_agent(text: str) -> str:
    """Classify document type using OpenRouter GPT-4o, with PDF content indexed by LangChain."""
    if not OPENROUTER_API_KEY:
        return "other"
    # Use the new FAISS store module
    vectorstore = store_text_in_faiss(text, embedding_size=1536)
    indexed_text = retrieve_relevant_chunk(
        vectorstore, text, k=1) or text[:2000]
    system_prompt = SYSTEM_PROMPT + \
        f"\n\n--- Indexed PDF Content ---\n{indexed_text}\n--- End ---"
    prompt = (
        "Classify the above medical document as one of: bill, discharge_summary, id_card, other. "
        "Respond with only the type."
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 10,
        "temperature": 0
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip(
            ).lower()
            if content in ["bill", "discharge_summary", "id_card", "other"]:
                return content
            return "other"
    except Exception:
        return "other"


async def extract_data_agent(text: str, doc_type: str):
    """Extract structured data from the PDF text using the LLM only."""
    if not OPENROUTER_API_KEY:
        # Fallback: return minimal data if no LLM available
        return OtherDocumentData(document_title=None, content_summary=text[:100])
    # Build a prompt for extraction based on doc_type
    if doc_type == "bill":
        user_prompt = (
            "Extract ONLY the following fields from the medical bill document below as a JSON object. Use this format (replace values with those from the document): "
            '{"type": "bill", "hospital_name": "HOSPITAL_NAME", "total_amount": 12345, "date_of_service": "2024-04-10"}'
            " If a field is missing, use null. Do not include extra fields or explanations.\n\nDocument:\n" +
            text[:2000]
        )
    elif doc_type == "discharge_summary":
        user_prompt = (
            "Extract ONLY the following fields from the discharge summary below as a JSON object. Use this format (replace values with those from the document): "
            '{"type": "discharge_summary", "patient_name": "PATIENT_NAME", "diagnosis": "DIAGNOSIS", "admission_date": "2024-04-01", "discharge_date": "2024-04-10"}'
            " If a field is missing, use null. Do not include extra fields or explanations.\n\nDocument:\n" +
            text[:2000]
        )
    elif doc_type == "id_card":
        user_prompt = (
            "Extract ONLY the following fields from the insurance ID card below as a JSON object. Use this format (replace values with those from the document): "
            '{"type": "id_card", "patient_name": "PATIENT_NAME", "patient_id": "ID", "insurance_provider": "PROVIDER", "policy_number": "POLICY", "validity_date": "2024-12-31"}'
            " If a field is missing, use null. Do not include extra fields or explanations.\n\nDocument:\n" +
            text[:2000]
        )
    else:
        user_prompt = (
            "Summarize the content of the following document in at least 100 words. Respond with a JSON object: {\"type\": \"other\", \"content_summary\": \"SUMMARY\"}. Do not include extra fields or explanations.\n\nDocument:\n" +
            text[:2000]
        )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 512,
        "temperature": 0
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            import json as _json
            if doc_type == "bill":
                parsed = _json.loads(content)
                return BillData(**parsed)
            elif doc_type == "discharge_summary":
                parsed = _json.loads(content)
                return DischargeSummaryData(**parsed)
            elif doc_type == "id_card":
                parsed = _json.loads(content)
                return IDCardData(**parsed)
            else:
                return OtherDocumentData(document_title=None, content_summary=content)
    except Exception as e:
        # Fallback: return minimal data if LLM or parsing fails
        return OtherDocumentData(document_title=None, content_summary=text[:100])


async def validate_claim_agent(extracted_data):
    """Validation agent: checks for required fields and values."""
    results = []
    for item in extracted_data:
        errors = []
        warnings = []
        # Bill validation
        if hasattr(item, 'type') and getattr(item, 'type', None) == 'bill':
            if getattr(item, 'total_amount', 0) <= 0:
                errors.append("Total amount must be greater than 0.")
            if getattr(item, 'patient_name', '').strip().lower() in ["", "tbd", "unknown"]:
                errors.append("Patient name is missing or invalid.")
        # Add more rules for other types if needed
        is_valid = len(errors) == 0
        results.append(ValidationResult(is_valid=is_valid,
                       errors=errors, warnings=warnings))
    return results


async def decide_claim_agent(extracted_data, validation_results):
    """Fake claim decision for testing."""
    return ClaimDecision(
        decision="approved",
        reason="Test approval",
        confidence_score=1.0,
        extracted_data=extracted_data,
        validation_results=validation_results
    )
