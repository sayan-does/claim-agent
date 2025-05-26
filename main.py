# main.py
from typing import List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException

from models import ClaimDecision
from agents import (
    extract_text_from_pdf,
    classify_document_agent,
    extract_data_agent,
    validate_claim_agent,
    decide_claim_agent
)

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Claim Processor", version="1.0.0")


@app.post("/process-claim")
async def process_claim(files: List[UploadFile] = File(...)):
    """
    Main endpoint to process claim with multiple PDF files
    Returns output in the preferred JSON format.
    """
    if not files:
        raise HTTPException(
            status_code=400, detail="At least one file is required")
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, detail=f"Only PDF files are supported. Got: {file.filename}")
    try:
        documents = []
        doc_types_found = set()
        for file in files:
            text = await extract_text_from_pdf(file)
            if not text:
                continue
            doc_type = await classify_document_agent(text)
            doc_types_found.add(doc_type)
            extracted = await extract_data_agent(text, doc_type)
            # Convert extracted object to dict and filter only required fields
            if doc_type == "bill":
                doc_dict = {
                    "type": "bill",
                    "hospital_name": getattr(extracted, "hospital_name", None),
                    "total_amount": getattr(extracted, "total_amount", None),
                    "date_of_service": str(getattr(extracted, "bill_date", None))[:10] if getattr(extracted, "bill_date", None) else None
                }
            elif doc_type == "discharge_summary":
                doc_dict = {
                    "type": "discharge_summary",
                    "patient_name": getattr(extracted, "patient_name", None),
                    "diagnosis": getattr(extracted, "diagnosis", None),
                    "admission_date": str(getattr(extracted, "admission_date", None))[:10] if getattr(extracted, "admission_date", None) else None,
                    "discharge_date": str(getattr(extracted, "discharge_date", None))[:10] if getattr(extracted, "discharge_date", None) else None
                }
            elif doc_type == "id_card":
                doc_dict = {
                    "type": "id_card",
                    "patient_name": getattr(extracted, "patient_name", None),
                    "patient_id": getattr(extracted, "patient_id", None),
                    "insurance_provider": getattr(extracted, "insurance_provider", None),
                    "policy_number": getattr(extracted, "policy_number", None),
                    "validity_date": str(getattr(extracted, "validity_date", None))[:10] if getattr(extracted, "validity_date", None) else None
                }
            else:
                doc_dict = {"type": "other", "content_summary": getattr(
                    extracted, "content_summary", None)}
            documents.append(doc_dict)
        # Validation: check for missing document types
        required_types = {"bill", "discharge_summary"}
        missing_documents = list(required_types - doc_types_found)
        discrepancies = []  # Add logic for cross-checks if needed
        # Claim decision logic
        if not missing_documents and not discrepancies:
            status = "approved"
            reason = "All required documents present and data is consistent"
        else:
            status = "rejected"
            reason = "Missing required documents or found discrepancies"
        return {
            "documents": documents,
            "validation": {
                "missing_documents": missing_documents,
                "discrepancies": discrepancies
            },
            "claim_decision": {
                "status": status,
                "reason": reason
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
