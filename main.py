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

@app.post("/process-claim", response_model=ClaimDecision)
async def process_claim(files: List[UploadFile] = File(...)):
    """
    Main endpoint to process claim with multiple PDF files
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    
    # Check file types
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Got: {file.filename}")
    
    try:
        extracted_data = []
        
        # Process each file through the agent pipeline
        for file in files:
            # Step 1: Extract text
            text = await extract_text_from_pdf(file)
            
            if not text:
                continue  # Skip empty files
            
            # Step 2: Classify document
            doc_type = await classify_document_agent(text)
            
            # Step 3: Extract structured data
            structured_data = await extract_data_agent(text, doc_type)
            extracted_data.append(structured_data)
        
        if not extracted_data:
            raise HTTPException(status_code=400, detail="No valid content found in uploaded files")
        
        # Step 4: Validate data
        validation_results = await validate_claim_agent(extracted_data)
        
        # Step 5: Make final decision
        claim_decision = await decide_claim_agent(extracted_data, validation_results)
        
        return claim_decision
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)