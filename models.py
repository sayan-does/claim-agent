from typing import List, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field

class BillData(BaseModel):
    type: Literal["bill"] = "bill"
    hospital_name: str
    patient_name: str
    patient_id: Optional[str] = None
    bill_number: str
    bill_date: datetime
    total_amount: float
    services: List[str] = []
    
class DischargeSummaryData(BaseModel):
    type: Literal["discharge_summary"] = "discharge_summary"
    hospital_name: str
    patient_name: str
    patient_id: Optional[str] = None
    admission_date: datetime
    discharge_date: datetime
    diagnosis: str
    treatment_summary: str
    doctor_name: str

class IDCardData(BaseModel):
    type: Literal["id_card"] = "id_card"
    patient_name: str
    patient_id: str
    insurance_provider: str
    policy_number: str
    validity_date: Optional[datetime] = None

class OtherDocumentData(BaseModel):
    type: Literal["other"] = "other"
    document_title: Optional[str] = None
    content_summary: str

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []

class ClaimDecision(BaseModel):
    decision: Literal["approved", "rejected", "requires_review"]
    reason: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    extracted_data: List[Union[BillData, DischargeSummaryData, IDCardData, OtherDocumentData]]
    validation_results: List[ValidationResult]