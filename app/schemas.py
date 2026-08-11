from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class CallRecordBase(BaseModel):
    file_name: Optional[str] = None
    customer_name: Optional[str] = "Unknown"
    agent_name: Optional[str] = "Support AI"
    duration: Optional[str] = "02:45"
    issue_category: Optional[str] = "General"
    
    # New Requested Intelligence Fields
    problem_statement: Optional[str] = "N/A"
    solution: Optional[str] = "N/A"
    resolved: str = "NO"  # "YES" | "NO"
    
    sentiment: Optional[str] = "Neutral"
    status: str = "PROCESSING"  # "PROCESSING" | "COMPLETED" | "FAILED"
    
    # Transcript strictly stored as JSON
    transcript_json: Optional[Dict[str, Any]] = {}

class CallRecordCreate(CallRecordBase):
    pass

class CallRecordResponse(CallRecordBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class AnalyticsSummary(BaseModel):
    total_calls: int
    completed_calls: int
    resolution_rate: float
    pending_issues: int
    top_categories: Dict[str, int]