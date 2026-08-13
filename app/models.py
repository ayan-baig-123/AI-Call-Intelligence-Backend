import uuid
from sqlalchemy import Column, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.database import Base

class CallRecord(Base):
    __tablename__ = "call_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    audio_filename = Column(String, nullable=True)  # <-- Added missing column here
    audio_url = Column(String, nullable=True)  # Public URL of audio in Supabase
    customer_name = Column(String, nullable=True, default="Unknown")
    agent_name = Column(String, nullable=True, default="Support AI")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    duration = Column(String, nullable=True, default="02:45")
    issue_category = Column(String, nullable=True, default="General")
    
    # New Columns
    problem_statement = Column(Text, nullable=True, default="N/A")
    solution = Column(Text, nullable=True, default="N/A")
    summary = Column(Text, nullable=True, default="N/A")
    resolved = Column(String, nullable=False, default="NO")  # "YES" or "NO"
    
    sentiment = Column(String, nullable=True, default="Neutral")
    status = Column(String, nullable=False, default="PROCESSING")  # PROCESSING, COMPLETED, FAILED
    
    # Transcript stored strictly as JSON
    transcript_json = Column(JSON, nullable=True, default=dict)
