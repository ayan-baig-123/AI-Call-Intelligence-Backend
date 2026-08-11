import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import asyncio

from app.database import get_db, SessionLocal
from app.models import CallRecord
from app.schemas import CallRecordResponse
from app.services.ai_engine import process_audio
from app.services.exporter import generate_excel_report, generate_pdf_report

router = APIRouter(prefix="/api/calls", tags=["Calls Processing"])
UPLOAD_DIR = "uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def run_ai_task(call_id: str, file_path: str, db_session_factory):
    """Background Worker for executing STT + LLM pipeline."""
    db = db_session_factory()
    try:
        call_rec = db.query(CallRecord).filter(CallRecord.id == call_id).first()
        if not call_rec:
            return

        ai_data = process_audio(file_path)

        call_rec.customer_name = ai_data.get("customer_name", "Live Caller")
        call_rec.agent_name = ai_data.get("agent_name", "Support AI")
        call_rec.issue_category = ai_data.get("issue_category", "Live Call")
        call_rec.problem_statement = ai_data.get("problem_statement", "N/A")
        call_rec.solution = ai_data.get("solution", "N/A")
        call_rec.summary = ai_data.get("summary", ai_data.get("ai_summary", "N/A"))
        call_rec.resolved = ai_data.get("resolved", "NO")
        call_rec.sentiment = ai_data.get("sentiment", "Neutral")
        call_rec.transcript_json = ai_data.get("transcript_json", {})
        call_rec.status = "COMPLETED"

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error in background task: {e}")
        call_rec = db.query(CallRecord).filter(CallRecord.id == call_id).first()
        if call_rec:
            call_rec.status = "FAILED"
            db.commit()
    finally:
        db.close()


@router.websocket("/ws/live-call-stream")
async def live_call_websocket(websocket: WebSocket):
    """WebSocket endpoint to capture live call stream chunks, save them on disconnect, and run AI processing."""
    await websocket.accept()
    audio_chunks = []
    
    db = SessionLocal()
    call_record = CallRecord(file_name="live_call_stream.webm", status="PROCESSING", issue_category="Live Call", customer_name="Live Caller", agent_name="Support AI")
    db.add(call_record)
    db.commit()
    db.refresh(call_record)
    call_id = call_record.id
    db.close()

    try:
        while True:
            data = await websocket.receive_bytes()
            if data:
                audio_chunks.append(data)
    except WebSocketDisconnect:
        if audio_chunks:
            file_path = os.path.join(UPLOAD_DIR, f"live_call_{call_id}.webm")
            with open(file_path, "wb") as f:
                for chunk in audio_chunks:
                    f.write(chunk)
            
            asyncio.create_task(asyncio.to_thread(run_ai_task, call_id, file_path, SessionLocal))
    except Exception as e:
        print(f"WebSocket error: {e}")
        db = SessionLocal()
        call_rec = db.query(CallRecord).filter(CallRecord.id == call_id).first()
        if call_rec:
            call_rec.status = "FAILED"
            db.commit()
        db.close()


@router.post("/upload", response_model=CallRecordResponse)
async def upload_audio_call(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    call_record = CallRecord(file_name=file.filename, status="PROCESSING")
    db.add(call_record)
    db.commit()
    db.refresh(call_record)

    background_tasks.add_task(run_ai_task, call_record.id, file_path, SessionLocal)

    return call_record


@router.get("/")
def get_all_calls(db: Session = Depends(get_db)):
    """Fetch all call records with explicit summary and alias fields mapped."""
    calls = db.query(CallRecord).order_by(CallRecord.created_at.desc()).all()
    response_list = []
    
    for call in calls:
        call_dict = {c.name: getattr(call, c.name) for c in call.__table__.columns}
        
        active_summary = call.summary or "Summary is currently processing or unavailable."
        call_dict["summary"] = active_summary
        call_dict["ai_summary"] = active_summary
        call_dict["executive_summary"] = active_summary
        
        response_list.append(call_dict)
        
    return response_list


@router.get("/export/excel")
def export_excel(db: Session = Depends(get_db)):
    """Export all calls as a clean Excel (.xlsx) file[cite: 5]."""
    try:
        records = db.query(CallRecord).order_by(CallRecord.created_at.desc()).all()
        excel_file = generate_excel_report(records)
        return StreamingResponse(
            excel_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=Call_Analytics_Report.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate Excel report: {str(e)}")


@router.get("/export/pdf")
def export_pdf(db: Session = Depends(get_db)):
    """Export all calls as a PDF report[cite: 5]."""
    try:
        records = db.query(CallRecord).order_by(CallRecord.created_at.desc()).all()
        pdf_stream = generate_pdf_report(records)
        return StreamingResponse(
            pdf_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Call_Intelligence_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")


@router.get("/{call_id}")
def get_call_by_id(call_id: str, db: Session = Depends(get_db)):
    """Fetch a single call record by ID with full summary mapping[cite: 5]."""
    call = db.query(CallRecord).filter(CallRecord.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call record not found")
    
    call_dict = {c.name: getattr(call, c.name) for c in call.__table__.columns}
    active_summary = call.summary or "Summary is currently processing or unavailable."
    
    call_dict["summary"] = active_summary
    call_dict["ai_summary"] = active_summary
    call_dict["executive_summary"] = active_summary
    
    return call_dict


@router.delete("/{call_id}")
def delete_call_record(call_id: str, db: Session = Depends(get_db)):
    call = db.query(CallRecord).filter(CallRecord.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call record not found")
    
    db.delete(call)
    db.commit()
    return {"message": "Call record deleted successfully", "id": call_id}