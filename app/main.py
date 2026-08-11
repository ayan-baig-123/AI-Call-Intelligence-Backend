import os
import shutil
import uuid
from datetime import datetime
from typing import List

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from sqlalchemy.sql.functions import coalesce

from app.database import engine, Base, get_db
from app.models import CallRecord
from app.schemas import CallRecordResponse, AnalyticsSummary
from app.services.exporter import generate_excel_report, generate_pdf_report

# Import Real AI Engine
from app.services.ai_engine import process_audio

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Call Intelligence Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------
# BACKGROUND TASK FUNCTION (For Uploaded & Live Audio)
# ---------------------------------------------------------
def process_audio_pipeline(call_id: str, file_path: str):
    """
    Background worker that runs the AI Engine
    and updates the database record upon completion with the new schema.
    """
    db = next(get_db())
    try:
        # Call AI Engine to transcribe & extract structured insights
        ai_result = process_audio(file_path)

        call = db.query(CallRecord).filter(CallRecord.id == call_id).first()
        if call:
            call.customer_name = ai_result.get("customer_name", "Live Caller")
            call.agent_name = ai_result.get("agent_name", "Support AI")
            call.issue_category = ai_result.get("issue_category", "Live Call")
            call.problem_statement = ai_result.get("problem_statement", "N/A")
            call.solution = ai_result.get("solution", "N/A")
            call.summary = ai_result.get("summary", "N/A")
            call.resolved = ai_result.get("resolved", "NO")
            call.sentiment = ai_result.get("sentiment", "Neutral")
            
            call.transcript_json = ai_result.get("transcript_json", {})
            call.status = "COMPLETED"
            
            db.commit()
            print(f"Successfully processed and updated call {call_id} in database columns.")
            
    except Exception as e:
        print(f"Error processing call {call_id}: {str(e)}")
        call = db.query(CallRecord).filter(CallRecord.id == call_id).first()
        if call:
            call.status = "FAILED"
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------

@app.get("/")
def root():
    return {"message": "AI Call Intelligence Backend Server Online"}


@app.get("/api/calls/search", response_model=List[CallRecordResponse])
def search_calls(q: str = "", db: Session = Depends(get_db)):
    """Search call records by customer name, agent, category, problem statement, or solution."""
    if not q or not q.strip():
        return []

    term = f"%{q.strip().lower()}%"

    results = db.query(CallRecord).filter(
        or_(
            func.lower(coalesce(CallRecord.customer_name, '')).like(term),
            func.lower(coalesce(CallRecord.agent_name, '')).like(term),
            func.lower(coalesce(CallRecord.issue_category, '')).like(term),
            func.lower(coalesce(CallRecord.problem_statement, '')).like(term),
            func.lower(coalesce(CallRecord.solution, '')).like(term)
        )
    ).all()

    return results


@app.get("/api/calls", response_model=List[CallRecordResponse])
def get_calls(db: Session = Depends(get_db)):
    """Fetch all recorded calls from the database."""
    calls = db.query(CallRecord).order_by(CallRecord.created_at.desc()).all()
    return calls


@app.get("/api/calls/export/excel")
def export_excel(db: Session = Depends(get_db)):
    """Export all calls as a clean Excel (.xlsx) file."""
    try:
        calls = db.query(CallRecord).order_by(CallRecord.created_at.desc()).all()
        file_stream = generate_excel_report(calls)
        return StreamingResponse(
            file_stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=Call_Analytics_Report.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate Excel report: {str(e)}")


@app.get("/api/calls/export/pdf")
def export_pdf(db: Session = Depends(get_db)):
    """Export all calls as a PDF report."""
    try:
        calls = db.query(CallRecord).order_by(CallRecord.created_at.desc()).all()
        pdf_stream = generate_pdf_report(calls)
        return StreamingResponse(
            pdf_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Call_Intelligence_Summary.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")


@app.get("/api/calls/{call_id}", response_model=CallRecordResponse)
def get_call_by_id(call_id: str, db: Session = Depends(get_db)):
    """Fetch a single call record by ID."""
    call = db.query(CallRecord).filter(CallRecord.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call record not found")
    return call


@app.get("/api/analytics", response_model=AnalyticsSummary)
def get_analytics(db: Session = Depends(get_db)):
    """Calculate live aggregated metrics from DB based on updated schema."""
    total_calls = db.query(CallRecord).count()
    if total_calls == 0:
        return AnalyticsSummary(
            total_calls=0,
            completed_calls=0,
            resolution_rate=0.0,
            pending_issues=0,
            top_categories={}
        )
    
    completed_calls = db.query(CallRecord).filter(CallRecord.status == "COMPLETED").count()
    resolved_count = db.query(CallRecord).filter(CallRecord.resolved == "YES").count()
    pending_count = db.query(CallRecord).filter(CallRecord.resolved == "NO").count()
    resolution_rate = round((resolved_count / total_calls) * 100, 1)

    categories = {}
    calls = db.query(CallRecord.issue_category).all()
    for (cat,) in calls:
        if cat:
            categories[cat] = categories.get(cat, 0) + 1

    return AnalyticsSummary(
        total_calls=total_calls,
        completed_calls=completed_calls,
        resolution_rate=resolution_rate,
        pending_issues=pending_count,
        top_categories=categories
    )


# ---------------------------------------------------------
# 1. MANUAL VOICE UPLOAD SYSTEM
# ---------------------------------------------------------
@app.post("/api/calls/upload", response_model=CallRecordResponse)
def upload_audio(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    call_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{call_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_call = CallRecord(
        id=call_id,
        audio_filename=file.filename,
        created_at=datetime.utcnow(),
        status="PROCESSING"
    )
    db.add(new_call)
    db.commit()
    db.refresh(new_call)

    background_tasks.add_task(process_audio_pipeline, call_id, file_path)

    return new_call


# ---------------------------------------------------------
# 2. AUTOMATED LIVE CALL SYSTEM (Webhook & WebSocket Integration)
# ---------------------------------------------------------
@app.post("/api/calls/incoming-webhook")
def incoming_call_webhook():
    twiml_response = """
    <Response>
        <Say>Please hold while your call is connected and recorded for quality assurance.</Say>
        <Start>
            <Stream url="wss://your-server-domain/ws/live-call-stream" />
        </Start>
        <Dial>+923001234567</Dial>
    </Response>
    """
    return StreamingResponse(iter([twiml_response]), media_type="application/xml")


@app.websocket("/ws/live-call-stream")
async def live_call_websocket(websocket: WebSocket, background_tasks: BackgroundTasks = None):
    """
    WebSocket endpoint that receives live audio chunks, saves them on disconnect, 
    and triggers the AI analysis pipeline to populate database columns.
    """
    await websocket.accept()
    call_id = str(uuid.uuid4())
    live_file_path = os.path.join(UPLOAD_DIR, f"live_{call_id}.wav")
    
    db = next(get_db())
    try:
        new_call = CallRecord(
            id=call_id,
            audio_filename=f"live_{call_id}.wav",
            customer_name="Live Caller",
            agent_name="Support AI",
            issue_category="Live Call",
            problem_statement="Recording live call stream...",
            solution="Pending analysis...",
            created_at=datetime.utcnow(),
            status="PROCESSING"
        )
        db.add(new_call)
        db.commit()

        with open(live_file_path, "wb") as audio_file:
            while True:
                message = await websocket.receive()
                if "bytes" in message:
                    audio_file.write(message["bytes"])
                elif "text" in message:
                    pass
    except WebSocketDisconnect:
        print(f"Live call stream disconnected for call ID: {call_id}. Starting AI analysis...")
        # WebSocket disconnect ke baad AI pipeline ko trigger karna taake columns update ho sakein
        process_audio_pipeline(call_id, live_file_path)
    except Exception as e:
        print(f"WebSocket error in live stream: {e}")
        call = db.query(CallRecord).filter(CallRecord.id == call_id).first()
        if call:
            call.status = "FAILED"
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------
# DELETE ENDPOINTS
# ---------------------------------------------------------

@app.delete("/api/calls/{call_id}", status_code=status.HTTP_200_OK)
def delete_call_by_id(call_id: str, db: Session = Depends(get_db)):
    call = db.query(CallRecord).filter(CallRecord.id == call_id).first()
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call record with ID '{call_id}' not found."
        )
    
    if call.audio_filename:
        file_path = os.path.join(UPLOAD_DIR, f"{call.id}_{call.audio_filename}") if not call.audio_filename.startswith("live_") else os.path.join(UPLOAD_DIR, call.audio_filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete physical file {file_path}: {e}")

    db.delete(call)
    db.commit()
    
    return {"message": f"Call record '{call_id}' successfully deleted."}


@app.delete("/api/calls", status_code=status.HTTP_200_OK)
def clear_all_calls(db: Session = Depends(get_db)):
    deleted_count = db.query(CallRecord).delete()
    db.commit()
    
    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete {file_path}: {e}")

    return {"message": f"Successfully deleted all {deleted_count} call records and associated audio files."}