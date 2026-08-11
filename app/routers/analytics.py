from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models import CallRecord

router = APIRouter(prefix="/api/analytics", tags=["Dashboard Metrics"])

@router.get("/summary")
def get_analytics(db: Session = Depends(get_db)):
    total_calls = db.query(CallRecord).count()
    completed = db.query(CallRecord).filter(CallRecord.status == "COMPLETED").count()
    resolved = db.query(CallRecord).filter(CallRecord.is_resolved == True).count()
    follow_ups = db.query(CallRecord).filter(CallRecord.follow_up_required == True).count()

    res_rate = round((resolved / total_calls * 100), 1) if total_calls > 0 else 0.0

    sentiments = db.query(CallRecord.sentiment, func.count(CallRecord.id)).group_by(CallRecord.sentiment).all()
    categories = db.query(CallRecord.issue_category, func.count(CallRecord.id)).group_by(CallRecord.issue_category).all()

    return {
        "kpis": {
            "total_calls": total_calls,
            "completed_calls": completed,
            "resolved_calls": resolved,
            "follow_ups_needed": follow_ups,
            "resolution_rate": f"{res_rate}%"
        },
        "charts": {
            "sentiment_distribution": {k or "Neutral": v for k, v in sentiments},
            "category_distribution": {k or "General": v for k, v in categories}
        }
    }