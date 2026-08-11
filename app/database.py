from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite Database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# Engine setup (check_same_thread=False for FastAPI async execution)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    pool_pre_ping=True  # Stale connections handle karne ke liye
)

# Session local factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base for models.py
Base = declarative_base()

# Dependency to get DB session in FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()