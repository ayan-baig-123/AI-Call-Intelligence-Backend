import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# -----------------------------------------------------------------
# Load variables from a local .env file (if present) into the process
# environment. This MUST run before os.getenv("DATABASE_URL") below,
# otherwise a .env file sitting on disk is never actually read and the
# app silently falls back to SQLite even when Postgres creds exist.
# On most hosting platforms (Render, Railway, Fly.io, etc.) you instead
# set DATABASE_URL directly in the platform's dashboard/environment
# settings - in that case this call is a harmless no-op since the
# variable is already in the environment.
# -----------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed - fine on platforms that inject env
    # vars directly, but local .env files won't be picked up until you
    # run: pip install python-dotenv
    pass

# -----------------------------------------------------------------
# Use environment variable DATABASE_URL; fallback to local SQLite for dev
# -----------------------------------------------------------------
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",                # Supabase / external PostgreSQL URL
    "sqlite:///./sql_app.db"      # Local fallback
)

import logging
logger = logging.getLogger("uvicorn.error")
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    logger.warning(
        "DATABASE_URL not found in environment - falling back to local SQLite. "
        "If you meant to use Postgres/Supabase, make sure DATABASE_URL is set "
        "(via .env + python-dotenv locally, or your hosting platform's env settings)."
    )
else:
    # Don't log the full URL - it contains the DB password
    safe_host = SQLALCHEMY_DATABASE_URL.split("@")[-1] if "@" in SQLALCHEMY_DATABASE_URL else "configured host"
    logger.info(f"Using external database at {safe_host}")

# Engine creation – works for both SQLite & PostgreSQL
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # SQLite‑specific flag; ignored for PostgreSQL
    connect_args={"check_same_thread": False}
    if SQLALCHEMY_DATABASE_URL.startswith("sqlite")
    else {},
    pool_pre_ping=True               # keep connections alive
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base for models
Base = declarative_base()

# Dependency for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
