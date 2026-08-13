"""
backend/app/services/storage.py

Uploads call-recording audio files to Supabase Storage and returns a
public URL. This is SEPARATE from DATABASE_URL/Postgres - Postgres
stores the call records (text/JSON data), this uploads the actual
.mp3/.wav audio files to Supabase's file storage (like an S3 bucket).

REQUIRED environment variables (add these to your .env, alongside
DATABASE_URL):
    SUPABASE_URL          - e.g. https://iteyugzuefptvxursf.supabase.co
                             (same project as your DATABASE_URL host)
    SUPABASE_SERVICE_KEY   - the "service_role" key from
                             Supabase Dashboard -> Project Settings -> API
                             (NOT the public "anon" key - the service role
                             key is needed to upload files server-side)
    SUPABASE_BUCKET        - name of the Storage bucket to upload into
                             (default: "call-audio" - create this bucket
                             in Supabase Dashboard -> Storage first, and
                             mark it "Public" if you want get_public_url()
                             links to work without a signed URL)

REQUIRED package:
    pip install supabase
"""

import os
import uuid
import logging

logger = logging.getLogger("uvicorn.error")

try:
    from supabase import create_client, Client
    SUPABASE_SDK_AVAILABLE = True
except ImportError:
    SUPABASE_SDK_AVAILABLE = False
    Client = None

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# Accept either name - some setups call it SUPABASE_SERVICE_KEY, others SUPABASE_KEY
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "call-audio")

_supabase_client = None


def get_supabase_client():
    """Lazily creates (and caches) the Supabase client on first use."""
    global _supabase_client

    if not SUPABASE_SDK_AVAILABLE:
        raise RuntimeError(
            "The 'supabase' package is not installed. Run: pip install supabase"
        )

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and/or SUPABASE_SERVICE_KEY are not set in the "
            "environment. Add them to your .env file (see storage.py header "
            "comment for details)."
        )

    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"Supabase Storage client initialized for bucket '{SUPABASE_BUCKET}'.")

    return _supabase_client


def upload_to_supabase(file) -> str:
    """
    Uploads a FastAPI UploadFile to Supabase Storage and returns its
    public URL.

    IMPORTANT: in calls.py this is called AFTER file.file has already
    been copied to local disk via shutil.copyfileobj(), which leaves the
    stream positioned at EOF. We explicitly seek(0) before reading here
    so the upload doesn't silently send an empty file, and seek(0) again
    afterwards in case any code after this call still needs to read it.
    """
    client = get_supabase_client()

    file.file.seek(0)
    file_bytes = file.file.read()
    file.file.seek(0)

    if not file_bytes:
        raise ValueError(
            f"'{file.filename}' read as empty - nothing to upload to Supabase Storage."
        )

    ext = os.path.splitext(file.filename or "")[1] or ".mp3"
    unique_name = f"{uuid.uuid4()}{ext}"
    content_type = file.content_type or "audio/mpeg"

    try:
        client.storage.from_(SUPABASE_BUCKET).upload(
            path=unique_name,
            file=file_bytes,
            file_options={"content-type": content_type},
        )
    except Exception as e:
        logger.error(f"Supabase Storage upload failed for '{file.filename}': {e}")
        raise

    public_url = client.storage.from_(SUPABASE_BUCKET).get_public_url(unique_name)
    logger.info(f"Uploaded '{file.filename}' to Supabase Storage as '{unique_name}'.")
    return public_url