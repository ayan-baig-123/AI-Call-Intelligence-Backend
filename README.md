# AI Call Intelligence Backend

## Overview

This repository contains the **backend** services for the AI Call Intelligence platform. It is a Python FastAPI application that handles:
- Audio upload and storage
- Speech‑to‑text transcription (using `faster-whisper`)
- Optional speaker diarisation (`pyannote.audio`)
- Post‑processing of transcripts (role fixing, sentiment, categorisation)
- Generation of JSON, Excel, and PDF reports

## Project Structure
```
backend/
├─ app/
│   ├─ main.py                # FastAPI entry point
│   └─ services/
│       ├─ ai_engine.py       # LLM‑based sentiment & categorisation
│       └─ exporter.py        # Transcript cleaning, role fixing, report creation
├─ requirements.txt            # Python dependencies
└─ .env                        # Environment variables (e.g., OLLAMA_HOST)
```

## Prerequisites
- Python 3.10+ (recommended 3.11)
- `ffmpeg` (required by `faster-whisper`)
- **Ollama** installed locally and a model pulled (e.g., `ollama pull llama3.1`)
- (Optional) `PYANNOTE_TOKEN` if you wish to enable diarisation

## Setup
```bash
# Clone the repository (if you haven’t already)
git clone https://github.com/ayan-baig-123/AI-Call-Intelligence-Backend.git
cd backend

# Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# or source venv/bin/activate on Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the `backend/` directory (or set environment variables directly):
```
# .env example
OLLAMA_HOST=http://localhost:11434
PYANNOTE_TOKEN=YOUR_PYANNOTE_TOKEN   # optional, skip diarisation if absent
TRANSCRIPTS_DIR=./exported_transcripts
```

## Running the API
```bash
uvicorn app.main:app --reload   # accessible at http://127.0.0.1:8000
```
Visit `http://127.0.0.1:8000/docs` for the automatically generated Swagger UI.

## Usage Flow
1. **Upload** an audio file via the `/upload` endpoint.
2. The service transcribes the audio, runs optional diarisation, and calls `exporter.fix_transcript_roles_with_ollama` to assign roles.
3. Sentiment, dispute detection, and summary are extracted in `ai_engine.py`.
4. The cleaned transcript and reports are saved under `exported_transcripts/`.

## Testing
```bash
pytest   # runs all unit tests in the backend
```
Add more tests as you develop new features.

## Contributing
- Fork the repo and create a feature branch.
- Follow PEP‑8 style and run `flake8 .` before committing.
- Open a Pull Request with a clear description of changes.

## License
MIT License – see the `LICENSE` file in the repository.
