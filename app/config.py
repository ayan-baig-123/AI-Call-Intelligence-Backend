# backend/app/config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploaded_audio"))
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # If set, LLM calls authenticate to Ollama Cloud (https://ollama.com) using
    # this key instead of connecting to a local/self-hosted Ollama at
    # OLLAMA_HOST. Get a free key (no credit card) at
    # https://ollama.com/settings/keys - when using it, also set
    # OLLAMA_HOST=https://ollama.com and use a ":cloud"-suffixed model name
    # in LLM_MODEL (e.g. "gpt-oss:20b-cloud").
    OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    # Language hint for Whisper. Empty string = auto-detect (recommended when
    # calls can be in ANY language - Urdu, Punjabi, Pashto, Arabic, English,
    # mixed code-switching, etc.). Set to a specific code like "ur" only if
    # you know EVERY call will be in that one language - forcing a language
    # hint on multi-language calls makes Whisper mis-transcribe anything that
    # isn't that language.
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "")
    HUGGINGFACE_TOKEN: str = os.getenv("HF_TOKEN", "")
    # Runs a dedicated LLM pass that ONLY translates/cleans each transcript
    # line into proper English (no summarization, no insight extraction)
    # before the main structured-extraction step. Keeping translation
    # separate from extraction makes both more reliable, especially for
    # languages other than Urdu. Set to "false" to disable and rely solely
    # on Whisper's built-in translate task.
    ENABLE_TRANSLATION_VALIDATION: bool = os.getenv("ENABLE_TRANSLATION_VALIDATION", "true").lower() == "true"

settings = Settings()

# Upload folder na ho toh auto create ho jaye
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
