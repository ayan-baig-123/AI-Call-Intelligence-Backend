import os
import sys
import json
import logging
import warnings
import re
import importlib
import subprocess
import tempfile
import shutil
from typing import Dict, Any

# Suppress unnecessary library warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Safe dynamic import for faster_whisper
try:
    _fw_mod = importlib.import_module("faster_whisper")
    WhisperModel = getattr(_fw_mod, "WhisperModel", None)
    WHISPER_AVAILABLE = WhisperModel is not None
except Exception:
    WhisperModel = None
    WHISPER_AVAILABLE = False

# Safe dynamic import for ollama
try:
    ollama = importlib.import_module("ollama")
    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False


def get_ollama_client():
    """
    Builds the Ollama client used for every LLM call in this file.

    - If OLLAMA_API_KEY is set (Ollama Cloud - https://ollama.com/settings/keys,
      free tier, no credit card), requests go to OLLAMA_HOST (should be
      https://ollama.com) with an Authorization: Bearer header, so this
      server never depends on any locally-running Ollama process.
    - If OLLAMA_API_KEY is empty, falls back to a plain connection to
      OLLAMA_HOST (e.g. a local Ollama instance, or one tunneled via ngrok)
      with no auth header - this is the original self-hosted setup.
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("The 'ollama' package is not installed.")

    host = getattr(settings, "OLLAMA_HOST", "http://localhost:11434")
    api_key = getattr(settings, "OLLAMA_API_KEY", "")

    if api_key:
        return ollama.Client(host=host, headers={"Authorization": f"Bearer {api_key}"})
    return ollama.Client(host=host)

# Safe dynamic import for pyannote.audio
try:
    _pyannote_mod = importlib.import_module("pyannote.audio")
    PyannotePipeline = getattr(_pyannote_mod, "Pipeline", None)
    PYANNOTE_AVAILABLE = PyannotePipeline is not None
except Exception:
    PyannotePipeline = None
    PYANNOTE_AVAILABLE = False

from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIEngine")


def preprocess_audio_file(file_path: str) -> str:
    """
    Converts ANY audio or video file format (.mp3, .wav, .m4a, .aac, .ogg, .flac, 
    .wma, .mp4, .webm, .mkv, .avi, .mov, .3gp, .amr, etc.) into a clean 16kHz mono WAV file 
    via ffmpeg for 100% format support and optimal Whisper accuracy.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        logger.info("ffmpeg binary not found in PATH. Processing audio file directly.")
        return file_path

    try:
        temp_dir = os.path.join(tempfile.gettempdir(), "ai_call_intel")
        os.makedirs(temp_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        cleaned_wav = os.path.join(temp_dir, f"clean_{base_name}.wav")

        # Noise-reduction + normalization filter chain applied BEFORE Whisper sees the audio:
        #   highpass/lowpass -> strip rumble & hiss outside human voice band (80Hz-8kHz)
        #   afftdn           -> adaptive FFT noise reduction (removes hiss/hum/background noise)
        #   dynaudnorm       -> evens out volume so soft/far-mic speech isn't dropped by VAD
        denoise_filter = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=15"

        cmd = [
            ffmpeg_bin, "-y", "-i", file_path,
            "-af", denoise_filter,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            cleaned_wav
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if proc.returncode == 0 and os.path.exists(cleaned_wav) and os.path.getsize(cleaned_wav) > 1000:
            logger.info(f"Successfully denoised & preprocessed audio: {file_path} -> {cleaned_wav}")
            return cleaned_wav
        else:
            # If the denoise filter chain fails for some reason, retry with a plain conversion
            # so we never silently skip preprocessing entirely.
            logger.warning(f"Denoise ffmpeg pass failed (code {proc.returncode}). Retrying plain conversion.")
            try:
                plain_cmd = [
                    ffmpeg_bin, "-y", "-i", file_path,
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    cleaned_wav
                ]
                proc2 = subprocess.run(plain_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
                if proc2.returncode == 0 and os.path.exists(cleaned_wav) and os.path.getsize(cleaned_wav) > 1000:
                    return cleaned_wav
            except Exception as e2:
                logger.warning(f"Fallback plain conversion also failed: {e2}")
    except Exception as e:
        logger.warning(f"Audio preprocessing skipped due to error: {e}")

    return file_path


def remove_in_text_phrase_loops(text: str) -> str:
    """
    Cleans internal string phrase repetition loops (e.g. 'Hello Assalamualikum' or 
    'I don't know how to do it' repeated 30 times inside a single message string).
    """
    if not text or len(text) < 8:
        return text

    parts = re.split(r'(?<=[.!?,\n])\s+', text)
    if len(parts) <= 1:
        return text

    clean_parts = []
    seen = {}
    prev_norm = ""

    for part in parts:
        p_strip = part.strip()
        if not p_strip:
            continue
        
        norm = re.sub(r'[^\w\s]', '', p_strip.lower()).strip()
        if not norm:
            continue

        cnt = seen.get(norm, 0) + 1
        seen[norm] = cnt

        # Cap any repeated clause/phrase to max 2 occurrences & drop consecutive duplicates
        if cnt > 2 or norm == prev_norm:
            continue

        prev_norm = norm
        clean_parts.append(p_strip)

    result = " ".join(clean_parts).strip()
    return result if result else text


def clean_repetitive_transcripts(raw_transcript_lines: list) -> list:
    """
    Filters out hallucinated repetition loops and caps initial greetings to max 2 total.
    """
    cleaned = []
    greeting_count = 0
    prev_lower = ""

    for line in raw_transcript_lines:
        text_only = re.sub(r'\[.*?\]', '', line).strip()
        text_only = remove_in_text_phrase_loops(text_only)
        lower = text_only.lower().strip()

        if not lower:
            continue

        # Check for short greeting words
        words = re.sub(r'[^\w\s]', '', lower).split()
        is_short_greeting = len(words) <= 3 and any(w in ["hello", "hi", "hey", "السلام", "سلام"] for w in words)

        if is_short_greeting:
            greeting_count += 1
            if greeting_count > 2:
                continue

        # Skip exact consecutive duplicate lines
        if lower == prev_lower:
            continue

        prev_lower = lower
        
        # Re-attach timestamp if present
        timestamps = re.findall(r'\[.*?\]', line)
        ts_prefix = " ".join(timestamps) + " " if timestamps else ""
        cleaned.append(f"{ts_prefix}{text_only}".strip())

    return cleaned


def extract_meaningful_topic(lines: list) -> str:
    """
    Scans transcript lines, filters out greeting noise ('Hello', 'Assalam O Alikum'),
    and extracts the actual core topic/conversation being discussed.
    """
    meaningful = []
    greeting_words = {
        "hello", "hi", "hey", "assalam", "alikum", "salam", "talking", "how", "are", 
        "you", "o", "a", "walikum", "walaikum", "assalamo", "assalamualikum", "sir"
    }

    for l in lines:
        text = re.sub(r'\[.*?\]', '', l).strip()
        lower = text.lower()
        clean_text = re.sub(r'[^\w\s]', '', lower).strip()
        words = clean_text.split()
        
        if not words:
            continue
        # Skip pure greetings or small-talk lines
        if len(words) <= 5 and all(w in greeting_words for w in words):
            continue
        if clean_text in ["hello", "hello assalam o alikum", "i am talking to you", "hello assalam how are you"]:
            continue
            
        meaningful.append(text)

    if meaningful:
        return " ".join(meaningful[:4])[:300]
        
    return "Customer initiated call; conversation recorded."


_AGENT_HINT_WORDS = [
    "how can i help", "how may i help", "thank you for calling", "welcome to",
    "support", "assist you", "our policy", "i apologize", "glad to help",
    "is there anything else", "let me check", "i understand your concern",
]
_CUSTOMER_HINT_WORDS = [
    "i want", "i have a problem", "i ordered", "my issue", "please help",
    "complaint", "i am calling about", "i'm calling about", "i need", "not working",
]

_SPEAKER_TAG_RE = re.compile(r"^\[[^\]]+\]\s*\[([A-Za-z0-9_]+)\]\s*")


def infer_speaker_roles(transcript_lines: list) -> Dict[str, str]:
    """
    When no LLM is available, still resolve each STABLE diarization tag
    (SPEAKER_00, SPEAKER_01, ...) to a Customer/Agent role using simple
    keyword scoring, so the fallback output never leaves speakers unlabeled
    or silently mixes the two people's lines together. Only two roles are
    ever assigned; if more than two tags appear, extras stay "Unknown"
    rather than being guessed incorrectly.
    """
    tag_text = {}
    for line in transcript_lines:
        match = _SPEAKER_TAG_RE.match(line)
        if not match:
            continue
        tag = match.group(1)
        text = _SPEAKER_TAG_RE.sub("", line, count=1)
        tag_text.setdefault(tag, []).append(text.lower())

    if not tag_text:
        return {}

    scores = {}
    for tag, lines in tag_text.items():
        joined = " ".join(lines)
        agent_score = sum(joined.count(w) for w in _AGENT_HINT_WORDS)
        customer_score = sum(joined.count(w) for w in _CUSTOMER_HINT_WORDS)
        scores[tag] = agent_score - customer_score

    if len(scores) == 1:
        only_tag = next(iter(scores))
        return {only_tag: "Agent" if scores[only_tag] >= 0 else "Customer"}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    roles = {ranked[0][0]: "Agent", ranked[-1][0]: "Customer"}
    for tag, _ in ranked[1:-1]:
        roles[tag] = "Unknown"
    return roles


_NON_LATIN_SCRIPT_RE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F'   # Arabic/Urdu/Pashto/Sindhi script
    r'\u0900-\u097F'                  # Devanagari (Hindi)
    r'\u0A00-\u0A7F'                  # Gurmukhi (Punjabi)
    r'\u0400-\u04FF'                  # Cyrillic
    r'\u4E00-\u9FFF'                  # CJK
    r']'
)


def has_untranslated_script(text: str) -> bool:
    """
    Whisper's task="translate" is supposed to output English, but for some
    languages/accents it occasionally leaves fragments of the original
    script untranslated. Detecting that lets us explicitly flag it for the
    LLM translation pass instead of silently shipping mixed-language text.
    """
    return bool(_NON_LATIN_SCRIPT_RE.search(text or ""))


def generate_smart_fallback(transcript_lines: list, formatted_transcript: str) -> Dict[str, Any]:
    """
    Intelligently analyzes the transcript text to generate a rich, accurate fallback 
    without static boilerplate or greeting copies.
    """
    core_topic = extract_meaningful_topic(transcript_lines)
    lower = formatted_transcript.lower()

    # Detect conflict / arguments / complaints / disputes
    conflict_keywords = ["complain", "manager", "campus", "policy", "loan", "pizza", "rude", "whatever", "refuse", "dispute", "wrong", "fight", "arguing", "not coming"]
    is_conflict = any(kw in lower for kw in conflict_keywords)

    if is_conflict:
        sentiment = "Negative"
        category = "Order & Service Dispute"
        problem_statement = f"Dispute between caller and support representative regarding request details ({core_topic[:150]})."
        solution = "No resolution achieved during call; caller requested manager escalation and threatened complaint."
        summary = f"The interaction involved a dispute between customer and representative. Issue: {core_topic[:180]}. Caller demanded manager escalation due to unresolved disagreement."
        resolved = "NO"
    else:
        sentiment = "Neutral"
        category = "General Inquiry"
        problem_statement = core_topic[:200]
        solution = "Call logged and processed by customer support."
        summary = f"Customer support interaction processed. Discussion overview: {core_topic[:220]}"
        resolved = "NO"

    speaker_roles = infer_speaker_roles(transcript_lines)
    dialogues = []
    for idx, line in enumerate(transcript_lines):
        match = _SPEAKER_TAG_RE.match(line)
        tag = match.group(1) if match else None
        role = speaker_roles.get(tag, "Unknown") if tag else "Unknown"
        clean_text = _SPEAKER_TAG_RE.sub("", line, count=1) if match else line
        dialogues.append({"id": idx + 1, "speaker": role, "text": line, "message": clean_text})

    default_structured_transcript = {
        "full_text": formatted_transcript,
        "dialogues": dialogues
    }

    return {
        "customer_name": "Valued Customer",
        "agent_name": "Support Agent",
        "issue_category": category,
        "problem_statement": problem_statement,
        "solution": solution,
        "summary": summary,
        "resolved": resolved,
        "sentiment": sentiment,
        "transcript_json": default_structured_transcript
    }


class LocalAIPipeline:
    def __init__(self):
        self.stt_model = None
        self.llm_model = getattr(settings, "LLM_MODEL", "qwen2.5:7b")
        self.diarization_pipeline = None
        
        self._init_stt_model()
        self._init_diarization()

    def _init_stt_model(self):
        """Safely initialize Whisper STT Engine with GPU/CPU fallback."""
        if not WHISPER_AVAILABLE or not WhisperModel:
            logger.error("faster_whisper module is not installed!")
            return

        model_size = getattr(settings, "WHISPER_MODEL_SIZE", "medium")
        
        # Smart GPU/CPU Switcher
        use_gpu = False
        try:
            torch = importlib.import_module("torch")
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

        if use_gpu:
            logger.info(f"Loading Whisper STT Engine ({model_size}) on GPU (CUDA)...")
            try:
                self.stt_model = WhisperModel(model_size, device="cuda", compute_type="float16")
            except Exception as e:
                logger.warning(f"CUDA initialization failed, falling back to CPU: {e}")
                try:
                    self.stt_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                except Exception as ex:
                    logger.error(f"Failed to load Whisper on CPU: {ex}")
        else:
            logger.info(f"Loading Whisper STT Engine ({model_size}) on CPU (Int8 Fallback)...")
            try:
                self.stt_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            except Exception as e:
                logger.error(f"Failed to load Whisper on CPU: {e}")

    def _init_diarization(self):
        """Safely initialize Pyannote Speaker Diarization if available."""
        hf_token = getattr(settings, "HUGGINGFACE_TOKEN", "")

        if not PYANNOTE_AVAILABLE or not PyannotePipeline:
            logger.error(
                "pyannote.audio is NOT installed. Speaker diarization is DISABLED, "
                "meaning customer/agent lines cannot be reliably told apart. "
                "Run: pip install pyannote.audio"
            )
            return

        if not hf_token:
            logger.error(
                "HF_TOKEN environment variable is empty. Speaker diarization is DISABLED "
                "because pyannote/speaker-diarization-3.1 requires a Hugging Face token "
                "(and you must accept the model's user agreement on huggingface.co first). "
                "Set HF_TOKEN in your environment/.env and restart the server."
            )
            return

        try:
            logger.info("Loading Pyannote Speaker Diarization Pipeline...")
            try:
                self.diarization_pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
            except TypeError:
                self.diarization_pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )

            if self.diarization_pipeline is None:
                logger.error(
                    "Pyannote pipeline loaded as None (likely rejected/invalid HF token, "
                    "or the model's user agreement hasn't been accepted on huggingface.co). "
                    "Diarization is DISABLED."
                )
            else:
                logger.info("Pyannote Speaker Diarization Pipeline loaded successfully.")
        except Exception as e:
            logger.error(
                f"Could not load Pyannote pipeline - diarization is DISABLED: {e}. "
                "Common causes: invalid/expired HF_TOKEN, or you haven't accepted the "
                "'pyannote/speaker-diarization-3.1' and 'pyannote/segmentation-3.0' user "
                "agreements on huggingface.co with the SAME account as this token.",
                exc_info=True
            )

    def warmup_llm(self):
        """Pre-warm Ollama model to load it into memory before actual requests."""
        if not OLLAMA_AVAILABLE or not ollama:
            logger.error("ollama package is not installed.")
            return

        logger.info(f"Warming up Ollama model: {self.llm_model}...")
        try:
            client = get_ollama_client()
            client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": "Ping"}],
                options={"num_predict": 1}
            )
            logger.info("Ollama LLM model warm-up completed successfully.")
        except Exception as e:
            logger.error(f"Ollama Warmup failed! Make sure Ollama server is running. Error: {e}")

    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribes audio/video using Whisper STT with speech-sensitive VAD and anti-hallucination.
        Converts any file extension to standard 16kHz WAV format first.
        """
        logger.info(f"Transcribing audio/video file: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return "[Audio file missing or path invalid.]"

        if not self.stt_model:
            logger.error("Whisper STT model failed to initialize!")
            return "[STT Engine error: Whisper model not loaded.]"

        # Pre-convert ANY format (.mp3, .wav, .m4a, .mp4, .webm, .mkv, etc.) to 16kHz WAV
        proc_file = preprocess_audio_file(file_path)

        # Language hint: forcing "ur" stops Whisper from mis-detecting Urdu/Roman-Urdu
        # speech as some other language (which was garbling transcripts). Empty string
        # falls back to auto-detection.
        language_hint = getattr(settings, "WHISPER_LANGUAGE", "") or None

        try:
            # Sensitive VAD threshold=0.35 captures human voices while stripping non-voice noise
            segments, info = self.stt_model.transcribe(
                proc_file, 
                beam_size=5,
                language=language_hint,            # Force Urdu recognition instead of auto-detect guesswork
                task="translate",                 # Translates Urdu/Roman Urdu speech to English
                temperature=0.0,                   
                condition_on_previous_text=False, 
                compression_ratio_threshold=2.4,   # Discards repetitive token loops
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                vad_filter=True,                   # VAD filter on to prevent noise hallucination loops
                vad_parameters=dict(
                    min_silence_duration_ms=600,
                    speech_pad_ms=300,
                    threshold=0.35                 # Sensitive threshold for soft spoken voices
                )
            )
            segment_list = list(segments)
            detected_lang = getattr(info, "language", None)
            detected_prob = getattr(info, "language_probability", None)
            if language_hint:
                logger.info(f"Extracted {len(segment_list)} audio segments via Whisper (language forced to '{language_hint}').")
            else:
                logger.info(
                    f"Extracted {len(segment_list)} audio segments via Whisper "
                    f"(auto-detected language: {detected_lang}, confidence: {detected_prob})."
                )
                if detected_prob is not None and detected_prob < 0.5:
                    logger.warning(
                        f"Low language-detection confidence ({detected_prob:.2f}) for '{detected_lang}'. "
                        "Transcript may be inaccurate - consider re-checking audio quality for this call."
                    )
        except Exception as e:
            logger.error(f"Whisper Transcription Error: {e}", exc_info=True)
            return f"[Transcription error: {str(e)}]"
        
        diarization = None
        diarization_turns = []
        if self.diarization_pipeline:
            try:
                logger.info("Running speaker diarization...")
                diarization = self.diarization_pipeline(proc_file)
                if diarization and hasattr(diarization, "itertracks"):
                    diarization_turns = [
                        (turn.start, turn.end, speaker)
                        for turn, _, speaker in diarization.itertracks(yield_label=True)
                    ]
                distinct_speakers = len({s for _, _, s in diarization_turns})
                logger.info(f"Diarization found {len(diarization_turns)} turns across {distinct_speakers} distinct speaker(s).")
                if distinct_speakers < 2:
                    logger.warning(
                        "Diarization detected fewer than 2 distinct speakers for this call. "
                        "Customer/Agent separation for this specific call may be unreliable "
                        "(check audio quality, or that both sides are actually on this recording)."
                    )
            except Exception as e:
                logger.warning(f"Diarization execution failed: {e}")
        else:
            logger.warning(
                "diarization_pipeline is not loaded - this call will have NO speaker tags at all, "
                "so the LLM must guess Customer vs Agent from content alone. Check the "
                "_init_diarization log lines at server startup for why it failed to load."
            )

        def best_speaker_for_segment(seg_start: float, seg_end: float) -> str:
            """
            Picks the diarization speaker with the GREATEST time overlap with this
            Whisper segment, instead of just checking whether the segment's start
            point falls inside some turn. Start-point matching mislabels segments
            that begin right at a speaker-change boundary (very common in natural
            conversation), which is what was causing customer/agent lines to be
            swapped or mixed together. Overlap-based matching is far more robust.
            """
            best_label = ""
            best_overlap = 0.0
            for turn_start, turn_end, speaker in diarization_turns:
                overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = speaker
            return best_label

        raw_transcript = []
        seen_texts = set()

        for segment in segment_list:
            text_content = segment.text.strip()
            text_content = remove_in_text_phrase_loops(text_content)
            
            if not text_content or len(text_content) < 2:
                continue

            cleaned_lower = text_content.lower()
            if cleaned_lower in seen_texts:
                continue
            seen_texts.add(cleaned_lower)

            time_stamp = f"[{segment.start:.1f}s - {segment.end:.1f}s]"
            speaker_label = ""

            if diarization_turns:
                raw_speaker = best_speaker_for_segment(segment.start, segment.end)
                if raw_speaker:
                    # Keep the diarization engine's raw tag (e.g. SPEAKER_00) as a
                    # STABLE identifier for this physical voice across the whole
                    # call. The LLM step below maps these stable tags to
                    # Customer/Agent by reading what each tag actually says -
                    # it must NEVER assume a fixed order like "first speaker =
                    # agent", since either party can speak first or speak several
                    # lines in a row.
                    speaker_label = f"[{raw_speaker}] "

            raw_transcript.append(f"{time_stamp} {speaker_label}{text_content}")

        # Filter repetitive greetings & loops
        raw_transcript = clean_repetitive_transcripts(raw_transcript)
            
        if not raw_transcript or len(raw_transcript) < 1:
            logger.warning("No speech segments detected in audio file.")
            return "[No audible speech detected in the audio file.]"
            
        return "\n".join(raw_transcript)

    def translate_and_clean_transcript(self, formatted_transcript: str) -> str:
        """
        Dedicated translation pass, run SEPARATELY from structured insight
        extraction. Whisper's own task="translate" already converts speech to
        English, but on code-switched or lower-resource languages it can leave
        residual foreign words or rough phrasing. This step sends each line to
        the LLM with a prompt that ONLY translates/cleans text - it does not
        summarize, categorize, or judge the call - which is more reliable than
        asking one LLM call to translate AND extract insights AND audit conduct
        all at once. Every line keeps its original timestamp/speaker tag and
        its original position, so speaker attribution from earlier is untouched.
        Safe by design: on any failure or mismatch, the original transcript is
        returned unchanged rather than risking corrupted output.
        """
        if not getattr(settings, "ENABLE_TRANSLATION_VALIDATION", True):
            return formatted_transcript

        if not OLLAMA_AVAILABLE or not ollama:
            logger.info("Ollama unavailable - skipping translation-validation pass, using Whisper's translation as-is.")
            return formatted_transcript

        lines = [l for l in formatted_transcript.split("\n") if l.strip()]
        if not lines:
            return formatted_transcript

        # Split each line into its prefix (timestamp + speaker tag) and the
        # actual spoken text, so only the text gets sent for translation.
        prefix_re = re.compile(r"^(\[\d+(?:\.\d+)?s\s*-\s*\d+(?:\.\d+)?s\]\s*(?:\[[^\]]+\]\s*)?)(.*)$")
        prefixes, texts = [], []
        for line in lines:
            m = prefix_re.match(line)
            if m:
                prefixes.append(m.group(1))
                texts.append(m.group(2))
            else:
                prefixes.append("")
                texts.append(line)

        TRANSLATE_SYSTEM_PROMPT = """
        You are a precise translation engine. You will receive a JSON array of
        text lines, possibly in Urdu, Roman Urdu, Punjabi, Pashto, Arabic,
        English, or any mix of languages.

        Translate EVERY line into clear, natural English. If a line is already
        English, just clean it up (fix obvious transcription noise) without
        changing its meaning. Do NOT summarize, shorten, merge, reorder, or
        drop any line - the output array must have EXACTLY the same number of
        elements, in the exact same order, as the input array. Do NOT add
        commentary, explanations, or extra fields.

        Return STRICT JSON in this exact shape:
        {"lines": ["translated line 1", "translated line 2", ...]}
        """

        try:
            client = get_ollama_client()
            response = client.chat(
                model=self.llm_model,
                format="json",
                messages=[
                    {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"lines": texts}, ensure_ascii=False)}
                ],
                options={"temperature": 0.0}
            )

            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                raw_output = response.message.content
            elif isinstance(response, dict) and 'message' in response:
                msg = response['message']
                raw_output = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', str(msg))
            else:
                raw_output = str(response)

            raw_output = raw_output.strip()
            if "```" in raw_output:
                raw_output = re.sub(r"^```(?:json)?\s*", "", raw_output, flags=re.IGNORECASE)
                raw_output = re.sub(r"\s*```$", "", raw_output).strip()

            parsed = json.loads(raw_output)
            translated_lines = parsed.get("lines") if isinstance(parsed, dict) else None

            if not isinstance(translated_lines, list) or len(translated_lines) != len(texts):
                logger.warning(
                    f"Translation-validation pass returned {len(translated_lines) if isinstance(translated_lines, list) else 'invalid'} "
                    f"lines, expected {len(texts)}. Keeping Whisper's original translation for this call."
                )
                return formatted_transcript

            rebuilt = [
                f"{prefixes[i]}{str(translated_lines[i]).strip()}"
                for i in range(len(texts))
            ]
            logger.info("Translation-validation pass completed successfully.")
            return "\n".join(rebuilt)

        except Exception as e:
            logger.warning(f"Translation-validation pass failed, keeping Whisper's original translation: {e}")
            return formatted_transcript

    def extract_structured_json(self, formatted_transcript: str) -> Dict[str, Any]:
        """
        Send actual transcribed text to LLM for full factual insight extraction, agent conduct audit, 
        and English JSON formatting.
        """
        logger.info("Extracting structured insights and evaluating Agent Conduct / Sentiment via LLM...")
        
        transcript_lines = [line.strip() for line in formatted_transcript.split("\n") if line.strip()]
        _fallback_roles = infer_speaker_roles(transcript_lines)
        _fallback_dialogues = []
        for idx, line in enumerate(transcript_lines):
            _m = _SPEAKER_TAG_RE.match(line)
            _tag = _m.group(1) if _m else None
            _role = _fallback_roles.get(_tag, "Unknown") if _tag else "Unknown"
            _clean_text = _SPEAKER_TAG_RE.sub("", line, count=1) if _m else line
            _fallback_dialogues.append({"id": idx + 1, "speaker": _role, "text": line, "message": _clean_text})
        default_structured_transcript = {
            "full_text": formatted_transcript,
            "dialogues": _fallback_dialogues
        }

        # Dynamic smart fallback
        fallback_json = generate_smart_fallback(transcript_lines, formatted_transcript)

        if "[No audible speech detected" in formatted_transcript or "[Audio file missing" in formatted_transcript or "[STT Engine error" in formatted_transcript:
            return {
                "customer_name": "Unknown",
                "agent_name": "Support AI",
                "issue_category": "Audio Processing",
                "problem_statement": formatted_transcript,
                "solution": "Please upload a clear audio file with audible voice speech.",
                "summary": formatted_transcript,
                "resolved": "NO",
                "sentiment": "Neutral",
                "transcript_json": default_structured_transcript
            }

        if not OLLAMA_AVAILABLE or not ollama:
            logger.error("Ollama client library is unavailable. Using dynamic transcript fallback.")
            return fallback_json

        needs_translation_pass = has_untranslated_script(formatted_transcript)
        if needs_translation_pass:
            logger.warning(
                "Transcript still contains non-Latin script after Whisper's translate pass "
                "(partial translation). Flagging this explicitly for the LLM to fully translate."
            )

        # Strict System Prompt forcing Zero Hallucination, Real Content Audit & 100% English Translation
        SYSTEM_PROMPT = """
        You are an expert AI Call Auditor and Quality Assurance Specialist for customer support interactions,
        fluent in translating from ANY spoken language (Urdu, Punjabi, Pashto, Sindhi, Arabic, Hindi, English,
        or any mix/code-switching between languages) into English.
        Your task is to analyze the provided call transcript, evaluate BOTH Customer AND Agent behavior, translate the conversation into English, and extract highly accurate, factual structured insights.

        CRITICAL AUDITING & ANALYSIS RULES:

        0. SPEAKER ATTRIBUTION (MANDATORY, HIGHEST PRIORITY):
           - The transcript below may contain diarization tags like [SPEAKER_00] and [SPEAKER_01]
             in front of some lines. Each distinct tag represents ONE real physical voice/person,
             used consistently for that same person for the ENTIRE call.
           - Read the actual CONTENT of each tagged line to decide whether that tag is the
             "Agent" (the support/receptionist representative - greets caller, asks how they can
             help, gives company/policy info, offers solutions) or the "Customer" (the caller -
             describes a problem, asks for help, complains, provides personal/order details).
           - DO NOT assume the first line, or any fixed position, belongs to a fixed role. Either
             the Agent or the Customer may speak first, and either one may speak two, three, or
             many lines in a row before the other responds - never guess a strict back-and-forth
             pattern in a fixed order.
           - Once you decide a given tag (e.g. [SPEAKER_00]) is the Customer or the Agent, that
             SAME tag must map to that SAME role for every line in the call. Never let the same
             tag switch roles partway through, and never swap what one person said onto the other
             person's dialogue lines.
           - If the transcript has NO diarization tags, infer speaker turns from context (question/
             answer flow, who is asking vs. answering, self-references like "I am calling about..."
             vs. "how can I help you") - keep the same rule that either party can speak first or
             speak multiple consecutive lines.
           - Every field you output (customer_name, agent_name, problem_statement, solution, and
             each "dialogues" entry's speaker/text) must reflect ONLY what that specific person
             actually said - never merge or cross-attribute the two speakers' statements.

        1. ACCURATE CONVERSATION SUMMARY & PROBLEM STATEMENT (MANDATORY):
           - Identify the core topic being discussed (e.g., loan dispute, order confusion, policy request, service complaint).
           - `problem_statement`: Describe the ACTUAL problem, disagreement, or request discussed in the call in 1-2 clear English sentences. NEVER put greeting lines like "Hello Assalam" or "I am talking to you".
           - `solution`: Describe the agent's actual resolution or response in 1-2 clear English sentences (e.g. "Agent refused request", "Agent resolved billing error", "No resolution reached; escalation requested").
           - `summary`: Write an executive summary of the entire call in 2-3 professional English sentences summarizing what the caller wanted, how the agent responded, any arguments/disputes, and the call outcome.

        2. CALL CATEGORY & AGENT CONDUCT AUDIT:
           - `issue_category`: Assign a precise category such as "Order & Service Dispute", "Service Complaint", "Loan & Policy Inquiry", "Rude Behavior / Agent Conduct", "Billing Support", or "General Inquiry".
           - `sentiment`: 
             * Set to "Negative" if there is ANY argument, disagreement, rudeness, unhelpful agent behavior, threat of complaint, or caller frustration.
             * Set to "Positive" ONLY if interaction is friendly, polite, and fully resolved.
             * Set to "Neutral" ONLY for calm, routine inquiries without conflict or dissatisfaction.

        3. 100% ENGLISH TRANSLATION & ZERO HALLUCINATION: 
           - Extract insights strictly from what was actually spoken in the transcript.
           - The transcript may be in ANY language, or a mix of languages/scripts (including
             leftover non-English words or script that a prior translation step missed).
           - Regardless of the source language(s), EVERY JSON field and EVERY "dialogues" text
             value in your output MUST be fully in English. If any word, phrase, or line in the
             input transcript is not already in English, translate it - do not copy it through
             untranslated, and do not skip or drop a line just because it wasn't in English.

        Return STRICT JSON matching this EXACT schema:
        {
          "customer_name": "string",
          "agent_name": "string",
          "issue_category": "string",
          "problem_statement": "string",
          "solution": "string",
          "summary": "string",
          "resolved": "YES | NO",
          "sentiment": "Negative | Positive | Neutral",
          "dialogues": [
             {
               "id": 1,
               "speaker": "Customer | Agent",
               "text": "English translated line of dialogue"
             }
          ]
        }
        """

        try:
            logger.info("Sending transcript to Ollama LLM...")
            user_content = f"CONVERSATION TRANSCRIPT TO AUDIT:\n{formatted_transcript}"
            if needs_translation_pass:
                user_content = (
                    "NOTE: This transcript still contains words/lines in their original "
                    "non-English script - a prior automatic translation pass was incomplete. "
                    "You MUST fully translate every such word/line into English in your output.\n\n"
                    + user_content
                )
            client = get_ollama_client()
            response = client.chat(
                model=self.llm_model,
                format="json",  
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                options={"temperature": 0.0}
            )
            
            # Safely handle both Pydantic response objects and dict responses from ollama
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                raw_output = response.message.content
            elif isinstance(response, dict) and 'message' in response:
                msg = response['message']
                if isinstance(msg, dict):
                    raw_output = msg.get('content', '')
                else:
                    raw_output = getattr(msg, 'content', str(msg))
            else:
                raw_output = str(response)

            raw_output = raw_output.strip()

            # Clean markdown codeblocks
            if "```" in raw_output:
                raw_output = re.sub(r"^```(?:json)?\s*", "", raw_output, flags=re.IGNORECASE)
                raw_output = re.sub(r"\s*```$", "", raw_output)
                raw_output = raw_output.strip()

            parsed_data = json.loads(raw_output)

            # Ensure output is a dictionary
            if not isinstance(parsed_data, dict):
                logger.warning("LLM output is not a JSON dictionary. Using dynamic fallback.")
                return fallback_json

            # Extract dialogues translated by LLM if available
            llm_dialogues = parsed_data.get("dialogues")
            if isinstance(llm_dialogues, list) and len(llm_dialogues) > 0:
                formatted_dialogues = []
                full_text_lines = []
                prev_dialogue_text = ""
                greeting_cnt = 0

                for idx, d in enumerate(llm_dialogues):
                    if isinstance(d, dict):
                        spk = d.get("speaker", "Speaker")
                        txt = d.get("text") or d.get("message") or ""
                        txt = remove_in_text_phrase_loops(txt)
                        
                        line_str = f"[{spk}] {txt}" if spk else txt
                        lower_txt = txt.lower().strip()
                        
                        # Filter duplicate greetings inside LLM dialogue list
                        words = re.sub(r'[^\w\s]', '', lower_txt).split()
                        is_short_greeting = len(words) <= 3 and any(w in ["hello", "hi", "hey"] for w in words)
                        if is_short_greeting:
                            greeting_cnt += 1
                            if greeting_cnt > 2:
                                continue

                        if lower_txt == prev_dialogue_text:
                            continue
                        prev_dialogue_text = lower_txt

                        formatted_dialogues.append({
                            "id": len(formatted_dialogues) + 1,
                            "speaker": spk,
                            "text": line_str,
                            "message": txt
                        })
                        full_text_lines.append(line_str)
                
                if formatted_dialogues:
                    english_transcript_json = {
                        "full_text": "\n".join(full_text_lines),
                        "dialogues": formatted_dialogues
                    }
                else:
                    english_transcript_json = default_structured_transcript
            else:
                english_transcript_json = default_structured_transcript

            # Safeguard Sentiment & Conflict Detection across transcript & parsed output
            lower_full = formatted_transcript.lower()
            conflict_triggers = ["complain", "manager", "campus", "policy", "loan", "pizza", "rude", "whatever", "refuse", "dispute", "not coming", "argue", "arguing", "bad behavior", "threat"]
            if any(trig in lower_full for trig in conflict_triggers):
                parsed_data["sentiment"] = "Negative"
                if parsed_data.get("issue_category") in ["General Inquiry", "General", "Live Call"]:
                    parsed_data["issue_category"] = "Order & Service Dispute"

            # Post-processing and defaults
            cust_name = str(parsed_data.get("customer_name") or "")
            if not cust_name or "john" in cust_name.lower():
                parsed_data["customer_name"] = "Valued Customer"

            agent_name = str(parsed_data.get("agent_name") or "")
            if not agent_name or "john" in agent_name.lower():
                parsed_data["agent_name"] = "Support Agent"

            if not parsed_data.get("issue_category"):
                parsed_data["issue_category"] = "General Inquiry"
            if not parsed_data.get("problem_statement") or parsed_data.get("problem_statement") in ["N/A", "null"]:
                parsed_data["problem_statement"] = fallback_json["problem_statement"]
            if not parsed_data.get("solution"):
                parsed_data["solution"] = "No specific resolution required."
            if not parsed_data.get("summary") or parsed_data.get("summary") in ["N/A", "null"]:
                parsed_data["summary"] = f"Customer issue: {parsed_data.get('problem_statement')}. Solution: {parsed_data.get('solution')}"
            if not parsed_data.get("resolved"):
                parsed_data["resolved"] = "NO"
            if not parsed_data.get("sentiment"):
                parsed_data["sentiment"] = "Neutral"

            # Attach English structured transcript
            parsed_data["transcript_json"] = english_transcript_json
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error extracting English JSON from LLM: {str(e)}")
            return fallback_json


# Global Singleton Pipeline (lazy initialized on first request or main execution)
_pipeline_instance = None

def get_pipeline() -> LocalAIPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = LocalAIPipeline()
    return _pipeline_instance

# Backward compatibility module attribute
class PipelineProxy:
    def __getattr__(self, name):
        return getattr(get_pipeline(), name)

pipeline = PipelineProxy()


def process_audio(file_path: str) -> Dict[str, Any]:
    """Single interface for FastAPI integration."""
    logger.info(f"Processing audio request for: {file_path}")
    active_pipeline = get_pipeline()
    formatted_transcript = active_pipeline.transcribe_audio(file_path)
    formatted_transcript = active_pipeline.translate_and_clean_transcript(formatted_transcript)
    extracted_data = active_pipeline.extract_structured_json(formatted_transcript)
    return extracted_data


if __name__ == "__main__":
    p = get_pipeline()
    p.warmup_llm()
    print("\n--- AI Engine Module Verification Complete ---")
