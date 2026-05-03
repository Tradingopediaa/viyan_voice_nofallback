import os
import base64
import uuid
from pathlib import Path

import runpod
import torch
import soundfile as sf

WORKDIR = Path("/workspace")
OUT_DIR = WORKDIR / "out"
OUT_DIR.mkdir(exist_ok=True)

# Viyan does NOT clone the user's voice.
# Viyan uses a designed artificial voice identity.
VIYAN_VOICE_INSTRUCT = os.getenv(
    "VIYAN_VOICE_INSTRUCT",
    "male, young adult, low pitch, indian accent"
)

OMNI_ASR_CARD = os.getenv("VIYAN_OMNI_ASR_CARD", "omniASR_CTC_1B_v2")
SENSE_MODEL = os.getenv("VIYAN_SENSE_MODEL", "FunAudioLLM/SenseVoiceSmall")
OMNIVOICE_MODEL = os.getenv("VIYAN_OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

_asr = None
_sense = None
_tts = None


def gpu_status():
    info = {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
        })
    return info


def load_asr():
    global _asr
    if _asr is None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        _asr = ASRInferencePipeline(model_card=OMNI_ASR_CARD)
    return _asr


def load_sense():
    global _sense
    if _sense is None:
        from funasr import AutoModel
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _sense = AutoModel(
            model=SENSE_MODEL,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            hub="hf",
        )
    return _sense


def load_tts():
    global _tts
    if _tts is None:
        from omnivoice import OmniVoice
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _tts = OmniVoice.from_pretrained(
            OMNIVOICE_MODEL,
            device_map=device,
            dtype=dtype,
        )
    return _tts


def save_b64_audio(audio_b64: str, suffix=".wav") -> str:
    path = OUT_DIR / f"input_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(base64.b64decode(audio_b64))
    return str(path)


def wav_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def core_nonverbals_to_text(text: str, delivery: dict) -> str:
    # Not a voice planner. Core decides. Mouth Motor only renders explicit Core commands.
    nonverbals = delivery.get("nonverbal") or []
    prefix = []
    suffix = []

    for nv in nonverbals:
        typ = str(nv.get("type", "")).lower().strip()
        pos = str(nv.get("position", "before_start")).lower().strip()

        tag = None
        if typ in {"laugh", "laughter", "soft_laugh"}:
            tag = "[laughter]"
        elif typ in {"sigh", "soft_sigh", "breath", "soft_breath"}:
            tag = "[sigh]"

        if not tag:
            continue

        if pos in {"before", "start", "before_start"}:
            prefix.append(tag)
        elif pos in {"after", "end", "after_end"}:
            suffix.append(tag)

    return " ".join(prefix + [text] + suffix).strip()


def handle_health():
    return {
        "ok": True,
        "stack": {
            "fallback": False,
            "ear_words": OMNI_ASR_CARD,
            "ear_emotion": SENSE_MODEL,
            "mouth": OMNIVOICE_MODEL,
            "mouth_mode": "voice_design",
            "voice_instruct": VIYAN_VOICE_INSTRUCT,
            "user_voice_clone": False,
        },
        "gpu": gpu_status(),
    }


def handle_preload():
    load_asr()
    load_sense()
    load_tts()
    return {
        "ok": True,
        "loaded": ["omni_asr", "sensevoice", "omnivoice"],
        "gpu": gpu_status(),
    }


def handle_ear(inp: dict):
    audio_b64 = inp.get("audio_base64")
    if not audio_b64:
        return {"ok": False, "error": "audio_base64 missing"}

    lang = inp.get("lang")
    audio_path = save_b64_audio(audio_b64)

    asr = load_asr()
    try:
        if lang:
            words = asr.transcribe([audio_path], lang=[lang], batch_size=1)
        else:
            words = asr.transcribe([audio_path], batch_size=1)
    except TypeError:
        words = asr.transcribe([audio_path], batch_size=1)

    sense = load_sense()
    sense_result = sense.generate(
        input=audio_path,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
    )

    return {
        "ok": True,
        "ear_packet": {
            "text": words[0] if isinstance(words, list) and words else words,
            "language_hint": lang,
            "emotion_audio_event_evidence": sense_result,
            "note": "Core decides final emotion and meaning. Ear only sends evidence.",
        },
        "gpu": gpu_status(),
    }


def handle_mouth(inp: dict):
    spoken_text = (inp.get("spoken_text") or inp.get("text") or "").strip()
    if not spoken_text:
        return {"ok": False, "error": "spoken_text/text missing"}

    delivery = inp.get("delivery") or {}
    render_text = core_nonverbals_to_text(spoken_text, delivery)

    # Core may override instruction only if it is defining Viyan's own voice identity,
    # not user-cloning. Default is fixed artificial Viyan voice.
    voice_instruct = (
        inp.get("voice_instruct")
        or inp.get("instruct")
        or VIYAN_VOICE_INSTRUCT
    )

    speed = float(delivery.get("speed", 1.0))
    language_id = inp.get("language_id") or inp.get("language")

    tts = load_tts()

    # Same model, same mouth. These tries are API compatibility only, not fallback.
    attempts = []
    base = {"text": render_text, "instruct": voice_instruct}
    if language_id:
        attempts.append({**base, "lang": language_id, "speed": speed})
        attempts.append({**base, "language_id": language_id, "speed": speed})
        attempts.append({**base, "lang": language_id})
        attempts.append({**base, "language_id": language_id})
    attempts.append({**base, "speed": speed})
    attempts.append(base)

    last_err = None
    audio = None
    used_kwargs = None
    for kwargs in attempts:
        try:
            audio = tts.generate(**kwargs)
            used_kwargs = kwargs
            break
        except TypeError as e:
            last_err = e
            continue

    if audio is None:
        return {"ok": False, "error": f"OmniVoice generate API mismatch: {last_err}"}

    out_path = OUT_DIR / f"viyan_{uuid.uuid4().hex}.wav"
    sf.write(str(out_path), audio[0], 24000)

    return {
        "ok": True,
        "audio_base64": wav_to_b64(str(out_path)),
        "sample_rate": 24000,
        "render_text": render_text,
        "voice_instruct": voice_instruct,
        "mouth_mode": "voice_design",
        "user_voice_clone": False,
        "used_generate_keys": list(used_kwargs.keys()) if used_kwargs else [],
        "model": OMNIVOICE_MODEL,
        "gpu": gpu_status(),
    }


def handler(job):
    inp = job.get("input") or {}
    mode = inp.get("mode", "health")

    try:
        if mode == "health":
            return handle_health()
        if mode == "preload":
            return handle_preload()
        if mode == "ear":
            return handle_ear(inp)
        if mode == "mouth":
            return handle_mouth(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}
    except Exception as e:
        return {
            "ok": False,
            "error": repr(e),
            "gpu": gpu_status(),
        }


runpod.serverless.start({"handler": handler})
