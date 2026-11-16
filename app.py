"""
TranscribeStudio — единый бэкенд
Транскрибация (Whisper / wav2vec2) + перевод (NLLB-200)
"""

import os, sys, json, uuid, shutil, threading, webbrowser, traceback, time, subprocess, re, math
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import CORS
import ffmpeg
import numpy as np

BASE_DIR     = Path(__file__).parent
PROJECTS_DIR = BASE_DIR / "projects"
UPLOAD_DIR   = BASE_DIR / "uploads"
EXPORT_DIR   = BASE_DIR / "exports"
MODELS_DIR   = BASE_DIR / "models"

SEP_DIR      = BASE_DIR / "separated"
SETTINGS_FILE = BASE_DIR / "settings.json"

for d in [PROJECTS_DIR, UPLOAD_DIR, EXPORT_DIR, MODELS_DIR, SEP_DIR]:
    d.mkdir(exist_ok=True)

_local_ff = BASE_DIR / "ffmpeg" / "ffmpeg.exe"
if _local_ff.exists():
    os.environ["PATH"] = str(_local_ff.parent) + os.pathsep + os.environ.get("PATH", "")
    print(f"[INFO] FFmpeg: portable ({_local_ff})")
else:
    print("[INFO] FFmpeg: system PATH")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
# Раньше здесь стоял жёсткий лимит 4 ГБ — двухчасовые записи в высоком
# качестве легко весят 10-15+ ГБ, и загрузка такого файла просто обрывалась
# с ошибкой 413, до того как файл вообще успевал сохраниться на диск.
# Приложение локальное (слушает только 127.0.0.1), так что искусственный
# лимит не даёт никакой защиты — реальным ограничителем остаётся свободное
# место на диске. Оставляем большой, но не бесконечный лимит на случай
# полностью некорректного запроса.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024 * 1024  # 64 ГБ

@app.errorhandler(413)
def err_413(e):
    return jsonify({"error": "Файл слишком большой (лимит 64 ГБ)"}), 413

@app.errorhandler(404)
def err_404(e):
    return jsonify({"error": "Не найдено"}), 404

@app.errorhandler(500)
def err_500(e):
    return jsonify({"error": f"Внутренняя ошибка: {e}"}), 500

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_hf_cache:   dict = {}
_cache_lock  = threading.Lock()

_download_status: dict = {}
_dl_lock = threading.Lock()

_translate_status: dict = {}
_tr_lock = threading.Lock()

_sep_status: dict = {}
_sep_lock = threading.Lock()

# Распознавание речи (ASR) — фоновый прогресс/отмена.
# Имена нарочно не "_tr_*", т.к. это уже занято переводом (NLLB) выше.
_asr_status: dict = {}
_asr_lock = threading.Lock()
_asr_cancel_flags: dict = {}   # pid -> bool, проверяется между чанками

class TranscribeCancelled(Exception):
    """Поднимается изнутри transcribe(), когда пользователь нажал "Отмена"."""
    pass

# openai-whisper (arch="whisper") сам решает, как бить аудио на 30-секундные
# окна, и это даёт лучшее качество (контекст между окнами), чем наше ручное
# нарезание. Чтобы не терять это качество, но всё равно получать честный
# прогресс, подменяем его внутренний tqdm-прогресс-бар на свой — он и так
# считает обработанные "кадры" аудио, нам остаётся только читать эти числа.
_whisper_tqdm_lock = threading.Lock()
_whisper_progress_local = threading.local()

class _ProgressTqdm:
    """Совместимая по интерфейсу замена tqdm.tqdm, используемая внутри
    whisper.transcribe(). Не рисует прогресс-бар в консоли — вместо этого
    прокидывает проценты в progress_cb текущего потока и даёт возможность
    прервать распознавание (TranscribeCancelled) между обновлениями."""
    def __init__(self, *a, total=None, **kw):
        self.total = total or 1
        self.n = 0
    def update(self, n=1):
        self.n += n
        cb = getattr(_whisper_progress_local, "cb", None)
        if cb:
            try: cb(min(99.0, 100.0 * self.n / max(1, self.total)), "Распознавание")
            except Exception: pass
        cancel = getattr(_whisper_progress_local, "cancel", None)
        if cancel and cancel():
            raise TranscribeCancelled()
    def set_description(self, *a, **kw): pass
    def set_description_str(self, *a, **kw): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False

DEMUCS_MODELS = {
    "htdemucs": {
        "label": "HTDemucs · универсальный · ~85 МБ (рекомендуется)",
        "size_mb": 85,
    },
    "htdemucs_ft": {
        "label": "HTDemucs Fine-tuned · лучшее качество · ~340 МБ",
        "size_mb": 340,
    },
    "mdx23c": {
        "label": "MDX23C · альтернатива · ~170 МБ",
        "size_mb": 170,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# КАТАЛОГ МОДЕЛЕЙ
# ─────────────────────────────────────────────────────────────────────────────
HF_MODELS = {
    # ── РУССКИЙ ──────────────────────────────────────────────────────────────
    "whisper-large-v3-ru": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "ru", "lang_name": "Русский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 RU ★ WER ~5% (рекомендуется)",
        "size_mb": 1550,
    },
    "w2v-ru-golos": {
        "id": "bond005/wav2vec2-large-ru-golos",
        "lang": "ru", "lang_name": "Русский",
        "arch": "wav2vec2",
        "label": "wav2vec2 RU Large · Golos — WER ~6%",
        "size_mb": 1260,
    },
    # ── АНГЛИЙСКИЙ ───────────────────────────────────────────────────────────
    "whisper-large-v3-en": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "en", "lang_name": "Английский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 EN ★ WER 2.7% (рекомендуется)",
        "size_mb": 1550,
    },
    "wav2vec2-en-lv60": {
        "id": "facebook/wav2vec2-large-960h-lv60-self",
        "lang": "en", "lang_name": "Английский",
        "arch": "wav2vec2",
        "label": "wav2vec2 EN Large LV60 — WER 1.9% (лучший EN)",
        "size_mb": 1260,
    },
    # ── НЕМЕЦКИЙ ─────────────────────────────────────────────────────────────
    "whisper-large-v3-de": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "de", "lang_name": "Немецкий",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 DE ★ WER ~5%",
        "size_mb": 1550,
    },
    "wav2vec2-de": {
        "id": "facebook/wav2vec2-large-xlsr-53-german",
        "lang": "de", "lang_name": "Немецкий",
        "arch": "wav2vec2",
        "label": "wav2vec2 DE Large XLSR — WER ~5%",
        "size_mb": 1260,
    },
    # ── ФРАНЦУЗСКИЙ ──────────────────────────────────────────────────────────
    "whisper-large-v3-fr": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "fr", "lang_name": "Французский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 FR ★ WER ~5%",
        "size_mb": 1550,
    },
    "w2v-fr-xlsr": {
        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
        "lang": "fr", "lang_name": "Французский",
        "arch": "wav2vec2",
        "label": "wav2vec2 FR XLSR-53 — WER ~9%, быстрее Whisper",
        "size_mb": 1260,
    },
    # ── ИСПАНСКИЙ ────────────────────────────────────────────────────────────
    "whisper-large-v3-es": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "es", "lang_name": "Испанский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 ES ★ WER ~5%",
        "size_mb": 1550,
    },
    "w2v-es-xlsr": {
        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
        "lang": "es", "lang_name": "Испанский",
        "arch": "wav2vec2",
        "label": "wav2vec2 ES XLSR-53 — WER ~8%, быстрее Whisper",
        "size_mb": 1260,
    },
    # ── УКРАИНСКИЙ ───────────────────────────────────────────────────────────
    "whisper-large-v3-uk": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "uk", "lang_name": "Украинский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 UK ★ WER ~6%",
        "size_mb": 1550,
    },
    "w2v-uk-xlsr": {
        "id": "theodor1289/wav2vec2-large-xlsr-53-ukrainian",
        "lang": "uk", "lang_name": "Украинский",
        "arch": "wav2vec2",
        "label": "wav2vec2 UK XLSR-53 — WER ~12%, легче Whisper",
        "size_mb": 1260,
    },
    # ── МНОГОЯЗЫЧНЫЕ ─────────────────────────────────────────────────────────
    "whisper-large-v3": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "multi", "lang_name": "Многоязычный",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 ★ 100+ языков, WER 2.7-6%",
        "size_mb": 1550,
    },
    "whisper-medium": {
        "id": "Systran/faster-whisper-medium",
        "lang": "multi", "lang_name": "Многоязычный",
        "arch": "faster_whisper",
        "label": "Whisper Medium · 100 языков, WER ~8%",
        "size_mb": 780,
    },
    "whisper-small": {
        "id": "Systran/faster-whisper-small",
        "lang": "multi", "lang_name": "Многоязычный",
        "arch": "faster_whisper",
        "label": "Whisper Small · быстрый, WER ~12%",
        "size_mb": 250,
    },
    "whisper-large-v3-turbo": {
        "id": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "lang": "multi", "lang_name": "Многоязычный",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 Turbo ★ 100+ языков · в 5× быстрее обычного Large",
        "size_mb": 1600,
    },
    "whisper-base": {
        "id": "Systran/faster-whisper-base",
        "lang": "multi", "lang_name": "Многоязычный",
        "arch": "faster_whisper",
        "label": "Whisper Base · самый быстрый, WER ~20%",
        "size_mb": 75,
    },
    # ── ИТАЛЬЯНСКИЙ ──────────────────────────────────────────────────────────
    "whisper-large-v3-it": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "it", "lang_name": "Итальянский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 IT ★ WER ~5%",
        "size_mb": 1550,
    },
    "w2v-it-xlsr": {
        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        "lang": "it", "lang_name": "Итальянский",
        "arch": "wav2vec2",
        "label": "wav2vec2 IT XLSR-53 — WER ~10%, быстрее Whisper",
        "size_mb": 1260,
    },
    # ── ПОРТУГАЛЬСКИЙ ────────────────────────────────────────────────────────
    "whisper-large-v3-pt": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "pt", "lang_name": "Португальский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 PT ★ WER ~6%",
        "size_mb": 1550,
    },
    "w2v-pt-xlsr": {
        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
        "lang": "pt", "lang_name": "Португальский",
        "arch": "wav2vec2",
        "label": "wav2vec2 PT XLSR-53 — WER ~11%, быстрее Whisper",
        "size_mb": 1260,
    },
    # ── ПОЛЬСКИЙ ─────────────────────────────────────────────────────────────
    "whisper-large-v3-pl": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "pl", "lang_name": "Польский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 PL ★ WER ~6%",
        "size_mb": 1550,
    },
    "w2v-pl-xlsr": {
        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
        "lang": "pl", "lang_name": "Польский",
        "arch": "wav2vec2",
        "label": "wav2vec2 PL XLSR-53 — WER ~13%, быстрее Whisper",
        "size_mb": 1260,
    },
    # ── ТУРЕЦКИЙ ─────────────────────────────────────────────────────────────
    "whisper-large-v3-tr": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "tr", "lang_name": "Турецкий",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 TR ★ WER ~7%",
        "size_mb": 1550,
    },
    # ── НИДЕРЛАНДСКИЙ ────────────────────────────────────────────────────────
    "whisper-large-v3-nl": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "nl", "lang_name": "Нидерландский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 NL ★ WER ~6%",
        "size_mb": 1550,
    },
    "w2v-nl-xlsr": {
        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
        "lang": "nl", "lang_name": "Нидерландский",
        "arch": "wav2vec2",
        "label": "wav2vec2 NL XLSR-53 — WER ~12%, быстрее Whisper",
        "size_mb": 1260,
    },
    # ── ЯПОНСКИЙ ─────────────────────────────────────────────────────────────
    "whisper-large-v3-ja": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "ja", "lang_name": "Японский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 JA ★ WER ~7%",
        "size_mb": 1550,
    },
    # ── КИТАЙСКИЙ ────────────────────────────────────────────────────────────
    "whisper-large-v3-zh": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "zh", "lang_name": "Китайский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 ZH ★ WER ~9%",
        "size_mb": 1550,
    },
    # ── КОРЕЙСКИЙ ────────────────────────────────────────────────────────────
    "whisper-large-v3-ko": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "ko", "lang_name": "Корейский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 KO ★ WER ~8%",
        "size_mb": 1550,
    },
    # ── АРАБСКИЙ ─────────────────────────────────────────────────────────────
    "whisper-large-v3-ar": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "ar", "lang_name": "Арабский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 AR ★ WER ~10%",
        "size_mb": 1550,
    },
    "w2v-ar-xlsr": {
        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        "lang": "ar", "lang_name": "Арабский",
        "arch": "wav2vec2",
        "label": "wav2vec2 AR XLSR-53 — WER ~15%, быстрее Whisper",
        "size_mb": 1260,
    },
    # ── ХИНДИ ────────────────────────────────────────────────────────────────
    "whisper-large-v3-hi": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "hi", "lang_name": "Хинди",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 HI ★ WER ~8%",
        "size_mb": 1550,
    },
    # ── ШВЕДСКИЙ ─────────────────────────────────────────────────────────────
    "whisper-large-v3-sv": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "sv", "lang_name": "Шведский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 SV ★ WER ~6%",
        "size_mb": 1550,
    },
    # ── ЧЕШСКИЙ ──────────────────────────────────────────────────────────────
    "whisper-large-v3-cs": {
        "id": "Systran/faster-whisper-large-v3",
        "lang": "cs", "lang_name": "Чешский",
        "arch": "faster_whisper",
        "label": "Whisper Large-v3 CS ★ WER ~7%",
        "size_mb": 1550,
    },
    # ── ПЕРЕВОД ──────────────────────────────────────────────────────────────
    "nllb-600m": {
        "id": "facebook/nllb-200-distilled-600M",
        "lang": "translate", "lang_name": "Перевод",
        "arch": "nllb",
        "label": "NLLB-200 distilled 600M · 200 языков · базовое качество",
        "size_mb": 2400,
    },
    "nllb-1.3b": {
        "id": "facebook/nllb-200-distilled-1.3B",
        "lang": "translate", "lang_name": "Перевод",
        "arch": "nllb",
        "label": "NLLB-200 distilled 1.3B · 200 языков · хорошее качество ★",
        "size_mb": 5200,
    },
    "nllb-3.3b": {
        "id": "facebook/nllb-200-3.3B",
        "lang": "translate", "lang_name": "Перевод",
        "arch": "nllb",
        "label": "NLLB-200 3.3B · 200 языков · лучшее качество (нужно 8+ ГБ RAM)",
        "size_mb": 13000,
    },
    "opus-en-ru": {
        "id": "Helsinki-NLP/opus-mt-tc-big-en-ru",
        "lang": "translate", "lang_name": "Перевод",
        "arch": "marian",
        "src_lang": "en", "tgt_lang": "ru",
        "label": "Helsinki opus-mt EN→RU · 1.3 ГБ · быстрый, хорошее качество",
        "size_mb": 1300,
    },
    "opus-ru-en": {
        "id": "Helsinki-NLP/opus-mt-ru-en",
        "lang": "translate", "lang_name": "Перевод",
        "arch": "marian",
        "src_lang": "ru", "tgt_lang": "en",
        "label": "Helsinki opus-mt RU→EN · 300 МБ · быстрый",
        "size_mb": 300,
    },
    "opus-en-de": {
        "id": "Helsinki-NLP/opus-mt-en-de",
        "lang": "translate", "lang_name": "Перевод",
        "arch": "marian",
        "src_lang": "en", "tgt_lang": "de",
        "label": "Helsinki opus-mt EN→DE · 300 МБ",
        "size_mb": 300,
    },
    "opus-en-fr": {
        "id": "Helsinki-NLP/opus-mt-en-fr",
        "lang": "translate", "lang_name": "Перевод",
        "arch": "marian",
        "src_lang": "en", "tgt_lang": "fr",
        "label": "Helsinki opus-mt EN→FR · 300 МБ",
        "size_mb": 300,
    },
}

NLLB_LANGS = {
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "pl": "pol_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
}

# ─────────────────────────────────────────────────────────────────────────────
# Утилиты времени
# ─────────────────────────────────────────────────────────────────────────────
def s2ts(s: float) -> str:
    ms = int(round(s * 1000)) % 1000
    t  = int(s)
    return f"{t//3600:02d}:{(t//60)%60:02d}:{t%60:02d},{ms:03d}"

def ts2s(ts: str) -> float:
    try:
        ts = ts.replace(",", ".")
        h, m, s = ts.split(":")
        return int(h)*3600 + int(m)*60 + float(s)
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Управление моделями
# ─────────────────────────────────────────────────────────────────────────────
def _shared_dir(hf_id: str) -> Path:
    """Одна папка для всех ключей с одинаковым hf_id — скачивается один раз."""
    return MODELS_DIR / "_shared" / hf_id.replace("/", "--")

def model_path(key: str) -> Path:
    info = HF_MODELS.get(key, {})
    arch = info.get("arch", "")
    if arch in ("faster_whisper", "whisper_hf", "wav2vec2", "nllb"):
        return _shared_dir(info["id"])
    return MODELS_DIR / key

def _shared_users(key: str) -> list:
    """Список других ключей, которые используют ту же папку (тот же hf_id)."""
    info = HF_MODELS.get(key, {})
    hf_id = info.get("id"); arch = info.get("arch")
    if arch not in ("faster_whisper", "whisper_hf", "wav2vec2", "nllb") or not hf_id:
        return []
    return [k for k, v in HF_MODELS.items()
            if k != key and v.get("id") == hf_id and v.get("arch") == arch]

def _find_model_dir(key: str) -> Path:
    """Возвращает реальный путь к модели — новая shared-папка или старая per-key."""
    new_p = model_path(key)
    if (new_p / "config.json").exists():
        return new_p
    old_p = MODELS_DIR / key
    if (old_p / "config.json").exists():
        return old_p
    return new_p   # не найдена — вернём новый путь, ошибка будет понятной

def _model_weights_present(local: Path) -> bool:
    """config.json alone is NOT proof a model is usable — it's a small file
    that downloads in a fraction of a second, while the actual weight file
    can be gigabytes and take much longer. If the download was interrupted
    (network drop, disk full, app/PC closed mid-download, antivirus
    quarantine, etc.) config.json can exist while the weights never
    finished — is_downloaded() would then wrongly report "downloaded",
    letting the user pick the model and only fail later at load time with
    a confusing "no file named model.safetensors/pytorch_model.bin" error."""
    if (local / "model.safetensors").exists():
        return True
    if (local / "pytorch_model.bin").exists():
        return True
    if (local / "model.bin").exists():   # CTranslate2 (faster-whisper) weight file
        return True
    idx = local / "model.safetensors.index.json"
    if idx.exists() and list(local.glob("model-*-of-*.safetensors")):
        return True
    idx2 = local / "pytorch_model.bin.index.json"
    if idx2.exists() and list(local.glob("pytorch_model-*-of-*.bin")):
        return True
    return False

def is_downloaded(key: str) -> bool:
    info = HF_MODELS.get(key)
    if not info: return False
    arch = info["arch"]
    if arch == "whisper":
        cache = Path(os.path.expanduser("~")) / ".cache" / "whisper"
        name  = info["id"].replace("whisper-", "")
        return cache.exists() and any(cache.glob(f"{name}*"))
    # Проверяем новую общую папку
    if (model_path(key) / "config.json").exists() and _model_weights_present(model_path(key)):
        return True
    # Обратная совместимость: старая папка per-key
    old_p = MODELS_DIR / key
    return (old_p / "config.json").exists() and _model_weights_present(old_p)

def _dl_with_retry(fn, key: str, max_retries: int = 8):
    delay = 3
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            is_net = any(x in msg for x in (
                "10054", "10060", "connectionreset", "connection",
                "timeout", "closed", "ssl", "eof", "remote", "broken",
                "reset", "refused", "network",
            ))
            if is_net and attempt < max_retries:
                print(f"[DL] Net error (attempt {attempt}/{max_retries}): {str(e)[:80]}")
                time.sleep(delay); delay = min(delay * 2, 60); continue
            raise

def _get_total_bytes(hf_id: str, ignore: list) -> int:
    try:
        import fnmatch
        from huggingface_hub import HfApi
        siblings = HfApi().model_info(hf_id, files_metadata=True).siblings or []
        return sum(s.size for s in siblings if s.size and not any(
            fnmatch.fnmatch(s.rfilename, pat) for pat in ignore))
    except Exception as e:
        print(f"[DL] Cannot get total size: {e}")
        return 0

def _folder_size(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0

def _monitor_progress(key: str, local: Path, total_bytes: int):
    while True:
        with _dl_lock:
            if _download_status.get(key, {}).get("status") != "downloading":
                break
        size = _folder_size(local)
        pct  = min(99, int(size / total_bytes * 100)) if total_bytes > 0 else 0
        with _dl_lock:
            if key in _download_status:
                _download_status[key]["downloaded_bytes"] = size
                _download_status[key]["pct"]              = pct
        time.sleep(1.5)

def _clean_hf_cache(local: Path):
    hf_cache = local / ".cache"
    if hf_cache.exists():
        shutil.rmtree(hf_cache)
    safe = local / "model.safetensors"
    pytorch_bin = local / "pytorch_model.bin"
    if safe.exists() and pytorch_bin.exists():
        pytorch_bin.unlink()
    pt_shards = list(local.glob("pytorch_model-*.bin"))
    sf_shards = list(local.glob("model-*.safetensors"))
    if sf_shards and pt_shards:
        for shard in pt_shards:
            shard.unlink()

def _find_shared_model(hf_id: str, exclude_key: str) -> Path | None:
    for k, info in HF_MODELS.items():
        if k == exclude_key: continue
        if info["id"] == hf_id and info["arch"] in ("faster_whisper", "whisper_hf", "wav2vec2"):
            candidate = model_path(k)
            if (candidate / "config.json").exists():
                return candidate
    return None

def _bg_download(key: str):
    info  = HF_MODELS[key]
    hf_id = info["id"]
    arch  = info["arch"]
    local = model_path(key)   # для HF-моделей — общая папка

    with _dl_lock:
        _download_status[key] = {
            "status": "downloading", "error": None,
            "downloaded_bytes": 0, "total_bytes": 0, "pct": 0,
        }

    try:
        if arch == "whisper":
            import whisper
            name = hf_id.replace("whisper-", "")
            _dl_with_retry(lambda: whisper.load_model(name), key)

        elif arch in ("faster_whisper", "whisper_hf", "wav2vec2", "nllb"):
            from huggingface_hub import snapshot_download

            # Если общая папка уже полностью скачана другим ключом — просто
            # помечаем готово. ВАЖНО: проверяем не только config.json (он
            # маленький и мог остаться от прерванной закачки), а именно
            # наличие файла весов — иначе сломанная папка так и останется
            # сломанной навсегда, потому что и is_downloaded(), и этот же
            # блок будут считать её готовой.
            if (local / "config.json").exists() and _model_weights_present(local):
                sz = _folder_size(local)
                with _dl_lock:
                    _download_status[key] = {
                        "status": "done", "error": None,
                        "downloaded_bytes": sz, "total_bytes": sz, "pct": 100,
                    }
                print(f"[DL] Shared folder reused: {key} -> {local.name}")
                return
            if (local / "config.json").exists():
                print(f"[DL] Shared folder for {key} looked downloaded but weights "
                      f"are missing/incomplete — repairing: {local}")

            local.mkdir(parents=True, exist_ok=True)
            ignore = ["flax_model*", "tf_model*", "rust_model*",
                      "*.msgpack", "*.ot", "*.h5",
                      "pytorch_model.bin", "pytorch_model-*.bin"]

            total_bytes = _get_total_bytes(hf_id, ignore)
            with _dl_lock:
                _download_status[key]["total_bytes"] = total_bytes
            threading.Thread(
                target=_monitor_progress, args=(key, local, total_bytes), daemon=True
            ).start()

            def _dl():
                snapshot_download(
                    repo_id=hf_id, local_dir=str(local),
                    local_dir_use_symlinks=False, ignore_patterns=ignore,
                )
            _dl_with_retry(_dl, key)
            _clean_hf_cache(local)

            if arch != "faster_whisper" and not list(local.glob("model*.safetensors")):
                ignore_no_pt = [p for p in ignore if "pytorch_model" not in p]
                def _dl2():
                    snapshot_download(
                        repo_id=hf_id, local_dir=str(local),
                        local_dir_use_symlinks=False, ignore_patterns=ignore_no_pt,
                    )
                _dl_with_retry(_dl2, key)
                _clean_hf_cache(local)

            if arch == "whisper_hf":
                from transformers import AutoProcessor
                proc = AutoProcessor.from_pretrained(str(local))
                proc.save_pretrained(str(local))
            elif arch == "wav2vec2":
                from transformers import AutoProcessor, Wav2Vec2Processor
                try:
                    proc = AutoProcessor.from_pretrained(str(local))
                except Exception as e:
                    if any(x in str(e).lower() for x in ("kenlm","pyctcdecode","ctcdecode","import","module")):
                        proc = Wav2Vec2Processor.from_pretrained(str(local))
                    else:
                        raise
                proc.save_pretrained(str(local))

        with _dl_lock:
            _download_status[key] = {"status": "done", "error": None, "pct": 100}
        print(f"[DL] Done: {key}")

    except Exception as e:
        traceback.print_exc()
        with _dl_lock:
            _download_status[key] = {"status": "error", "error": str(e)}


def _torch_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _evict_hf_cache(keep_key: str | None = None):
    """Выгружает из памяти все загруженные модели транскрипции, кроме
    (опционально) одной. Без этого каждая модель, которую хоть раз
    выбрали в сессии, оставалась в _hf_cache и на GPU НАВСЕГДА — на
    видеокартах с небольшим объёмом памяти (6 ГБ и меньше) достаточно
    было один раз попробовать 2 разные модели, чтобы поймать
    "CUDA out of memory", даже если каждая модель по отдельности
    прекрасно помещается в память."""
    with _cache_lock:
        keys_to_drop = [k for k in _hf_cache if k != keep_key]
        for k in keys_to_drop:
            del _hf_cache[k]
    if not keys_to_drop:
        return
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[MODEL] Выгружены из памяти: {keys_to_drop}")
    except Exception:
        pass

def _friendly_load_error(e: Exception) -> Exception:
    """transformers/torch иногда отказываются загружать веса модели, если
    файлы лежат в старом .bin-формате (не safetensors) и установлен torch
    старее 2.6 — это защитная мера против CVE-2025-32434 (уязвимость в
    torch.load). Сообщение об этом приходит длинной англоязычной простынёй
    без единого слова о том, что делать — переводим её в понятную
    инструкцию с готовым решением (см. UPGRADE_TORCH.bat)."""
    msg = str(e)
    if "torch.load" in msg and ("v2.6" in msg or "weights_only" in msg):
        return RuntimeError(
            "Установленная версия PyTorch слишком старая для этой модели: "
            "начиная с определённой версии transformers, загрузка непроверенных "
            ".bin-файлов весов требует PyTorch не старее 2.6 (защита от "
            "уязвимости CVE-2025-32434). Запустите UPGRADE_TORCH.bat в папке "
            "программы, чтобы обновить PyTorch без потери GPU/CPU-версии, затем "
            f"попробуйте распознать ещё раз.\n\nИсходная ошибка: {msg[:300]}"
        )
    return e

def get_model(key: str, force_device: str | None = None):
    cache_key = key if not force_device else f"{key}#{force_device}"
    with _cache_lock:
        if cache_key in _hf_cache:
            return _hf_cache[cache_key]
    # Держим в памяти/на GPU только одну модель транскрипции за раз —
    # см. комментарий в _evict_hf_cache(). Форсированный CPU-фолбэк не
    # трогаем при обычной эвикции — см. _evict_hf_cache(keep_key=...).
    _evict_hf_cache(keep_key=cache_key if force_device else None)
    info  = HF_MODELS[key]
    arch  = info["arch"]
    local = str(_find_model_dir(key))

    try:
        return _load_model_impl(key, info, arch, local, force_device, cache_key)
    except Exception as e:
        raise _friendly_load_error(e) from e

def _load_model_impl(key: str, info: dict, arch: str, local: str,
                     force_device: str | None, cache_key: str):

    if arch == "faster_whisper":
        from faster_whisper import WhisperModel
        device = force_device or _torch_device()
        # int8 на CPU — основной источник ускорения (обычно 3-4× быстрее,
        # чем fp32 через обычный transformers/openai-whisper, при почти
        # незаметной потере точности). На GPU остаёмся на float16 — там
        # это уже быстро и без компромиссов по качеству.
        # На CPU принудительно int8 — это наш осознанный выбор ради
        # скорости (см. историю с faster-whisper). На GPU НЕ фиксируем
        # float16 жёстко: карты до Volta (например, GTX 10xx/Pascal, у
        # которых нет тензорных ядер) не умеют эффективно считать в
        # float16, и CTranslate2 в этом случае просто отказывается
        # запускаться с ошибкой. "auto" даёт библиотеке самой выбрать
        # тип, который эта конкретная карта реально поддерживает.
        compute_type = "int8" if device == "cpu" else "auto"
        m = WhisperModel(local, device=device, compute_type=compute_type)
        with _cache_lock: _hf_cache[cache_key] = ("faster_whisper", m)
        return ("faster_whisper", m)

    if arch == "whisper":
        import whisper
        device = force_device or _torch_device()
        m = whisper.load_model(info["id"].replace("whisper-", ""), device=device)
        with _cache_lock: _hf_cache[cache_key] = ("whisper", m)
        return ("whisper", m)

    if arch == "whisper_hf":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
        import torch
        device = force_device or _torch_device()
        # ВАЖНО: transformers.pipeline() без явного device= ВСЕГДА использует
        # CPU, даже если torch видит GPU. Раньше это никак не задавалось —
        # самые рекомендуемые модели (Whisper Large-v3 ★) работали на CPU,
        # даже на машинах с мощной NVIDIA-видеокартой, простаивающей рядом.
        #
        # На GPU считаем в float16 — вдвое меньше памяти, чем float32, при
        # практически незаметной разнице в качестве распознавания. На CPU
        # float16 не ускоряет (а часто и не поддерживается), поэтому там
        # остаёмся на float32.
        dtype = torch.float16 if device == "cuda" else torch.float32
        hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            local, torch_dtype=dtype, low_cpu_mem_usage=True)
        hf_proc = AutoProcessor.from_pretrained(local)
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=hf_model, tokenizer=hf_proc.tokenizer,
            feature_extractor=hf_proc.feature_extractor,
            torch_dtype=dtype, chunk_length_s=30, batch_size=1,
            device=(0 if device == "cuda" else -1),
        )
        with _cache_lock: _hf_cache[cache_key] = ("whisper_hf", pipe)
        return ("whisper_hf", pipe)

    # wav2vec2
    from transformers import AutoProcessor, Wav2Vec2Processor, AutoModelForCTC
    try:
        p = AutoProcessor.from_pretrained(local)
    except Exception as _e:
        if any(x in str(_e).lower() for x in ("kenlm","pyctcdecode","ctcdecode","import","module")):
            p = Wav2Vec2Processor.from_pretrained(local)
        else:
            raise
    m = AutoModelForCTC.from_pretrained(local); m.eval()
    device = force_device or _torch_device()
    m = m.to(device)
    with _cache_lock: _hf_cache[cache_key] = ("wav2vec2", p, m, device)
    return ("wav2vec2", p, m, device)

# ─────────────────────────────────────────────────────────────────────────────
# Аудио
# ─────────────────────────────────────────────────────────────────────────────
def extract_audio(src: Path, dst: Path):
    try:
        (ffmpeg.input(str(src))
               .output(str(dst), acodec="pcm_s16le", ac=1, ar=16000, loglevel="error")
               .overwrite_output().run())
    except FileNotFoundError:
        raise RuntimeError("FFmpeg не найден. Убедитесь что ffmpeg.exe лежит в папке ffmpeg/")
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg: {e.stderr.decode() if e.stderr else e}")

def load_wav(wav: Path) -> np.ndarray:
    import wave

    def _read_raw(path: Path):
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.readframes(wf.getnframes())

    # Все остальные wav в приложении — 16кГц/моно/16-бит PCM (см.
    # extract_audio()), но некоторые источники (например, сырой вывод
    # Demucs до ресемплинга, см. _do_separate) могли этому не
    # соответствовать — вплоть до формата float32, который стандартный
    # модуль wave вообще не умеет открыть (упадёт исключением). Чтение
    # такого файла "как есть" без проверки незаметно портит и звук
    # (интерливинг каналов), и тайминги (неверная длительность) — см.
    # историю с реальным видео 3:48, у которого распознавание выдавало
    # тайминги до 19:42. Пробуем прочитать как есть; при любой проблеме
    # (неверный формат или несовпадающие метаданные) приводим файл к
    # ожидаемому формату через ffmpeg вместо того, чтобы молча читать
    # неправильно или падать с непонятной ошибкой.
    try:
        rate, ch, width, frames = _read_raw(wav)
        if (rate, ch, width) != (16000, 1, 2):
            raise ValueError(f"unexpected wav format: {rate}Hz, {ch}ch, {width*8}bit")
    except Exception as e:
        print(f"[AUDIO] {wav.name}: {e} — привожу к 16kHz/mono/16bit")
        fixed = wav.with_name(wav.stem + "_norm16k.wav")
        (ffmpeg.input(str(wav))
               .output(str(fixed), acodec="pcm_s16le", ac=1, ar=16000, loglevel="error")
               .overwrite_output().run())
        try:
            _, _, _, frames = _read_raw(fixed)
        finally:
            try: fixed.unlink()
            except Exception: pass
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

def detect_speaker_changes(audio: np.ndarray, sr: int = 16000,
                            hop_sec: float = 0.05, threshold: float = 0.38) -> list:
    WIN_SEC = 1.0
    win = int(WIN_SEC * sr); hop = int(hop_sec * sr)
    if len(audio) < win * 4: return []
    hann = np.hanning(win); features, times = [], []
    for start in range(0, len(audio) - win, hop):
        frame = audio[start:start + win]
        rms   = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        zcr   = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)
        fft_mag = np.abs(np.fft.rfft(frame * hann))
        freqs   = np.fft.rfftfreq(win, 1.0 / sr)
        denom   = np.sum(fft_mag) + 1e-12
        centroid = float(np.sum(freqs * fft_mag) / denom / 8000.0)
        spread   = float(np.sqrt(np.maximum(0, np.sum(((freqs/8000.0-centroid)**2)*fft_mag)/denom)))
        v_mask   = (freqs >= 80) & (freqs <= 3400)
        v_energy = float(np.sum(fft_mag[v_mask]) / denom)
        features.append([rms, zcr, centroid, spread, v_energy]); times.append(start/sr + WIN_SEC/2.0)
    if len(features) < 20: return []
    feat = np.array(features, dtype=np.float64)
    for i in range(feat.shape[1]):
        std = feat[:, i].std()
        if std > 1e-9: feat[:, i] = (feat[:, i] - feat[:, i].mean()) / std
    deltas = np.zeros_like(feat)
    for i in range(1, len(feat) - 1):
        deltas[i] = (feat[i+1] - feat[i-1]) / 2.0
    combined = np.concatenate([feat, deltas], axis=1)
    half = max(8, int(1.5 / hop_sec)); scores = np.zeros(len(combined))
    for i in range(half, len(combined) - half):
        scores[i] = float(np.linalg.norm(combined[i:i+half].mean(axis=0) - combined[i-half:i].mean(axis=0)))
    s_med = float(np.median(scores[scores > 0])); s_std = float(scores[scores > 0].std())
    adaptive_threshold = max(threshold, s_med + 1.8 * s_std)
    candidates = [(times[i], scores[i]) for i in range(len(scores)) if scores[i] > adaptive_threshold]
    filtered, last_t = [], -999.0
    for t, score in sorted(candidates, key=lambda x: -x[1]):
        if t - last_t > 2.0: filtered.append(t); last_t = t
    return sorted(filtered)

def _adaptive_pause(words: list, user_pause: float) -> float:
    gaps = [words[i]["start"] - words[i-1]["end"] for i in range(1, len(words))
            if 0.05 < words[i]["start"] - words[i-1]["end"] < 5.0]
    if len(gaps) < 5: return user_pause
    gaps.sort(); p75 = gaps[int(len(gaps) * 0.75)]
    return max(user_pause * 0.4, min(2.0, p75))

# ─────────────────────────────────────────────────────────────────────────────
# Защита от "галлюцинаций" Whisper — известный артефакт: на тишине/шуме/музыке
# модель иногда не молчит, а выдаёт связный, но полностью выдуманный текст
# (он был в обучающих данных — субтитры с YouTube, титры каналов и т.п.).
# Характерные признаки: высокая no_speech_prob (модель сама не уверена, что
# это речь), низкая avg_logprob (низкая уверенность в тексте) и/или высокий
# compression_ratio (текст ненормально повторяющийся/зацикленный — ровно то,
# что видно в примере "Субтитры сделал DimaTorzok Субтитры сделал DimaTorzok").
# Пороги — стандартные дефолты самого Whisper для этих же целей.
_HALLUCINATION_PHRASES = (
    "субтитры сделал dimatorzok", "субтитры делал dimatorzok",
    "редактор субтитров", "корректор а.egorova", "корректор а.синецкая",
    "продолжение следует", "untertitelung des zdf", "amara.org community",
    "субтитры создавал", "фондтелевидения", "субтитры подготовлены",
)
def _looks_hallucinated(seg) -> bool:
    get = (lambda k, d=0: seg.get(k, d)) if isinstance(seg, dict) else (lambda k, d=0: getattr(seg, k, d))
    text_low = (get("text", "") or "").strip().lower()
    if not text_low:
        return False
    if any(p in text_low for p in _HALLUCINATION_PHRASES):
        return True
    stat_suspect = (get("no_speech_prob", 0) > 0.6 and get("avg_logprob", 0) < -1.0) \
                   or get("compression_ratio", 0) > 2.4
    return stat_suspect

# ─────────────────────────────────────────────────────────────────────────────
# Защита от неверных таймингов у предлогов на стыке сцен. Модель иногда
# распознаёт короткий предлог ВЕРНО по тексту, но ставит ему тайминг в
# КОНЦЕ предыдущей фразы, перед длинной паузой — хотя по смыслу (и по
# грамматике: предлог не может заканчивать предложение) он относится к
# НАЧАЛУ следующей. Из-за этого при группировке реплик по паузам предлог
# "прилипает" не к той сцене, а тайминг всей новой реплики съезжает на всю
# длину паузы раньше (у пользователя видели расхождение до 2 минут).
_RU_PREPOSITIONS = {
    "в","во","на","с","со","к","ко","у","о","об","обо","от","ото","до","из","изо",
    "по","за","над","надо","под","подо","при","для","без","безо","через","чрез",
    "из-за","из-под","между","меж","перед","передо","про","среди","вокруг","около",
    "кроме","после","вместо","сквозь","ради","согласно","благодаря","вдоль","возле",
    "внутри","снаружи","напротив","мимо","путём","путем","насчёт","насчет","вроде",
}
def _fix_misaligned_prepositions(words: list, gap_threshold: float = 1.2) -> list:
    if len(words) < 2:
        return words
    fixed = [dict(w) for w in words]
    for i in range(len(fixed) - 1):
        w = fixed[i]; nxt = fixed[i+1]
        word_clean = w["word"].strip().strip(".,!?;:…").lower()
        gap = nxt["start"] - w["end"]
        if word_clean in _RU_PREPOSITIONS and gap > gap_threshold:
            # Переносим тайминг предлога вплотную к следующему слову —
            # то есть в начало следующей фразы, как и должно быть по
            # грамматике, вместо конца предыдущей.
            dur = max(0.05, min(0.35, w["end"] - w["start"]))
            w["end"] = nxt["start"]
            w["start"] = max(w["start"], nxt["start"] - dur)
    return fixed

def _has_immediate_repeat(text: str, min_unit: int = 3, max_unit: int = 8) -> bool:
    """Общий детектор зацикливания: ищет фразу из min_unit..max_unit слов,
    которая идёт СРАЗУ ЖЕ повторно (как в примере "Субтитры сделал
    DimaTorzok Субтитры сделал DimaTorzok"). Не привязан к конкретным
    фразам — ловит и другие подобные галлюцинации. min_unit=3, а не 1-2 —
    короткие повторы вроде "да да" или "нет нет" бывают в естественной
    речи, а вот 3+ слова подряд дважды — почти всегда галлюцинация."""
    words = text.split()
    n = len(words)
    for unit in range(min_unit, min(max_unit, n // 2) + 1):
        for start in range(0, n - unit * 2 + 1):
            if words[start:start+unit] == words[start+unit:start+unit*2]:
                return True
    return False

def group_words(words: list, pause: float, speaker_changes: list | None = None,
                audio=None) -> list:
    words = _fix_misaligned_prepositions(words)
    if not words: return []
    auto_pause = _adaptive_pause(words, pause)
    SENT_END = {".", "!", "?", "...", "…"}; SOFT_END = {",", ";", ":"}
    MAX_WORDS = 15; SOFT_GAP = 0.20; SPK_RADIUS = 1.2
    spk_sorted = sorted(speaker_changes) if speaker_changes else []
    def near_change(mid_t):
        if not spk_sorted: return False
        import bisect
        idx = bisect.bisect_left(spk_sorted, mid_t)
        return any(abs(c - mid_t) <= SPK_RADIUS for c in spk_sorted[max(0,idx-1):idx+2])
    # Самый фундаментальный рубеж защиты от галлюцинаций: если под текстом
    # физически почти нет звука (сегмент ниже порога громкости), это не
    # может быть настоящей речью — модель просто придумала текст. В
    # отличие от проверки по фразам/повторам, это ловит ЛЮБУЮ галлюцинацию
    # на тихом участке, а не только уже известные. Порог занижен нарочно
    # (это почти цифровая тишина), чтобы не резать тихую настоящую речь —
    # шёпот, дальний план и т.п. должны проходить нормально.
    SILENCE_RMS = 0.004
    def is_near_silent(t0, t1):
        if audio is None: return False
        i0, i1 = int(t0 * 16000), int(t1 * 16000)
        chunk = audio[max(0,i0):min(len(audio),i1)]
        if len(chunk) == 0: return False
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        return rms < SILENCE_RMS
    segs, cur = [], [words[0]]
    for w in words[1:]:
        gap = w["start"] - cur[-1]["end"]; prev = cur[-1]["word"].strip()
        mid_t = (cur[-1]["end"] + w["start"]) / 2.0
        hard = (gap > auto_pause or any(prev.endswith(p) for p in SENT_END) or near_change(mid_t))
        soft = (len(cur) >= MAX_WORDS and (any(prev.endswith(p) for p in SOFT_END) or gap > SOFT_GAP))
        if hard or soft: segs.append(cur); cur = [w]
        else: cur.append(w)
    if cur: segs.append(cur)
    result = []
    for sg in segs:
        text = " ".join(w["word"].strip() for w in sg).strip()
        if text and text == text.upper() and len(text) > 1: text = text.lower()
        if text: text = text[0].upper() + text[1:]
        # Второй и третий рубежи защиты от галлюцинаций — на уже СОБРАННОМ
        # тексте реплики (а не на сыром сегменте модели, где фраза могла
        # быть разбита на части и не попасть под фильтр раньше):
        # известные фразы/повторы текста, и — самое надёжное — проверка,
        # что под текстом вообще есть звук.
        text_low = text.lower()
        if any(p in text_low for p in _HALLUCINATION_PHRASES) or _has_immediate_repeat(text_low):
            continue
        if is_near_silent(sg[0]["start"], sg[-1]["end"]):
            continue
        result.append({"id": str(uuid.uuid4()), "start": sg[0]["start"], "end": sg[-1]["end"],
                        "start_ts": s2ts(sg[0]["start"]), "end_ts": s2ts(sg[-1]["end"]), "text": text})
    return result

def apply_padding(segments: list, padding: float, audio_duration: float) -> list:
    if not segments or padding <= 0: return segments
    # Defensive: process in chronological order so neighbour-based clamping
    # below can't invert start/end if the model returned out-of-order timestamps
    segments = sorted(segments, key=lambda s: s["start"])
    result = []; n = len(segments)
    for i, seg in enumerate(segments):
        min_start = (segments[i-1]["end"] + 0.05) if i > 0 else 0.0
        max_end   = (segments[i+1]["start"] - 0.05) if i < n-1 else audio_duration
        ns = max(min_start, seg["start"] - padding)
        ne = min(max_end,   seg["end"]   + padding)
        if ne <= ns:  # neighbour clamp inverted the interval — fall back to raw times
            ns, ne = seg["start"], max(seg["end"], seg["start"] + 0.05)
        result.append({**seg, "start": round(ns,3), "end": round(ne,3),
                        "start_ts": s2ts(ns), "end_ts": s2ts(ne)})
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Транскрибация
# ─────────────────────────────────────────────────────────────────────────────
def transcribe(wav: Path, key: str, lang: str, pause: float, prompt: str = "",
               progress_cb=None, cancel_check=None, force_device: str | None = None) -> list:
    """
    progress_cb(pct: float 0-100, stage: str) — необязательный колбэк, чтобы
    вызывающий код (веб-эндпоинт) мог показать честный прогресс вместо
    гадания по таймеру.
    cancel_check() -> bool — необязательный колбэк; если возвращает True,
    распознавание прерывается между чанками с TranscribeCancelled (частичный
    результат не возвращается — по крайней мере пока не транскрибирован файл
    целиком, отменённый прогон считается неудавшимся).
    force_device — "cpu"/"cuda"/None. Позволяет вызывающему коду принудительно
    прогнать модель на конкретном устройстве в обход автоопределения — нужно
    для автоотката на CPU при CUDA out of memory (см. _do_transcribe).
    """
    def _tick(pct, stage="Распознавание"):
        if progress_cb:
            try: progress_cb(min(99.0, pct), stage)
            except Exception: pass
        if cancel_check and cancel_check():
            raise TranscribeCancelled()

    info = HF_MODELS[key]; arch = info["arch"]

    if arch == "faster_whisper":
        _, m = get_model(key, force_device); l = None if lang == "auto" else lang
        _tick(1, "Загрузка модели")
        segments_gen, audio_info = m.transcribe(
            str(wav), language=l, word_timestamps=True,
            initial_prompt=prompt if prompt else None,
            vad_filter=True,   # встроенный VAD (Silero) — режет тишину/шум,
                                # не даёт модели "галлюцинировать" текст на
                                # пустых участках длинных записей
            hallucination_silence_threshold=2.0,  # доп. защита от галлюцинаций
                                # именно на длинных паузах — встроенная фича
                                # faster-whisper для ровно этого случая
        )
        words = []
        total_dur = audio_info.duration if audio_info and audio_info.duration else None
        for seg in segments_gen:
            if _looks_hallucinated(seg):
                _tick(100.0 * seg.end / total_dur if total_dur else 1)
                continue  # пропускаем весь сегмент — не добавляем его слова
            for w in (seg.words or []):
                word = w.word.strip()
                if word: words.append({"word": word, "start": w.start, "end": w.end})
            if total_dur:
                _tick(100.0 * seg.end / total_dur)
            elif cancel_check and cancel_check():
                raise TranscribeCancelled()
        audio = load_wav(wav)
        return group_words(words, pause, detect_speaker_changes(audio), audio)

    if arch == "whisper":
        import whisper
        _, m = get_model(key, force_device); l = None if lang == "auto" else lang
        _tick(1, "Загрузка модели")
        # Прогресс берём из внутреннего tqdm whisper'а (см. _ProgressTqdm) —
        # так сохраняется оригинальное 30-секундное окно контекста модели
        # (а значит и качество), в отличие от ручного нарезания на куски.
        with _whisper_tqdm_lock:
            _whisper_progress_local.cb = lambda pct, stage: _tick(pct, stage)
            _whisper_progress_local.cancel = cancel_check
            orig_tqdm = whisper.transcribe.tqdm.tqdm
            whisper.transcribe.tqdm.tqdm = _ProgressTqdm
            try:
                res = m.transcribe(str(wav), language=l, word_timestamps=True, verbose=False,
                                   initial_prompt=prompt if prompt else None)
            finally:
                whisper.transcribe.tqdm.tqdm = orig_tqdm
                _whisper_progress_local.cb = None
                _whisper_progress_local.cancel = None
        words = [{"word": w["word"], "start": w["start"], "end": w["end"]}
                 for seg in res.get("segments", []) if not _looks_hallucinated(seg)
                 for w in seg.get("words", [])]
        audio = load_wav(wav)
        return group_words(words, pause, detect_speaker_changes(audio), audio)

    if arch == "whisper_hf":
        _, pipe = get_model(key, force_device); lang_arg = None if lang == "auto" else lang
        gen_kwargs = {"return_timestamps": "word"}
        if lang_arg: gen_kwargs["language"] = lang_arg
        if prompt:
            try:
                gen_kwargs["prompt_ids"] = pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
            except Exception as e:
                print(f"[WARN] Prompt encoding failed: {e}")
        # HF-пайплайн не даёт колбэк прогресса по чанкам, поэтому режем
        # сами (как и раньше делалось для wav2vec2) — так видно честный %
        # и можно прервать между кусками. Небольшая потеря контекста на
        # границах кусков — тот же компромисс, что уже был у wav2vec2.
        audio = load_wav(wav); CHUNK = 30 * 16000
        n_chunks = max(1, math.ceil(len(audio) / CHUNK))
        words = []
        _tick(1, "Загрузка модели")
        for i, start_s in enumerate(range(0, len(audio), CHUNK)):
            chunk = audio[start_s:start_s+CHUNK]; offset = start_s / 16000.0
            result = pipe(chunk, generate_kwargs=gen_kwargs, return_timestamps="word")
            for c in result.get("chunks", []):
                ts = c.get("timestamp"); text = c.get("text", "").strip()
                if not text or not ts or ts[0] is None: continue
                t_start = offset + float(ts[0])
                t_end = offset + (float(ts[1]) if ts[1] is not None else float(ts[0]) + 0.3)
                sub_words = text.split()
                if len(sub_words) <= 1:
                    words.append({"word": text, "start": t_start, "end": t_end})
                else:
                    step = (t_end - t_start) / len(sub_words)
                    for j, sw in enumerate(sub_words):
                        words.append({"word": sw, "start": t_start+j*step, "end": t_start+(j+1)*step})
            _tick(100.0 * (i+1) / n_chunks)
        return group_words(words, pause, detect_speaker_changes(audio), audio)

    if arch == "wav2vec2":
        import torch
        pkg = get_model(key, force_device); proc, model, device = pkg[1], pkg[2], pkg[3]
        audio = load_wav(wav); CHUNK = 30 * 16000; all_words = []
        n_chunks = max(1, math.ceil(len(audio) / CHUNK))
        _tick(1, "Загрузка модели")
        for i, start_s in enumerate(range(0, len(audio), CHUNK)):
            chunk = audio[start_s:start_s+CHUNK]; offset = start_s / 16000.0
            inputs = proc(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad(): logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1).cpu()
            tpf = len(chunk) / 16000.0 / logits.shape[1]
            try:
                decoded = proc.tokenizer.decode(predicted_ids[0], output_word_offsets=True)
                raw_text  = decoded.text if hasattr(decoded,"text") else decoded["text"]
                word_offs = decoded.word_offsets if hasattr(decoded,"word_offsets") else decoded.get("word_offsets",[])
                for wo in word_offs:
                    word = wo["word"].strip()
                    if word: all_words.append({"word": word, "start": offset+wo["start_offset"]*tpf,
                                                "end": offset+wo["end_offset"]*tpf})
            except (TypeError, AttributeError, KeyError):
                text = proc.batch_decode(predicted_ids)[0].strip()
                if text:
                    chunk_end = offset + len(chunk)/16000.0; ws = text.split()
                    if ws:
                        step = (chunk_end - offset) / len(ws)
                        for j, w in enumerate(ws):
                            all_words.append({"word":w,"start":offset+j*step,"end":offset+(j+1)*step})
            _tick(100.0 * (i+1) / n_chunks)
        segs = group_words(all_words, pause, detect_speaker_changes(audio), audio)
        # wav2vec2/CTC-модели (Golos и др.) не восстанавливают пунктуацию и
        # регистр — это ограничение самой архитектуры, не баг: на выходе
        # всегда сплошной нижний регистр без знаков препинания. Полноценно
        # это можно исправить только отдельной моделью восстановления
        # пунктуации, но хотя бы заглавную букву в начале каждой реплики
        # проставить можно дёшево — уже заметно читабельнее.
        for seg in segs:
            t = seg.get("text", "")
            if t:
                seg["text"] = t[0].upper() + t[1:]
        return segs
    raise ValueError(f"Unknown arch: {arch}")

# ─────────────────────────────────────────────────────────────────────────────
# Проекты
# ─────────────────────────────────────────────────────────────────────────────
def load_proj(pid):
    p = PROJECTS_DIR / f"{pid}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None

def save_proj(proj):
    proj["updated_at"] = datetime.now().isoformat()
    (PROJECTS_DIR / f"{proj['id']}.json").write_text(
        json.dumps(proj, ensure_ascii=False, indent=2), "utf-8")

def list_projs():
    out = []
    for p in PROJECTS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
            out.append({"id": d["id"], "name": d.get("name","Без названия"),
                        "created_at": d.get("created_at",""), "updated_at": d.get("updated_at",""),
                        "segments_count": len(d.get("segments",[])),
                        "audio_file": d.get("audio_file",""), "has_video": d.get("has_video",False),
                        "has_translation": d.get("has_translation",False)})
        except: pass
    return sorted(out, key=lambda x: x.get("updated_at",""), reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# API — Модели
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/debug")
def api_debug():
    models_info = [{"key":k,"arch":info["arch"],"downloaded":is_downloaded(k),
                    "path":str(model_path(k)),"path_exists":Path(model_path(k)).exists()}
                   for k,info in HF_MODELS.items()]
    return jsonify({"hf_models_count":len(HF_MODELS),"models_dir":str(MODELS_DIR),
                    "models_dir_exists":MODELS_DIR.exists(),"models":models_info})

def _scan_reclaimable_space() -> dict:
    """Find ALL disk space wasted by duplicate model data:

    Case A — Per-key folders duplicating an existing shared folder:
      MODELS_DIR/<key>/config.json  AND  MODELS_DIR/_shared/<hf_id>/config.json
      → the per-key folder is fully redundant.

    Case B — Multiple per-key folders for the same hf_id, NO shared folder:
      e.g. whisper-large-v3-ru + whisper-large-v3-en + ... all pointing to
      openai/whisper-large-v3.  The biggest/first is the canonical copy;
      the rest are pure duplicates.  We mark the extras for deletion and
      recommend migrating the canonical one to _shared/ so future
      downloads stay deduplicated.

    Case C — Stale blobs in ~/.cache/huggingface/hub for models we already
      have locally — pure redundancy once we have our own copy.
    """
    items = []
    total_bytes = 0

    # Group catalog entries by hf_id (models with the same underlying weights)
    id_to_keys: dict[str, list[str]] = {}
    for key, info in HF_MODELS.items():
        if info.get("arch") not in ("faster_whisper", "whisper_hf", "wav2vec2", "nllb", "whisper"):
            continue
        hf_id = info["id"]
        id_to_keys.setdefault(hf_id, []).append(key)

    for hf_id, keys in id_to_keys.items():
        shared = _shared_dir(hf_id)
        # config.json — крошечный файл, скачивается за долю секунды; реальные
        # веса — гигабайты и могут не докачаться (обрыв сети, антивирус,
        # закрытие приложения на середине). Раньше здесь проверялся только
        # config.json — если в _shared/ он был, а веса не докачались, папка
        # <key>/ с полноценными весами всё равно помечалась как "устаревший
        # дубликат" и удалялась, оставляя только битую _shared/ — модель
        # переставала считаться скачанной и требовала повторной загрузки.
        shared_ok = shared.exists() and (shared / "config.json").exists() \
                    and _model_weights_present(shared)

        if shared_ok:
            # Case A: any per-key folder alongside a shared folder is redundant
            for key in keys:
                per_key = MODELS_DIR / key
                if per_key.exists() and (per_key / "config.json").exists() \
                        and per_key.resolve() != shared.resolve():
                    sz = _folder_size(per_key)
                    items.append({
                        "type": "legacy_folder",
                        "label": f"Устаревшая копия «{HF_MODELS[key]['label']}»",
                        "path": str(per_key), "key": key, "size_bytes": sz,
                    })
                    total_bytes += sz
        else:
            # Case B: find all per-key folders that exist for this hf_id
            existing = [(key, MODELS_DIR / key)
                        for key in keys
                        if (MODELS_DIR / key / "config.json").exists()]
            if len(existing) >= 2:
                # Кандидат с реально докачанными весами — вперёд остальных;
                # среди равных по этому признаку — самый большой на диске.
                existing.sort(
                    key=lambda kp: (_model_weights_present(kp[1]), _folder_size(kp[1])),
                    reverse=True)
                _canonical_key, canonical_path = existing[0]
                for dup_key, dup_path in existing[1:]:
                    sz = _folder_size(dup_path)
                    items.append({
                        "type": "same_model_duplicate",
                        "label": f"Дубль той же модели «{HF_MODELS[dup_key]['label']}»",
                        "path": str(dup_path), "key": dup_key, "size_bytes": sz,
                        # also record canonical so clean_duplicates can migrate it
                        "canonical_path": str(canonical_path),
                        "hf_id": hf_id,
                    })
                    total_bytes += sz

    # Case C: stale HF cache blobs
    hf_cache_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
    if hf_cache_root.exists():
        materialized_ids = {info["id"] for info in HF_MODELS.values()
                             if info.get("arch") in ("faster_whisper", "whisper_hf", "wav2vec2", "nllb", "whisper")
                             and (_shared_dir(info["id"]) / "config.json").exists()}
        for repo_id in materialized_ids:
            cache_dir = hf_cache_root / ("models--" + repo_id.replace("/", "--"))
            if cache_dir.exists():
                sz = _folder_size(cache_dir)
                if sz > 1024 * 1024:
                    items.append({
                        "type": "hf_cache",
                        "label": f"Кеш HuggingFace: {repo_id}",
                        "path": str(cache_dir), "key": repo_id, "size_bytes": sz,
                    })
                    total_bytes += sz

    return {"items": items, "total_bytes": total_bytes}


@app.route("/api/models/scan_duplicates")
def api_scan_duplicates():
    return jsonify(_scan_reclaimable_space())


@app.route("/api/models/clean_duplicates", methods=["POST"])
def api_clean_duplicates():
    scan = _scan_reclaimable_space()
    freed = 0
    errors = []
    migrated = set()

    for item in scan["items"]:
        try:
            p = Path(item["path"])
            # Case B: before deleting the duplicate, ensure the canonical copy
            # is moved to _shared/ so future downloads stay deduplicated
            if item["type"] == "same_model_duplicate" and item.get("hf_id"):
                hf_id = item["hf_id"]
                if hf_id not in migrated:
                    canonical = Path(item["canonical_path"])
                    shared = _shared_dir(hf_id)
                    if canonical.exists() and not shared.exists():
                        shared.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(str(canonical), str(shared))
                    migrated.add(hf_id)
            if p.exists():
                sz = _folder_size(p)
                shutil.rmtree(p, ignore_errors=False)
                freed += sz
        except Exception as e:
            errors.append({"path": item["path"], "error": str(e)})

    return jsonify({"ok": True, "freed_bytes": freed, "errors": errors})


@app.route("/api/models")
def api_models():
    out = []
    for key, info in HF_MODELS.items():
        with _dl_lock: dl_st = _download_status.get(key, {}).get("status")
        out.append({"key":key,"label":info["label"],"lang":info["lang"],
                    "lang_name":info["lang_name"],"arch":info["arch"],"size_mb":info["size_mb"],
                    "hf_id":info["id"],
                    "downloaded":is_downloaded(key),"dl_status":dl_st})
    return jsonify(out)

@app.route("/api/models/<key>/download", methods=["POST"])
def api_dl_start(key):
    if key not in HF_MODELS: return jsonify({"error":"Модель не найдена"}), 404
    with _dl_lock: st = _download_status.get(key,{}).get("status")
    if st == "downloading": return jsonify({"ok":True,"status":"already_downloading"})
    threading.Thread(target=_bg_download, args=(key,), daemon=True).start()
    return jsonify({"ok":True,"status":"started"})

@app.route("/api/models/<key>/status")
def api_dl_status(key):
    with _dl_lock: st = _download_status.get(key, {})
    return jsonify({"key":key,"status":st.get("status"),"error":st.get("error"),
                    "downloaded":is_downloaded(key),"downloaded_bytes":st.get("downloaded_bytes",0),
                    "total_bytes":st.get("total_bytes",0),"pct":st.get("pct",0)})

@app.route("/api/models/<key>/delete", methods=["POST"])
def api_dl_delete(key):
    info = HF_MODELS.get(key)
    if not info: return jsonify({"error":"Не найдена"}), 404

    if info["arch"] == "whisper":
        cache = Path(os.path.expanduser("~")) / ".cache" / "whisper"
        name  = info["id"].replace("whisper-", "")
        deleted = []
        if cache.exists():
            for f in list(cache.glob(f"{name}*")):
                try: f.unlink(); deleted.append(f.name)
                except Exception as e:
                    return jsonify({"error": f"Не удалось удалить {f.name}: {e}"}), 500
        with _cache_lock: _hf_cache.pop(key, None)
        with _dl_lock:    _download_status.pop(key, None)
        if not deleted:
            return jsonify({"error": f"Файлы не найдены в {cache}"}), 404
        return jsonify({"ok": True, "deleted": deleted})

    # Для HF-моделей: удаляем общую папку → затрагивает все ключи с тем же hf_id
    p = model_path(key)
    also = _shared_users(key)  # другие ключи, использующие ту же папку
    also_labels = [HF_MODELS[k]["label"] for k in also if k in HF_MODELS]

    if p.exists(): shutil.rmtree(p)
    # Также удаляем старую per-key папку если осталась
    old_p = MODELS_DIR / key
    if old_p.exists() and old_p != p: shutil.rmtree(old_p)

    with _cache_lock:
        _hf_cache.pop(key, None)
        for k in also: _hf_cache.pop(k, None)
    with _dl_lock: _download_status.pop(key, None)

    return jsonify({"ok": True, "also_removed": also_labels})


@app.route("/")
def index():
    resp = send_from_directory("templates", "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp

@app.route("/translator")
def translator():
    resp = send_from_directory("templates", "translator.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp

# ─────────────────────────────────────────────────────────────────────────────
# API — Проекты
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/projects")
def api_list(): return jsonify(list_projs())

@app.route("/api/projects", methods=["POST"])
def api_create():
    d = request.json or {}; pid = str(uuid.uuid4())
    proj = {"id":pid,"name":d.get("name","Новый проект"),
            "created_at":datetime.now().isoformat(),"updated_at":datetime.now().isoformat(),
            "segments":[],"audio_file":"","has_video":False,"model_key":"","language":"en"}
    save_proj(proj); return jsonify(proj)

@app.route("/api/projects/<pid>")
def api_get(pid):
    p = load_proj(pid)
    return jsonify(p) if p else (jsonify({"error":"Не найден"}), 404)

@app.route("/api/projects/<pid>", methods=["PUT"])
def api_put(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    for k in ["name","segments","model_key","language","prompt","character_colors"]:
        if k in (request.json or {}): p[k] = request.json[k]
    save_proj(p); return jsonify(p)

@app.route("/api/projects/<pid>", methods=["DELETE"])
def api_del(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    (PROJECTS_DIR / f"{pid}.json").unlink(missing_ok=True)
    locked = []
    for f in UPLOAD_DIR.glob(f"{pid}*"):
        for attempt in range(3):
            try:
                f.unlink(missing_ok=True); break
            except PermissionError:
                if attempt < 2: time.sleep(0.4)
                else: locked.append(f.name)
    if locked:
        return jsonify({"ok":True,"warning":f"Не удалось удалить файлы (заняты): {', '.join(locked)}"}), 200
    return jsonify({"ok":True})

# ─────────────────────────────────────────────────────────────────────────────
# API — Загрузка файла
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/projects/<pid>/upload", methods=["POST"])
def api_upload(pid):
    try:
        p = load_proj(pid)
        if not p: return jsonify({"error":"Не найден"}), 404
        if "file" not in request.files: return jsonify({"error":"Нет файла"}), 400
        f = request.files["file"]; suf = Path(f.filename).suffix.lower()
        if suf not in SUPPORTED_AUDIO | SUPPORTED_VIDEO:
            return jsonify({"error":f"Формат {suf} не поддерживается"}), 400
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        is_video = suf in SUPPORTED_VIDEO
        if is_video:
            vid_p = UPLOAD_DIR / f"{pid}_video{suf}"
            f.save(str(vid_p)); raw = vid_p
        else:
            raw = UPLOAD_DIR / f"{pid}_raw{suf}"; f.save(str(raw))
        wav = UPLOAD_DIR / f"{pid}.wav"
        try:
            extract_audio(raw, wav)
        except Exception as e:
            return jsonify({"error":str(e)}), 500
        finally:
            if not is_video: raw.unlink(missing_ok=True)
        # Keep an immutable copy of the original audio for later switching back
        orig_copy = UPLOAD_DIR / f"{pid}_original.wav"
        try:
            shutil.copy2(str(wav), str(orig_copy))
        except Exception:
            pass
        p.update({"audio_file":f.filename,"audio_ready":True,
                  "has_video":is_video,"video_ext":suf if is_video else ""})
        save_proj(p)
        return jsonify({"ok":True,"filename":f.filename,"has_video":is_video})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":f"Ошибка загрузки: {e}"}), 500

# ─────────────────────────────────────────────────────────────────────────────
# API — Медиа
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/projects/<pid>/audio")
def api_audio(pid):
    wav = UPLOAD_DIR / f"{pid}.wav"
    if not wav.exists(): return jsonify({"error":"Нет аудио"}), 404
    return send_from_directory(str(UPLOAD_DIR), f"{pid}.wav", mimetype="audio/wav")


@app.route("/api/projects/<pid>/audio_variants")
def api_audio_variants(pid):
    """List all audio sources available for playback in this project:
    the immutable original plus any completed Demucs stems."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404

    variants = []
    orig = UPLOAD_DIR / f"{pid}_original.wav"

    # Backward-compat: projects created before original-preservation was added
    # don't have this file yet. Backfill it now, on first request.
    if not orig.exists():
        try:
            video_ext = p.get("video_ext", "")
            video_src = UPLOAD_DIR / f"{pid}_video{video_ext}" if video_ext else None
            if p.get("has_video") and video_src and video_src.exists():
                # Best source: re-extract straight from the untouched video file
                extract_audio(video_src, orig)
            else:
                main_wav = UPLOAD_DIR / f"{pid}.wav"
                if main_wav.exists():
                    # Best-effort fallback: current main track (may already be
                    # a separated stem if it was swapped before this fix existed)
                    shutil.copy2(str(main_wav), str(orig))
        except Exception:
            pass

    if orig.exists():
        variants.append({"key": "original", "label": "Оригинал (со звуком)", "icon": "🎬"})

    if SEP_DIR.exists():
        for model_dir in SEP_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            proj_dir = model_dir / pid
            if not proj_dir.is_dir():
                continue
            for stem, label, icon in (("no_vocals", "Без голоса", "🎵"), ("vocals", "Только голос", "🎤")):
                f = proj_dir / f"{stem}.wav"
                if f.exists():
                    variants.append({
                        "key": f"{model_dir.name}:{stem}",
                        "label": f"{label} ({model_dir.name})",
                        "icon": icon,
                    })

    for t in p.get("extra_tracks", []):
        f = _track_file(pid, t["id"])
        if f.exists():
            variants.append({
                "key": f"track:{t['id']}",
                "label": t.get("name", "Доп. дорожка"),
                "icon": "🎧",
                "removable": True,
            })

    active = p.get("active_audio_variant", "original" if orig.exists() else "")
    return jsonify({"variants": variants, "active": active})


@app.route("/api/projects/<pid>/audio_variant/<path:variant>")
def api_audio_variant_file(pid, variant):
    """Serve a specific audio variant for PLAYBACK ONLY.
    Does not touch the transcription source or segments."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404

    if variant == "original":
        f = UPLOAD_DIR / f"{pid}_original.wav"
    elif variant.startswith("track:"):
        track_id = variant.split(":", 1)[1]
        f = _track_file(pid, track_id)
    elif ":" in variant:
        model, stem = variant.split(":", 1)
        if stem not in ("vocals", "no_vocals"):
            return jsonify({"error": "Неверный stem"}), 400
        f = _sep_file(pid, model, stem)
    else:
        return jsonify({"error": "Неверный вариант"}), 400

    if not f.exists():
        return jsonify({"error": "Файл не найден"}), 404
    return send_file(str(f), mimetype="audio/wav", as_attachment=False)


@app.route("/api/projects/<pid>/active_audio_variant", methods=["POST"])
def api_set_active_audio_variant(pid):
    """Remember which audio variant the user picked for playback (UI state only)."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404
    d = request.json or {}
    p["active_audio_variant"] = d.get("variant", "original")
    save_proj(p)
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────────────────────
# API — Дополнительные аудиодорожки (например: другая озвучка/дубляж,
# загруженные отдельно от основного видео/аудио проекта)
# ─────────────────────────────────────────────────────────────────────────────
def _track_file(pid: str, track_id: str) -> Path:
    return UPLOAD_DIR / f"{pid}_track_{track_id}.wav"

@app.route("/api/projects/<pid>/tracks", methods=["POST"])
def api_upload_track(pid):
    """Upload an extra audio track for this project. Accepts audio OR video —
    if a video file is given, only its audio is extracted (like the main
    upload flow). Does NOT touch the main working audio or the original."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404
    if "file" not in request.files:
        return jsonify({"error": "Нет файла"}), 400
    f = request.files["file"]
    suf = Path(f.filename).suffix.lower()
    if suf not in SUPPORTED_AUDIO | SUPPORTED_VIDEO:
        return jsonify({"error": f"Формат {suf} не поддерживается"}), 400

    name = (request.form.get("name") or Path(f.filename).stem or "Доп. дорожка").strip()[:80]
    track_id = uuid.uuid4().hex[:10]

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw = UPLOAD_DIR / f"{pid}_track_{track_id}_raw{suf}"
    f.save(str(raw))
    dst = _track_file(pid, track_id)
    try:
        extract_audio(raw, dst)
    except Exception as e:
        raw.unlink(missing_ok=True)
        return jsonify({"error": f"Ошибка обработки: {e}"}), 500
    finally:
        raw.unlink(missing_ok=True)

    p.setdefault("extra_tracks", []).append({
        "id": track_id,
        "name": name,
        "added_at": datetime.now().isoformat(),
    })
    save_proj(p)
    return jsonify({"ok": True, "track": {"id": track_id, "name": name}})

@app.route("/api/projects/<pid>/tracks/<track_id>", methods=["DELETE"])
def api_delete_track(pid, track_id):
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404
    tracks = p.get("extra_tracks", [])
    if not any(t["id"] == track_id for t in tracks):
        return jsonify({"error": "Дорожка не найдена"}), 404
    p["extra_tracks"] = [t for t in tracks if t["id"] != track_id]

    # If this track was the active playback variant or the manually chosen
    # transcription source, fall back to safe defaults.
    if p.get("active_audio_variant") == f"track:{track_id}":
        p["active_audio_variant"] = "original"
    if p.get("sep_source", {}).get("track") == track_id:
        p["transcribe_source_manual"] = False
        p.pop("sep_source", None)

    save_proj(p)
    _track_file(pid, track_id).unlink(missing_ok=True)
    return jsonify({"ok": True})

@app.route("/api/projects/<pid>/tracks/<track_id>/use_as_audio", methods=["POST"])
def api_track_use_as_audio(pid, track_id):
    """Copy an extra track as the project's transcription-source audio.
    Mirrors /separated/<model>/<stem>/use_as_audio for Demucs stems."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404
    src = _track_file(pid, track_id)
    if not src.exists():
        return jsonify({"error": "Дорожка не найдена"}), 404
    dst = UPLOAD_DIR / f"{pid}.wav"
    try:
        shutil.copy2(str(src), str(dst))
    except Exception as e:
        return jsonify({"error": f"Ошибка копирования: {e}"}), 500
    track_name = next((t["name"] for t in p.get("extra_tracks", []) if t["id"] == track_id), track_id)
    p["audio_ready"] = True
    p["sep_source"] = {"track": track_id, "name": track_name}
    p["transcribe_source_manual"] = True
    save_proj(p)
    return jsonify({"ok": True})

@app.route("/api/projects/<pid>/video")
def api_video(pid):
    p = load_proj(pid)
    if not p or not p.get("has_video"): return jsonify({"error":"Нет видео"}), 404
    ext  = p.get("video_ext", ".mp4"); path = UPLOAD_DIR / f"{pid}_video{ext}"
    if not path.exists(): return jsonify({"error":"Файл не найден"}), 404
    mime = {".mp4":"video/mp4",".mov":"video/quicktime",".avi":"video/x-msvideo",
            ".mkv":"video/x-matroska",".webm":"video/webm"}.get(ext,"video/mp4")
    fs = path.stat().st_size; rh = request.headers.get("Range")
    if rh:
        b1, b2 = 0, None; parts = rh.replace("bytes=","").split("-")
        b1 = int(parts[0])
        if parts[1]: b2 = int(parts[1])
        if b2 is None: b2 = fs - 1
        length = b2 - b1 + 1
        def gen():
            with open(path,"rb") as fh:
                fh.seek(b1); rem = length
                while rem:
                    chunk = fh.read(min(65536,rem))
                    if not chunk: break
                    rem -= len(chunk); yield chunk
        resp = Response(gen(), 206, mimetype=mime, direct_passthrough=True)
        resp.headers.update({"Content-Range":f"bytes {b1}-{b2}/{fs}",
                              "Accept-Ranges":"bytes","Content-Length":str(length)})
        return resp
    return send_file(str(path), mimetype=mime)

# ─────────────────────────────────────────────────────────────────────────────
# API — Транскрибация
# ─────────────────────────────────────────────────────────────────────────────
def _do_transcribe(pid: str, wav: Path, key: str, lang: str, paus: float,
                   padding: float, prompt: str, sep_source: dict | None):
    with _asr_lock:
        _asr_status[pid] = {"status":"running","progress":1,"error":None,
                            "stage":"Запуск модели...","model_key":key,
                            "started_at":time.time()}
    _asr_cancel_flags[pid] = False

    def _cb(pct, stage):
        with _asr_lock:
            if pid in _asr_status:
                _asr_status[pid]["progress"] = round(pct, 1)
                _asr_status[pid]["stage"] = stage

    def _cancelled():
        return _asr_cancel_flags.get(pid, False)

    try:
        t0 = time.time()
        used_device = _torch_device()
        try:
            segs = transcribe(wav, key, lang, paus, prompt=prompt,
                              progress_cb=_cb, cancel_check=_cancelled)
        except TranscribeCancelled:
            with _asr_lock:
                _asr_status[pid] = {"status":"cancelled","progress":0,"error":None,
                                    "stage":"Отменено пользователем","model_key":key}
            return
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise  # не OOM — пробрасываем исходную ошибку как есть
            # Частая причина: в кэше застряла модель от предыдущего
            # запуска (см. _evict_hf_cache). На случай, если что-то
            # всё же осталось висеть в памяти GPU — освобождаем и
            # пробуем ещё один раз, прежде чем сдаваться.
            print("[ASR] CUDA OOM — очищаю кэш моделей и пробую ещё раз...")
            with _asr_lock:
                _asr_status[pid]["stage"] = "Не хватило памяти GPU — очищаю кэш и пробую снова..."
                _asr_status[pid]["progress"] = 1
            _evict_hf_cache(keep_key=None)
            try:
                segs = transcribe(wav, key, lang, paus, prompt=prompt,
                                  progress_cb=_cb, cancel_check=_cancelled)
            except TranscribeCancelled:
                with _asr_lock:
                    _asr_status[pid] = {"status":"cancelled","progress":0,"error":None,
                                        "stage":"Отменено пользователем","model_key":key}
                return
            except Exception as e2:
                if "out of memory" not in str(e2).lower():
                    raise
                # Видеокарте по-прежнему не хватает памяти даже на пустом
                # кэше — значит модель попросту не помещается в её VRAM
                # (см. историю с GTX 1060 6ГБ). Вместо того чтобы просто
                # сдаться и заставить человека вручную переустанавливать
                # torch (см. UPGRADE_TORCH.bat/FIX_GPU_TORCH.bat), прогоняем
                # эту же модель на CPU — медленнее, зато результат будет,
                # а не ошибка. GPU-модель остаётся в кэше нетронутой (см.
                # get_model(force_device=...) — форсированная версия
                # кэшируется отдельным ключом), так что следующий обычный
                # запуск снова попробует GPU.
                print("[ASR] CUDA OOM повторно — переключаюсь на CPU для этого запуска...")
                with _asr_lock:
                    _asr_status[pid]["stage"] = "Видеокарте не хватает памяти — считаю на CPU (медленнее)..."
                    _asr_status[pid]["progress"] = 1
                _evict_hf_cache(keep_key=None)
                try:
                    segs = transcribe(wav, key, lang, paus, prompt=prompt,
                                      progress_cb=_cb, cancel_check=_cancelled,
                                      force_device="cpu")
                    used_device = "cpu"
                except TranscribeCancelled:
                    with _asr_lock:
                        _asr_status[pid] = {"status":"cancelled","progress":0,"error":None,
                                            "stage":"Отменено пользователем","model_key":key}
                    return
                except Exception as e3:
                    raise RuntimeError(
                        "Не хватает памяти видеокарты для этой модели, а откат "
                        "на CPU тоже не сработал. Попробуйте: закрыть другие "
                        "программы, использующие GPU (игры, браузер с "
                        "аппаратным ускорением и т.п.), либо выбрать модель "
                        f"поменьше. Исходная ошибка: {e3}"
                    ) from e3

        # Defensive sanitation: drop/skip any segment where end <= start
        # (can happen on noisy/low-quality audio sources that confuse the model)
        segs = sorted(segs, key=lambda s: s["start"])
        segs = [s for s in segs if s["end"] > s["start"]]
        audio_duration = len(load_wav(wav)) / 16000.0
        if padding > 0:
            segs = apply_padding(segs, padding, audio_duration)
        elapsed = round(time.time() - t0, 1)
        _record_speed("transcribe", key, used_device, audio_duration, elapsed)

        p = load_proj(pid)
        if p is not None:
            p.update({"segments":segs,"model_key":key,"language":lang,
                      "padding":padding,"prompt":prompt})
            if sep_source is not None:
                p["sep_source"] = sep_source
            save_proj(p)

        with _asr_lock:
            _asr_status[pid] = {"status":"done","progress":100,"error":None,
                                "stage":"Готово!","model_key":key,
                                "segments":segs,"elapsed":elapsed,
                                "sep_source":sep_source}
    except Exception as e:
        traceback.print_exc()
        with _asr_lock:
            _asr_status[pid] = {"status":"error","progress":0,"error":str(e)[:800],
                                "stage":"Ошибка","model_key":key}
    finally:
        _asr_cancel_flags.pop(pid, None)


@app.route("/api/projects/<pid>/transcribe", methods=["POST"])
def api_transcribe(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    wav = UPLOAD_DIR / f"{pid}.wav"
    if not wav.exists(): return jsonify({"error":"Сначала загрузите файл"}), 400

    with _asr_lock:
        st = _asr_status.get(pid, {})
    if st.get("status") == "running":
        return jsonify({"error": "Распознавание уже выполняется"}), 409

    d       = request.json or {}
    key     = d.get("model_key", p.get("model_key",""))

    # ── Auto-select best transcription source ──────────────────────────
    # Default: "только голос" (vocals-only) if a separation has produced
    # one, otherwise the original audio. Skipped if the user explicitly
    # picked a track via "Сделать источником распознавания".
    #
    # IMPORTANT: by default only applied for Whisper-family models, since
    # raw CTC models (wav2vec2) are sensitive to separation artifacts.
    # A project can opt in via "prefer_vocals_only" — this flag lives
    # inside the project's own JSON, so it only affects THIS project and
    # never leaks to other projects in the hub.
    model_arch = HF_MODELS.get(key, {}).get("arch", "")
    prefer_vocals = p.get("prefer_vocals_only", False)
    sep_source = None
    if not p.get("transcribe_source_manual", False):
        orig = UPLOAD_DIR / f"{pid}_original.wav"
        try:
            if model_arch in ("faster_whisper", "whisper", "whisper_hf") or prefer_vocals:
                best_vocals = _find_best_vocals(pid)
                if best_vocals and best_vocals.exists():
                    shutil.copy2(str(best_vocals), str(wav))
                    sep_source = {"auto": True, "stem": "vocals"}
                elif orig.exists():
                    shutil.copy2(str(orig), str(wav))
                    sep_source = {"auto": True, "stem": "original"}
            elif orig.exists():
                # Non-Whisper (e.g. wav2vec2) without opt-in: clean original
                shutil.copy2(str(orig), str(wav))
                sep_source = {"auto": True, "stem": "original"}
        except Exception:
            pass  # fall through and use whatever is currently in wav

    lang    = d.get("language",  p.get("language","en"))
    paus    = float(d.get("pause_threshold", 0.5))
    padding = float(d.get("padding", p.get("padding", 0.0)))
    prompt  = d.get("prompt", p.get("prompt","")).strip()
    if not key or key not in HF_MODELS: return jsonify({"error":"Выберите модель"}), 400
    if not is_downloaded(key): return jsonify({"error":"Модель не скачана"}), 400

    threading.Thread(target=_do_transcribe,
                     args=(pid, wav, key, lang, paus, padding, prompt, sep_source),
                     daemon=True).start()
    return jsonify({"ok": True, "status": "started"})


@app.route("/api/projects/<pid>/transcribe/status")
def api_transcribe_status(pid):
    with _asr_lock:
        st = dict(_asr_status.get(pid, {"status": None}))
    return jsonify(st)


@app.route("/api/projects/<pid>/transcribe/cancel", methods=["POST"])
def api_transcribe_cancel(pid):
    with _asr_lock:
        st = _asr_status.get(pid, {})
        if st.get("status") != "running":
            return jsonify({"error": "Распознавание не выполняется"}), 409
        st["stage"] = "Отмена..."
    _asr_cancel_flags[pid] = True
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────────────────────
# API — Сплит / Мёрдж / Перетранскрибация
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/projects/<pid>/split", methods=["POST"])
def api_split(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    d = request.json or {}; sid, st = d.get("segment_id"), d.get("split_time")
    if not sid or st is None: return jsonify({"error":"Нужны segment_id и split_time"}), 400
    ss = ts2s(st) if isinstance(st,str) else float(st)
    word_index = d.get("word_index")  # JS sends exact word index for word-mode splits
    new_segs = []
    for seg in p.get("segments",[]):
        if seg["id"] == sid:
            sv, ev = seg["start"], seg["end"]
            if not (sv < ss < ev):
                return jsonify({"error":f"Время вне реплики ({seg['start_ts']} — {seg['end_ts']})"}), 400
            words = seg["text"].split()
            if word_index is not None:
                # Use the exact word index sent by JS (split AFTER word[word_index])
                idx = max(1, min(int(word_index) + 1, len(words) - 1))
            else:
                # Time-based split: round() instead of int() avoids float rounding errors
                ratio = (ss - sv) / (ev - sv)
                idx = max(1, min(round(len(words) * ratio), len(words) - 1))
            new_segs += [
                {"id":str(uuid.uuid4()),"start":sv,"end":ss,"start_ts":s2ts(sv),"end_ts":s2ts(ss),"text":" ".join(words[:idx]),"character":seg.get("character","")},
                {"id":str(uuid.uuid4()),"start":ss,"end":ev,"start_ts":s2ts(ss),"end_ts":s2ts(ev),"text":" ".join(words[idx:]),"character":seg.get("character","")},
            ]
            # Глава была привязана к началу исходной реплики — после
            # разбиения она логически всё ещё "перед" самым первым словом,
            # то есть остаётся на первой половине.
            if seg.get("chapter"):
                new_segs[-2]["chapter"] = seg["chapter"]
        else: new_segs.append(seg)
    p["segments"] = new_segs; save_proj(p)
    return jsonify({"ok":True,"segments":new_segs})

_spell_vocab_cache: dict = {}   # pid -> (mtime, vocab_dict)

@app.route("/api/spellcheck", methods=["POST"])
def api_spellcheck():
    """Context-aware suggestions from the project's own vocabulary."""
    import re as _re

    def _levenshtein(a: str, b: str) -> int:
        if len(a) > len(b): a, b = b, a
        row = list(range(len(a) + 1))
        for ch in b:
            nr = [row[0] + 1]
            for j, d in enumerate(a):
                nr.append(min(row[j + 1] + 1, nr[-1] + 1, row[j] + (0 if ch == d else 1)))
            row = nr
        return row[-1]

    d = request.json or {}
    raw_word = d.get("word", "").strip()
    pid      = d.get("pid", "")
    if not raw_word or len(raw_word) < 2:
        return jsonify({"suggestions": [], "correct": True})

    word = raw_word.lower()

    # Словарь проекта раньше пересобирался с нуля (чтение файла с диска +
    # обход всех реплик) на КАЖДЫЙ вызов — а вызывается это при каждой паузе
    # в наборе текста (раз в ~650мс). На больших транскрипциях (сотни реплик,
    # тысячи уникальных слов) само построение было терпимым, но обход этого
    # словаря чуть ниже с полным Левенштейном на слово — уже нет (см. там же).
    # Здесь просто кешируем словарь и пересобираем только если файл проекта
    # реально изменился (по mtime) — обычно это редко, в отличие от вызовов.
    vocab: dict[str, int] = {}
    if pid:
        proj_file = PROJECTS_DIR / f"{pid}.json"
        try:
            mtime = proj_file.stat().st_mtime
        except OSError:
            mtime = None
        cached = _spell_vocab_cache.get(pid)
        if cached and mtime is not None and cached[0] == mtime:
            vocab = cached[1]
        else:
            p = load_proj(pid)
            if p:
                for seg in p.get("segments", []):
                    for w in _re.findall(r"[^\s\-\u2013\u2014,.!?;:\"'()\[\]{}]+", seg.get("text", "")):
                        wl = w.lower()
                        if len(wl) >= 2:
                            vocab[wl] = vocab.get(wl, 0) + 1
                if mtime is not None:
                    _spell_vocab_cache[pid] = (mtime, vocab)

    if not vocab:
        return jsonify({"suggestions": [], "correct": True})

    # Exact match — word is already correct
    if word in vocab:
        return jsonify({"suggestions": [], "correct": True})

    # Max edit distance scales with word length
    max_dist = 1 if len(word) <= 3 else 2 if len(word) <= 7 else 3
    wlen = len(word)

    results: list[tuple] = []
    for vw, freq in vocab.items():
        if vw == word:
            continue
        # Prefix / autocomplete mode (word still being typed)
        if vw.startswith(word) and len(vw) > len(word):
            # score: (tier=0, -freq) — most frequent completions first
            results.append((0, -freq, vw))
            continue
        # Дешёвая отсечка перед дорогим Левенштейном: если длины отличаются
        # больше чем на max_dist, расстояние гарантированно больше max_dist —
        # незачем считать полную матрицу. На словаре в тысячи слов это на
        # порядки сокращает число реальных вычислений Левенштейна за запрос.
        if abs(len(vw) - wlen) > max_dist:
            continue
        # Edit-distance similarity
        dist = _levenshtein(word, vw)
        if dist <= max_dist:
            # tier 1: same first letter; tier 2: different first letter
            tier = 1 if vw[0] == word[0] else 2
            results.append((tier, dist, -freq, vw))

    results.sort()
    suggestions = []
    for item in results[:8]:
        suggestions.append(item[-1])  # last element is always the word

    # Restore original capitalisation if the source word was capitalised
    if raw_word[0].isupper():
        suggestions = [s[0].upper() + s[1:] for s in suggestions]

    return jsonify({"suggestions": suggestions[:6], "correct": not suggestions})

@app.route("/api/projects/<pid>/retranscribe_segment", methods=["POST"])
def api_retranscribe_segment(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    wav_full = UPLOAD_DIR / f"{pid}.wav"
    if not wav_full.exists(): return jsonify({"error":"Аудиофайл не найден"}), 400
    d = request.json or {}; seg_id = d.get("segment_id")
    if not seg_id: return jsonify({"error":"Нужен segment_id"}), 400
    seg = next((s for s in p.get("segments",[]) if s["id"]==seg_id), None)
    if not seg: return jsonify({"error":"Сегмент не найден"}), 404
    mkey = p.get("model_key",""); lang = p.get("language","en"); prompt = p.get("prompt","").strip()
    if not mkey or mkey not in HF_MODELS: return jsonify({"error":"Модель не задана"}), 400
    if not is_downloaded(mkey): return jsonify({"error":"Модель не скачана"}), 400
    t_start = seg["start"]; t_end = seg["end"]; padding = float(p.get("padding",0.0))
    if t_end <= t_start:
        return jsonify({"error": "Повреждённый сегмент: конец раньше начала. Исправьте время вручную."}), 400
    cut_start = max(0.0, t_start - padding); cut_end = t_end + padding; duration = cut_end - cut_start
    if (t_end - t_start) < 0.1: return jsonify({"error":"Сегмент слишком короткий"}), 400
    if duration < 0.15:
        return jsonify({"error": "Слишком короткий отрезок аудио для распознавания"}), 400
    tmp_wav = UPLOAD_DIR / f"{pid}_retrans_{seg_id[:8]}.wav"
    try:
        (ffmpeg.input(str(wav_full), ss=cut_start, t=duration)
            .output(str(tmp_wav), acodec="pcm_s16le", ac=1, ar=16000, loglevel="error")
            .overwrite_output().run())
    except Exception as e:
        return jsonify({"error":f"FFmpeg: {e}"}), 500
    if not tmp_wav.exists() or tmp_wav.stat().st_size < 200:
        tmp_wav.unlink(missing_ok=True)
        return jsonify({"error": "Вырезанный фрагмент пуст — проверьте время сегмента"}), 400
    try:
        info = HF_MODELS[mkey]; arch = info["arch"]
        if arch == "faster_whisper":
            _, m = get_model(mkey); l = None if lang=="auto" else lang
            segs_gen, _ai = m.transcribe(str(tmp_wav), language=l,
                                         initial_prompt=prompt if prompt else None)
            new_text = " ".join(s.text.strip() for s in segs_gen).strip()
        elif arch == "whisper":
            import whisper as _whisper; _, m = get_model(mkey)
            l = None if lang=="auto" else lang
            res = m.transcribe(str(tmp_wav), language=l, verbose=False,
                               initial_prompt=prompt if prompt else None)
            new_text = res.get("text","").strip()
        elif arch == "whisper_hf":
            _, pipe = get_model(mkey); lang_arg = None if lang=="auto" else lang
            gen_kwargs = {}
            if lang_arg: gen_kwargs["language"] = lang_arg
            if prompt:
                try: gen_kwargs["prompt_ids"] = pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
                except: pass
            result = pipe(str(tmp_wav), generate_kwargs=gen_kwargs)
            new_text = result.get("text","").strip()
        else:
            import torch; pkg = get_model(mkey); proc, model, device = pkg[1], pkg[2], pkg[3]
            audio = load_wav(tmp_wav)
            inputs = proc(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad(): logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1).cpu()
            try:
                decoded = proc.tokenizer.decode(predicted_ids[0], output_word_offsets=False)
                new_text = decoded if isinstance(decoded,str) else decoded.text
            except: new_text = proc.batch_decode(predicted_ids)[0]
            new_text = new_text.strip()
            if new_text == new_text.upper() and len(new_text) > 1: new_text = new_text.lower()
            if new_text: new_text = new_text[0].upper() + new_text[1:]
        for s in p["segments"]:
            if s["id"] == seg_id: s["text"] = new_text; break
        save_proj(p)
        return jsonify({"ok":True,"segment_id":seg_id,"new_text":new_text})
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500
    finally:
        tmp_wav.unlink(missing_ok=True)

@app.route("/api/projects/<pid>/merge", methods=["POST"])
def api_merge(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    ids = (request.json or {}).get("segment_ids",[]); segs = p.get("segments",[]); ids_set = set(ids)
    if len(ids) < 2: return jsonify({"error":"Нужно минимум 2 реплики"}), 400
    idxs = sorted([i for i,s in enumerate(segs) if s["id"] in ids_set])
    if len(idxs) != len(ids): return jsonify({"error":"Некоторые реплики не найдены"}), 404
    for k in range(1,len(idxs)):
        if idxs[k]-idxs[k-1] != 1: return jsonify({"error":"Выберите непрерывный диапазон"}), 400
    i_first, i_last = idxs[0], idxs[-1]; to_merge = segs[i_first:i_last+1]
    merged = {"id":str(uuid.uuid4()),"start":to_merge[0]["start"],"end":to_merge[-1]["end"],
              "start_ts":to_merge[0]["start_ts"],"end_ts":to_merge[-1]["end_ts"],
              "text":" ".join(s["text"] for s in to_merge)}
    # Оглавление и персонаж были "привязаны" к первой из объединяемых
    # реплик — переносим их на результат, иначе объединение молча стирает
    # и то и другое.
    if to_merge[0].get("character"):
        merged["character"] = to_merge[0]["character"]
    if to_merge[0].get("chapter"):
        merged["chapter"] = to_merge[0]["chapter"]
    p["segments"] = segs[:i_first] + [merged] + segs[i_last+1:]
    save_proj(p)
    return jsonify({"ok":True,"segments":p["segments"]})

# ─────────────────────────────────────────────────────────────────────────────
# API — Перевод (NLLB-200)
# ─────────────────────────────────────────────────────────────────────────────
def _translate_texts_preserving_lines(texts, tokenizer, model, device, tok_kwargs, gen_kwargs):
    """Переводит список текстов реплик, СОХРАНЯЯ разбивку на строки внутри
    каждой реплики. Если модель перевода получает текст с переносами
    строк (например, пользователь вручную расставил их для стихотворной
    структуры — см. реальный кейс с "Alice in Wonderland"), она просто
    склеивает всё в один сплошной абзац на выходе — переносы для нейросети
    не несут смысловой нагрузки, это чисто визуальное форматирование.
    Поэтому бьём каждую реплику на строки, переводим их как отдельные
    короткие тексты одним общим батчем (эффективнее, чем по одной), а
    потом собираем обратно в том же порядке и с тем же числом строк."""
    import torch
    flat_lines, owner, line_counts = [], [], []
    for j, text in enumerate(texts):
        raw_lines = (text or "").split("\n")
        line_counts.append(len(raw_lines))
        for li, line in enumerate(raw_lines):
            line = line.strip()
            if line:
                flat_lines.append(line); owner.append((j, li))
    translated_lines = [""] * len(flat_lines)
    if flat_lines:
        inputs = tokenizer(flat_lines, return_tensors="pt", padding=True,
                           truncation=True, max_length=512, **tok_kwargs).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        translated_lines = tokenizer.batch_decode(out, skip_special_tokens=True)
    per_text_lines = [[""] * lc for lc in line_counts]
    for (j, li), tr in zip(owner, translated_lines):
        per_text_lines[j][li] = tr
    return ["\n".join(lines).strip("\n") for lines in per_text_lines]

def _do_translate(pid: str, model_key: str, src_lang: str, tgt_lang: str):
    p = load_proj(pid)
    if not p:
        with _tr_lock: _translate_status[pid] = {"status":"error","error":"Проект не найден","done":0,"total":0}
        return
    segments = p.get("segments",[]); total = len(segments)
    with _tr_lock: _translate_status[pid] = {"status":"running","done":0,"total":total,"error":None}
    model = None
    _t0 = time.time()
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        local = _find_model_dir(model_key)
        info  = HF_MODELS.get(model_key, {})
        arch  = info.get("arch", "nllb")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(str(local))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(local))
        model.eval(); model = model.to(device)

        BATCH = 8; done = 0

        if arch == "marian":
            # Helsinki/Marian: нет языковых кодов, просто переводим текст
            for i in range(0, total, BATCH):
                batch_segs = segments[i:i+BATCH]
                texts = [s.get("text","").strip() for s in batch_segs]
                translations = _translate_texts_preserving_lines(
                    texts, tokenizer, model, device, tok_kwargs={},
                    gen_kwargs={"num_beams":5,"length_penalty":1.0,"max_length":512})
                for j, seg in enumerate(batch_segs): seg["translation"] = translations[j]
                done += len(batch_segs)
                with _tr_lock: _translate_status[pid]["done"] = done

        else:
            # NLLB: нужны языковые коды
            try:
                tgt_id = tokenizer.lang_code_to_id[tgt_lang]
            except AttributeError:
                try:
                    tgt_id = tokenizer.get_lang_id(tgt_lang)
                except AttributeError:
                    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)

            for i in range(0, total, BATCH):
                batch_segs = segments[i:i+BATCH]
                texts = [s.get("text","").strip() for s in batch_segs]
                translations = _translate_texts_preserving_lines(
                    texts, tokenizer, model, device,
                    tok_kwargs={"src_lang": src_lang},
                    gen_kwargs={"forced_bos_token_id": tgt_id, "num_beams":5,
                               "length_penalty":1.0, "max_length":512,
                               "no_repeat_ngram_size":3})
                for j, seg in enumerate(batch_segs): seg["translation"] = translations[j]
                done += len(batch_segs)
                with _tr_lock: _translate_status[pid]["done"] = done

        p["segments"] = segments; p["has_translation"] = True
        p["tr_src"] = src_lang; p["tr_tgt"] = tgt_lang; p["tr_model"] = model_key
        save_proj(p)
        try:
            _char_count = sum(len(s.get("text","")) for s in segments)
            _record_speed("translate", model_key, device, _char_count, time.time() - _t0)
        except Exception as _rec_err:
            print(f"[SPEED] translate: не удалось записать статистику ({_rec_err})")
        with _tr_lock: _translate_status[pid] = {"status":"done","done":total,"total":total,"error":None}
    except Exception as e:
        traceback.print_exc()
        with _tr_lock: _translate_status[pid] = {"status":"error","done":0,"total":total,"error":str(e)}
    finally:
        # Модель перевода не кэшируется между запусками — освобождаем память
        # сразу, а не полагаемся на то, что сборщик мусора когда-нибудь сам
        # доберётся до GPU-памяти (на маленьких видеокартах это может не
        # успеть произойти до следующей операции и вызвать "out of memory").
        if model is not None:
            del model
            try:
                import torch, gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


@app.route("/api/projects/<pid>/translate", methods=["POST"])
def api_translate(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Проект не найден"}), 404
    d = request.json or {}
    model_key = d.get("model_key","nllb-600m")
    src_lang  = d.get("src_lang","eng_Latn")
    tgt_lang  = d.get("tgt_lang","rus_Cyrl")
    model_info = HF_MODELS.get(model_key)
    if not model_info: return jsonify({"error":"Модель не найдена"}), 404
    if not is_downloaded(model_key):
        return jsonify({"error":f"Модель '{model_info['label']}' не скачана. Перейдите во вкладку «Модели»."}), 400
    if not p.get("segments"):
        return jsonify({"error":"Нет сегментов для перевода. Сначала выполните транскрибацию."}), 400
    # Для marian-моделей проверяем совместимость языков
    if model_info.get("arch") == "marian":
        m_src = model_info.get("src_lang","")
        m_tgt = model_info.get("tgt_lang","")
        # Приводим NLLB-коды к простым (eng_Latn -> en)
        simple_src = src_lang.split("_")[0] if "_" in src_lang else src_lang
        simple_tgt = tgt_lang.split("_")[0] if "_" in tgt_lang else tgt_lang
        if simple_src != m_src or simple_tgt != m_tgt:
            return jsonify({"error":f"Модель '{model_info['label']}' работает только {m_src.upper()}→{m_tgt.upper()}. Выберите другую модель или измените языки."}), 400
    with _tr_lock: st = _translate_status.get(pid, {})
    if st.get("status") == "running": return jsonify({"error":"Перевод уже выполняется"}), 409
    threading.Thread(target=_do_translate, args=(pid,model_key,src_lang,tgt_lang), daemon=True).start()
    return jsonify({"ok":True})

@app.route("/api/projects/<pid>/translate/status")
def api_translate_status(pid):
    with _tr_lock: st = _translate_status.get(pid, {"status":None,"done":0,"total":0,"error":None})
    result = {"status":st.get("status"),"done":st.get("done",0),
              "total":st.get("total",0),"error":st.get("error")}
    if st.get("status") == "done":
        proj = load_proj(pid)
        result["segments"] = proj.get("segments",[]) if proj else []
    return jsonify(result)

@app.route("/api/projects/<pid>/translation/clear", methods=["POST"])
def api_translation_clear(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Проект не найден"}), 404
    for seg in p.get("segments",[]): seg.pop("translation", None)
    p.pop("has_translation", None)
    save_proj(p)
    return jsonify({"ok":True})

@app.route("/api/nllb_langs")
def api_nllb_langs():
    return jsonify(NLLB_LANGS)

# ─────────────────────────────────────────────────────────────────────────────
# API — Экспорт (с поддержкой перевода)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SAVE-TO-DISK HELPERS  (pywebview cannot download via browser API)
# ─────────────────────────────────────────────────────────────────────────────

def _get_downloads_dir() -> Path:
    """Return the OS Downloads folder (Windows registry → fallback ~/Downloads)."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
        path = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")[0]
        winreg.CloseKey(key)
        return Path(path)
    except Exception:
        return Path.home() / "Downloads"


_settings_lock = threading.Lock()

def _load_settings() -> dict:
    """Load persistent settings from settings.json."""
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_settings(data: dict) -> None:
    with _settings_lock:
        current = _load_settings()
        current.update(data)
        SETTINGS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Прогноз времени обработки — самообучающийся по реальным замерам
# ─────────────────────────────────────────────────────────────────────────────
# Единственный способ дать честный прогноз "сколько это займёт" — запомнить,
# сколько реально заняла обработка НА ЭТОМ КОНКРЕТНОМ устройстве в прошлый
# раз. Любая формула "по паспорту GPU/CPU" ошибается в разы: реальная
# скорость зависит от драйверов, троттлинга, других процессов, версии
# библиотек и т.п. Поэтому после каждого успешного запуска транскрипции,
# разделения дорожек или перевода мы измеряем реальную скорость
# (workload/elapsed) и копим экспоненциальное скользящее среднее по ключу
# (тип задачи, модель, устройство) — так прогноз со временем становится
# всё точнее именно для железа пользователя, а не усреднённым по интернету.
SPEED_STATS_FILE = BASE_DIR / "speed_stats.json"
_speed_lock = threading.Lock()

def _load_speed_stats() -> dict:
    try:
        if SPEED_STATS_FILE.exists():
            return json.loads(SPEED_STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _speed_key(task: str, model_key: str, device: str) -> str:
    return f"{task}:{model_key}:{device}"

def _record_speed(task: str, model_key: str, device: str, workload: float, elapsed: float):
    """workload — секунды аудио (transcribe/separate) или символы текста
    (translate). Игнорируем совсем короткие/нулевые прогоны — на них скорость
    считается ненадёжно (перекос из-за времени загрузки модели) и только
    портила бы статистику."""
    if elapsed <= 1.0 or workload <= 0:
        return
    rate = workload / elapsed
    with _speed_lock:
        stats = _load_speed_stats()
        k = _speed_key(task, model_key, device)
        prev = stats.get(k)
        if prev and prev.get("n", 0) > 0 and prev.get("rate", 0) > 0:
            alpha = 0.35  # чем выше — тем быстрее забываем старые замеры
            new_rate = prev["rate"] * (1 - alpha) + rate * alpha
            n = min(prev.get("n", 0) + 1, 50)
        else:
            new_rate = rate
            n = 1
        stats[k] = {"rate": round(new_rate, 5), "n": n}
        try:
            SPEED_STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[SPEED] Не удалось сохранить статистику скорости: {e}")

# Запасные (очень грубые) оценки — используются только пока для этой пары
# "модель+устройство" ещё нет ни одного реального замера. Единицы: секунды
# обработанного контента за 1 секунду счёта (для translate — символы/сек).
_HEURISTIC_RATE = {
    ("transcribe", "cuda"): 12.0,
    ("transcribe", "cpu"):  1.5,
    ("separate",   "cuda"): 8.0,
    ("separate",   "cpu"):  0.5,
    ("translate",  "cuda"): 150.0,
    ("translate",  "cpu"):  25.0,
}
_HEURISTIC_BASELINE_MB = {"transcribe": 1550, "separate": 300, "translate": 2400}

def _model_size_mb(task: str, model_key: str) -> float:
    if task == "separate":
        return DEMUCS_MODELS.get(model_key, {}).get("size_mb", 300)
    return HF_MODELS.get(model_key, {}).get("size_mb", 1500)

def _estimate_seconds(task: str, model_key: str, device: str, workload: float) -> dict:
    if workload <= 0:
        return {"seconds": None, "confidence": "unknown"}
    stats = _load_speed_stats()
    entry = stats.get(_speed_key(task, model_key, device))
    if entry and entry.get("rate", 0) > 0:
        # Реальный замер уже включает в себя всё (в т.ч. 4х для htdemucs_ft,
        # т.к. elapsed на ансамбле и так был в ~4 раза больше) — никаких
        # дополнительных поправок сверху не накручиваем.
        confidence = "measured" if entry.get("n", 0) >= 3 else "rough"
        return {"seconds": round(workload / entry["rate"]), "confidence": confidence,
                "samples": entry.get("n", 0)}
    rate = _HEURISTIC_RATE.get((task, device))
    if not rate:
        return {"seconds": None, "confidence": "unknown"}
    size_mb = _model_size_mb(task, model_key)
    baseline = _HEURISTIC_BASELINE_MB.get(task, 1500)
    size_factor = max(0.4, min(4.0, size_mb / baseline))  # крупнее модель — пропорционально медленнее
    if task == "separate" and model_key == "htdemucs_ft":
        # htdemucs_ft — это 4 модели в ансамбле, а не 1; size_mb (340) сам
        # по себе даёт лишь ~1.1x от базовой htdemucs (300) и не отражает
        # реальную 4х-нагрузку — учитываем её явно, но только в эвристике
        # "на глазок" (реальный замер, см. выше, эту нагрузку уже содержит).
        size_factor *= 4
    rate = rate / size_factor
    return {"seconds": round(workload / rate), "confidence": "heuristic", "samples": 0}

@app.route("/api/estimate")
def api_estimate():
    """Прогноз времени для выбранной модели ДО запуска обработки.
    task: transcribe | separate | translate. workload считается сервером
    по данным проекта, чтобы фронтенд не дублировал логику подсчёта
    длительности аудио / объёма текста."""
    task      = request.args.get("task", "")
    model_key = request.args.get("model_key", "")
    pid       = request.args.get("pid", "")
    if task not in ("transcribe", "separate", "translate"):
        return jsonify({"error": "Неизвестный тип задачи"}), 400

    device = _torch_device()
    workload = 0.0
    try:
        if task in ("transcribe", "separate"):
            wav = UPLOAD_DIR / f"{pid}.wav"
            if wav.exists():
                workload = len(load_wav(wav)) / 16000.0
        elif task == "translate":
            p = load_proj(pid)
            if p:
                workload = float(sum(len(s.get("text","")) for s in p.get("segments", [])))
    except Exception as e:
        print(f"[ESTIMATE] Не удалось посчитать объём работы: {e}")

    result = _estimate_seconds(task, model_key, device, workload)
    result["device"] = device
    result["workload"] = round(workload, 1)
    return jsonify(result)



def _get_save_dir() -> Path:
    """Return user-configured save dir, or Downloads as fallback."""
    custom = _load_settings().get("save_dir", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    return _get_downloads_dir()


def _unique_dst(folder: Path, filename: str) -> Path:
    """Return a unique path: if file exists add (1), (2), …"""
    dst = folder / filename
    if not dst.exists():
        return dst
    stem_p, ext = os.path.splitext(filename)
    i = 1
    while (folder / f"{stem_p} ({i}){ext}").exists():
        i += 1
    return folder / f"{stem_p} ({i}){ext}"


def _save_bytes(data: bytes, filename: str, folder: Path | None = None) -> Path:
    dl = folder or _get_save_dir(); dl.mkdir(parents=True, exist_ok=True)
    dst = _unique_dst(dl, filename); dst.write_bytes(data); return dst


def _save_copy(src: Path, filename: str, folder: Path | None = None) -> Path:
    dl = folder or _get_save_dir(); dl.mkdir(parents=True, exist_ok=True)
    dst = _unique_dst(dl, filename); shutil.copy2(str(src), str(dst)); return dst

@app.route("/api/projects/<pid>/export/csv")
def exp_csv(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    import csv, io
    has_tr = any(s.get("translation") for s in p.get("segments",[]))
    buf = io.StringIO(); w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    hdrs = ["Персонаж","Начало реплики","Конец реплики","Текст реплики"]
    if has_tr: hdrs.append("Перевод")
    w.writerow(hdrs)
    for s in p.get("segments",[]):
        row = [s.get("character",""), s["start_ts"], s["end_ts"], s["text"]]
        if has_tr: row.append(s.get("translation",""))
        w.writerow(row)
    return Response("\ufeff"+buf.getvalue(), mimetype="text/csv; charset=utf-8-sig",
                    headers={"Content-Disposition":f'attachment; filename="{p["name"]}.csv"'})

@app.route("/api/projects/<pid>/export/srt")
def exp_srt(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    has_tr = any(s.get("translation") for s in p.get("segments",[])); lines = []
    for i, s in enumerate(p.get("segments",[]), 1):
        char = s.get("character","").strip()
        text = f"{char}: {s['text']}" if char else s["text"]
        if has_tr and s.get("translation"): text += f"\n{s['translation']}"
        lines += [str(i), f"{s['start_ts']} --> {s['end_ts']}", text, ""]
    return Response("\n".join(lines), mimetype="text/plain; charset=utf-8",
                    headers={"Content-Disposition":f'attachment; filename="{p["name"]}.srt"'})

@app.route("/api/projects/<pid>/export/json")
def exp_json(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    has_tr = any(s.get("translation") for s in p.get("segments",[]))
    def seg_dict(s):
        d = {"character":s.get("character",""),"start":s["start_ts"],"end":s["end_ts"],"text":s["text"]}
        if s.get("chapter"): d["chapter"] = s["chapter"]
        if has_tr: d["translation"] = s.get("translation","")
        return d
    data = {"project":p["name"],"exported_at":datetime.now().isoformat(),
            "character_colors":p.get("character_colors",{}) or {},
            "segments":[seg_dict(s) for s in p.get("segments",[])]}
    return Response(json.dumps(data, ensure_ascii=False, indent=2), mimetype="application/json",
                    headers={"Content-Disposition":f'attachment; filename="{p["name"]}.json"'})

@app.route("/api/projects/<pid>/export/docx")
def exp_docx(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    cols_arg = request.args.get("columns")
    columns = [c.strip() for c in cols_arg.split(",") if c.strip()] if cols_arg else None
    out = EXPORT_DIR / f"{pid}.docx"
    _gen_docx_to_path(p, out, columns)
    return send_file(str(out), as_attachment=True, download_name=f"{p['name']}.docx",
                     mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

@app.route("/api/projects/<pid>/export/archive")
def exp_archive(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Проект не найден"}), 404
    import zipfile; from flask import after_this_request
    safe_name = "".join(c for c in p["name"] if c.isalnum() or c in " ._-").strip()[:50] or pid
    tmp_path = EXPORT_DIR / f"{pid}_archive.tsproject"
    try:
        with zipfile.ZipFile(str(tmp_path),"w") as zf:
            zf.writestr("project.json", json.dumps(p, ensure_ascii=False, indent=2), zipfile.ZIP_DEFLATED)
            wav = UPLOAD_DIR / f"{pid}.wav"
            if wav.exists(): zf.write(str(wav), "audio.wav", zipfile.ZIP_STORED)
            if p.get("has_video"):
                ext = p.get("video_ext",".mp4"); vid = UPLOAD_DIR / f"{pid}_video{ext}"
                if vid.exists(): zf.write(str(vid), f"video{ext}", zipfile.ZIP_STORED)
    except Exception as e:
        tmp_path.unlink(missing_ok=True); return jsonify({"error":str(e)}), 500
    @after_this_request
    def _cleanup(response):
        try: tmp_path.unlink(missing_ok=True)
        except: pass
        return response
    return send_file(str(tmp_path), mimetype="application/zip",
                     as_attachment=True, download_name=f"{safe_name}.tsproject")



# ─────────────────────────────────────────────────────────────────────────────
# SAVE-TO-DOWNLOADS ROUTES  (called from JS, return {"ok":True,"path":"…"})
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/projects/<pid>/export/csv/save", methods=["POST"])
def exp_csv_save(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    import csv, io
    has_tr = any(s.get("translation") for s in p.get("segments",[]))
    buf = io.StringIO(); w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    hdrs = ["Персонаж","Начало реплики","Конец реплики","Текст реплики"]
    if has_tr: hdrs.append("Перевод")
    w.writerow(hdrs)
    for s in p.get("segments",[]):
        row = [s.get("character",""), s["start_ts"], s["end_ts"], s["text"]]
        if has_tr: row.append(s.get("translation",""))
        w.writerow(row)
    try:
        dst = _save_bytes(("\ufeff"+buf.getvalue()).encode("utf-8-sig"), f"{p['name']}.csv")
        return jsonify({"ok":True,"path":str(dst)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/projects/<pid>/export/srt/save", methods=["POST"])
def exp_srt_save(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    has_tr = any(s.get("translation") for s in p.get("segments",[])); lines = []
    for i, s in enumerate(p.get("segments",[]), 1):
        char = s.get("character","").strip()
        text = f"{char}: {s['text']}" if char else s["text"]
        if has_tr and s.get("translation"): text += f"\n{s['translation']}"
        lines += [str(i), f"{s['start_ts']} --> {s['end_ts']}", text, ""]
    try:
        dst = _save_bytes("\n".join(lines).encode("utf-8"), f"{p['name']}.srt")
        return jsonify({"ok":True,"path":str(dst)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/projects/<pid>/export/json/save", methods=["POST"])
def exp_json_save(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    has_tr = any(s.get("translation") for s in p.get("segments",[]))
    def seg_dict(s):
        d = {"character":s.get("character",""),"start":s["start_ts"],"end":s["end_ts"],"text":s["text"]}
        if s.get("chapter"): d["chapter"] = s["chapter"]
        if has_tr: d["translation"] = s.get("translation","")
        return d
    data = {"project":p["name"],"exported_at":datetime.now().isoformat(),
            "character_colors":p.get("character_colors",{}) or {},
            "segments":[seg_dict(s) for s in p.get("segments",[])]}
    try:
        _dj2 = request.json or {}; _sdj2 = Path(_dj2["save_dir"]) if _dj2.get("save_dir") else None
        dst = _save_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"), f"{p['name']}.json", folder=_sdj2)
        return jsonify({"ok":True,"path":str(dst)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/api/projects/<pid>/export/docx/save", methods=["POST"])
def exp_docx_save(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Не найден"}), 404
    columns = (request.json or {}).get("columns")
    tmp = EXPORT_DIR / f"{pid}_save.docx"
    _gen_docx_to_path(p, tmp, columns)
    try:
        dst = _save_copy(tmp, f"{p['name']}.docx")
        tmp.unlink(missing_ok=True)
        return jsonify({"ok":True,"path":str(dst)})
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return jsonify({"error":str(e)}), 500


@app.route("/api/projects/<pid>/export/archive/save", methods=["POST"])
def exp_archive_save(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error":"Проект не найден"}), 404
    import zipfile
    safe_name = "".join(c for c in p["name"] if c.isalnum() or c in " ._-").strip()[:50] or pid
    tmp_path = EXPORT_DIR / f"{pid}_archive_save.tsproject"
    try:
        with zipfile.ZipFile(str(tmp_path),"w") as zf:
            zf.writestr("project.json", json.dumps(p, ensure_ascii=False, indent=2), zipfile.ZIP_DEFLATED)
            wav = UPLOAD_DIR / f"{pid}.wav"
            if wav.exists(): zf.write(str(wav), "audio.wav", zipfile.ZIP_STORED)
            if p.get("has_video"):
                ext = p.get("video_ext",".mp4"); vid = UPLOAD_DIR / f"{pid}_video{ext}"
                if vid.exists(): zf.write(str(vid), f"video{ext}", zipfile.ZIP_STORED)
        dst = _save_copy(tmp_path, f"{safe_name}.tsproject")
        tmp_path.unlink(missing_ok=True)
        return jsonify({"ok":True,"path":str(dst)})
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error":str(e)}), 500


@app.route("/api/projects/<pid>/separated/<model>/<stem>/save", methods=["POST"])
def api_separated_save(pid, model, stem):
    if stem not in ("vocals", "no_vocals"):
        return jsonify({"error":"Неверный stem"}), 400
    f = _sep_file(pid, model, stem)
    if not f.exists():
        return jsonify({"error":"Файл не найден — запустите обработку"}), 404
    p = load_proj(pid); proj_name = p["name"] if p else pid
    label = "голос" if stem == "vocals" else "без_голоса"
    _dss = request.json or {}; _sds = Path(_dss["save_dir"]) if _dss.get("save_dir") else None
    try:
        dst = _save_copy(f, f"{proj_name}_{label}.wav", folder=_sds)
        return jsonify({"ok":True,"path":str(dst)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/projects/<pid>/separated/<model>/<stem>/save_video", methods=["POST"])
def api_separated_save_video(pid, model, stem):
    """Merge the separated audio stem back into the original video and save to Downloads."""
    if stem not in ("vocals", "no_vocals"):
        return jsonify({"error": "Неверный stem"}), 400

    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404
    if not p.get("has_video"):
        return jsonify({"error": "У проекта нет видео"}), 400

    audio_wav = _sep_file(pid, model, stem)
    if not audio_wav.exists():
        return jsonify({"error": "Обработанное аудио не найдено — запустите разделение"}), 404

    video_ext = p.get("video_ext", ".mp4")
    video_src = UPLOAD_DIR / f"{pid}_video{video_ext}"
    if not video_src.exists():
        return jsonify({"error": f"Исходное видео не найдено: {video_src.name}"}), 404

    proj_name = p.get("name", pid)
    label = "голос" if stem == "vocals" else "без_голоса"
    out_name = f"{proj_name}_{label}.mp4"

    tmp_out = EXPORT_DIR / f"{pid}_{stem}_video_tmp.mp4"
    try:
        # Use ffmpeg to replace audio: copy video stream, encode audio as AAC
        ff_bin = str(_local_ff) if _local_ff.exists() else "ffmpeg"
        cmd = [
            ff_bin,
            "-i", str(video_src),   # input 0: original video
            "-i", str(audio_wav),   # input 1: processed audio
            "-map", "0:v:0",        # take video from input 0
            "-map", "1:a:0",        # take audio from input 1
            "-c:v", "copy",         # copy video — no re-encoding, fast
            "-c:a", "aac",          # encode audio as AAC for MP4
            "-b:a", "192k",
            "-shortest",            # end when shorter stream ends
            "-y",                   # overwrite tmp if exists
            str(tmp_out),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-600:]
            return jsonify({"error": f"FFmpeg завершился с ошибкой:\n{err}"}), 500

        dst = _save_copy(tmp_out, out_name)
        return jsonify({"ok": True, "path": str(dst)})

    except subprocess.TimeoutExpired:
        return jsonify({"error": "FFmpeg: превышено время ожидания (10 мин)"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: tmp_out.unlink(missing_ok=True)
        except: pass



# ─────────────────────────────────────────────────────────────────────────────
# HUB — generation helpers
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

def _sanitize_filename(name: str) -> str:
    return _re.sub(r'[<>:"/\\|?*]', '_', name).strip().strip('.')[:100] or "unnamed"


def _save_to_hub_folder(folder_name: str, filename: str,
                         data: bytes = None, src_path: Path = None,
                         base_dir: Path | None = None) -> Path:
    root = base_dir if (base_dir and base_dir.exists()) else _get_save_dir()
    folder = root / _sanitize_filename(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / filename
    if data is not None:
        dst.write_bytes(data)
    elif src_path is not None:
        shutil.copy2(str(src_path), str(dst))
    return dst


def _gen_csv_bytes(p: dict) -> bytes:
    import csv, io
    has_tr = any(s.get("translation") for s in p.get("segments", []))
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    hdrs = ["Персонаж", "Начало реплики", "Конец реплики", "Текст реплики"]
    if has_tr: hdrs.append("Перевод")
    w.writerow(hdrs)
    for s in p.get("segments", []):
        row = [s.get("character", ""), s["start_ts"], s["end_ts"], s["text"]]
        if has_tr: row.append(s.get("translation", ""))
        w.writerow(row)
    return ("\ufeff" + buf.getvalue()).encode("utf-8-sig")


def _gen_srt_bytes(p: dict) -> bytes:
    has_tr = any(s.get("translation") for s in p.get("segments", []))
    lines = []
    for i, s in enumerate(p.get("segments", []), 1):
        char = s.get("character", "").strip()
        text = f"{char}: {s['text']}" if char else s["text"]
        if has_tr and s.get("translation"): text += f"\n{s['translation']}"
        lines += [str(i), f"{s['start_ts']} --> {s['end_ts']}", text, ""]
    return "\n".join(lines).encode("utf-8")


def _gen_json_bytes(p: dict) -> bytes:
    has_tr = any(s.get("translation") for s in p.get("segments", []))
    def seg_dict(s):
        d = {"character": s.get("character",""), "start": s["start_ts"],
             "end": s["end_ts"], "text": s["text"]}
        if has_tr: d["translation"] = s.get("translation", "")
        return d
    data = {"project": p["name"], "exported_at": datetime.now().isoformat(),
            "segments": [seg_dict(s) for s in p.get("segments", [])]}
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


DOCX_COLUMN_SPECS = {
    # weight определяет долю общей ширины таблицы; text всегда самый
    # "тяжёлый" столбец, поэтому ему и достаётся основной объём.
    "character":   {"label": "Персонаж",       "weight": 1.3},
    "start_ts":    {"label": "Начало",          "weight": 1.3},
    "end_ts":      {"label": "Конец",           "weight": 1.3},
    "text":        {"label": "Текст реплики",   "weight": 6.0},
    "translation": {"label": "Перевод",         "weight": 3.0},
}
DOCX_TABLE_WIDTH_CM = 18.0

def _gen_docx_to_path(p: dict, out_path: Path, columns: list | None = None):
    from docx import Document; from docx.shared import Pt, Cm, RGBColor
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn; from docx.oxml import OxmlElement
    segments = p.get("segments", [])
    char_colors = p.get("character_colors", {}) or {}
    has_tr = any(s.get("translation") for s in segments)

    all_cols = ["character", "start_ts", "end_ts", "text"] + (["translation"] if has_tr else [])
    if columns:
        # "text" всегда остаётся, даже если её случайно не передали или
        # сняли галочку — без неё файл теряет смысл как транскрипт.
        included = [c for c in all_cols if c in columns]
        if "text" not in included: included.append("text")
        included = [c for c in all_cols if c in included]  # вернуть исходный порядок
    else:
        included = all_cols
    cols = len(included)
    total_weight = sum(DOCX_COLUMN_SPECS[c]["weight"] for c in included)
    widths = [Cm(DOCX_TABLE_WIDTH_CM * DOCX_COLUMN_SPECS[c]["weight"] / total_weight) for c in included]
    hdrs_txt = [DOCX_COLUMN_SPECS[c]["label"] for c in included]
    char_col_idx = included.index("character") if "character" in included else None

    doc = Document()
    titl = doc.add_heading(p["name"], level=1); titl.alignment = 1
    doc.add_paragraph(f"Дата экспорта: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    chapters = [s.get("chapter") for s in segments if s.get("chapter")]
    if chapters:
        doc.add_heading("Оглавление", level=2)
        for ch in chapters:
            doc.add_paragraph(ch, style="List Bullet")

    doc.add_paragraph("")

    def _cell_highlight(cell, hex_color: str):
        """Заливка ячейки цветом персонажа ("маркер"), а не рамка — рамки
        на соседних ячейках визуально сливались и перекрывали друг друга.
        Плюс автоконтраст текста: на тёмной заливке буквы становятся
        белыми, иначе остаются чёрными — иначе текст на тёмном цвете
        просто не читается."""
        hex_color = hex_color.lstrip("#").upper()
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)
        r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
        luminance = 0.299*r + 0.587*g + 0.114*b  # воспринимаемая яркость
        text_rgb = RGBColor(0xFF,0xFF,0xFF) if luminance < 140 else RGBColor(0x00,0x00,0x00)
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.color.rgb = text_rgb

    def _new_table():
        tbl = doc.add_table(rows=1, cols=cols)
        tbl.style = "Table Grid"; tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        # Без этого Word может пересчитать ширину столбцов по содержимому
        # при открытии файла, игнорируя заданные cm — именно поэтому при
        # малом числе столбцов пропорции "уезжали", а при трёх выглядели
        # нормально (там автоподбор случайно давал похожий результат).
        tbl.autofit = False
        tblPr = tbl._tbl.tblPr
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed")
        tblPr.append(layout)
        for i, w in enumerate(widths):
            tbl.columns[i].width = w
            for cell in tbl.columns[i].cells: cell.width = w
        hdr = tbl.rows[0].cells
        for i, txt in enumerate(hdrs_txt):
            hdr[i].text = txt
            for run in hdr[i].paragraphs[0].runs: run.bold = True
            tc = hdr[i]._tc; tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"),"clear"); shd.set(qn("w:color"),"auto")
            shd.set(qn("w:fill"),"D9D9D9"); tcPr.append(shd)
        return tbl

    seg_val = {
        "character":   lambda s: s.get("character",""),
        "start_ts":    lambda s: s["start_ts"],
        "end_ts":      lambda s: s["end_ts"],
        "text":        lambda s: s["text"],
        "translation": lambda s: s.get("translation",""),
    }

    tbl = _new_table()
    for seg in segments:
        if seg.get("chapter"):
            # Таблицу в .docx нельзя "прервать" параграфом посередине —
            # закрываем текущую и начинаем новую после заголовка главы.
            doc.add_paragraph("")
            doc.add_heading(seg["chapter"], level=2)
            tbl = _new_table()
        row = tbl.add_row().cells
        for i, c in enumerate(included):
            row[i].width = widths[i]  # новые строки не наследуют ширину сами
            row[i].text = seg_val[c](seg)
        for cell in row:
            for para in cell.paragraphs:
                for run in para.runs: run.font.size = Pt(10)
        # Имя персонажа выделяем цветом-маркером — только если у
        # персонажа есть имя И для него явно выбран цвет в приложении.
        char_name = seg.get("character", "")
        if char_col_idx is not None and char_name and char_colors.get(char_name):
            _cell_highlight(row[char_col_idx], char_colors[char_name])
    doc.save(str(out_path))


# ─────────────────────────────────────────────────────────────────────────────
# HUB — routes
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# YOUTUBE DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _yt_dlp_bin() -> str:
    """yt-dlp executable — portable env first, then PATH."""
    local = BASE_DIR / "python" / "Scripts" / "yt-dlp.exe"
    if local.exists(): return str(local)
    local2 = BASE_DIR / "python" / "bin" / "yt-dlp"
    if local2.exists(): return str(local2)
    return "yt-dlp"


@app.route("/api/youtube/info", methods=["POST"])
def api_youtube_info():
    """Return available formats for a YouTube URL (no download)."""
    d = request.json or {}
    url = d.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL не указан"}), 400
    try:
        import yt_dlp  # noqa
    except ImportError:
        return jsonify({"error": "yt-dlp не установлен. Запустите setup.bat снова."}), 500

    try:
        browser   = d.get("browser", "")
        cookie_pid = d.get("cookie_pid", "")  # hub project pid used as cookie file key
        def _cookie_opts(base: dict) -> dict:
            if browser:
                base["cookiesfrombrowser"] = (browser, None, None, None)
            elif cookie_pid:
                cf = UPLOAD_DIR / f"_cookies_{cookie_pid}.txt"
                if cf.exists():
                    base["cookiefile"] = str(cf)
            return base

        import re as _re
        # Strip YouTube tracking params (?si=, ?t=) that can confuse some yt-dlp versions
        clean_url = _re.sub(r'[?&](si|utm_[a-z]+)=[^&]*', '', url).rstrip('?&')

        info = None
        last_err = None
        base_cookie_opts = _cookie_opts({})

        def _is_cookie_decrypt_error(msg: str) -> bool:
            m = (msg or "").lower()
            return "decrypt" in m and ("dpapi" in m or "cookie" in m or "keyring" in m)

        # Strategy 1: process=False → get raw extractor data (title/thumb/duration always work)
        #             then process_ie_result(download=False) → populate formats without downloading
        raw_info = None
        try:
            opts = {**base_cookie_opts, "quiet": True, "no_warnings": True, "noplaylist": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                raw_info = ydl.extract_info(clean_url, download=False, process=False)
                if raw_info:
                    # Try to process formats from raw data
                    try:
                        info = ydl.process_ie_result(dict(raw_info), download=False)
                    except Exception:
                        info = ydl.sanitize_info(raw_info)
        except Exception as e:
            last_err = e

        # Chrome/Edge cookie DB иногда невозможно расшифровать (DPAPI /
        # App-Bound Encryption шифрование новых версий Chromium) — если явно
        # выбранный браузер упал именно с такой ошибкой, тихо пробуем другие
        # установленные браузеры, прежде чем сдаваться. Firefox не использует
        # DPAPI и чаще всего срабатывает, если он установлен и залогинен.
        if not raw_info and browser and last_err and _is_cookie_decrypt_error(str(last_err)):
            for br in ("firefox", "edge", "brave", "chrome"):
                if br == browser:
                    continue
                try:
                    opts = {"cookiesfrombrowser": (br, None, None, None),
                            "quiet": True, "no_warnings": True, "noplaylist": True}
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        raw_info = ydl.extract_info(clean_url, download=False, process=False)
                        if raw_info:
                            browser = br  # запоминаем рабочий браузер для Strategy 2 и скачивания
                            base_cookie_opts = _cookie_opts({})
                            try:
                                info = ydl.process_ie_result(dict(raw_info), download=False)
                            except Exception:
                                info = ydl.sanitize_info(raw_info)
                            last_err = None
                            break
                except Exception as e2:
                    last_err = e2

        # YouTube в 2026 году периодически принудительно переводит часть
        # клиентов плеера (обычно tv_downgraded/web по умолчанию для
        # залогиненных сессий) на потоковую передачу SABR, которую конкретная
        # версия yt-dlp может ещё не поддерживать — проявляется как раз как
        # "The page needs to be reloaded" (см. yt-dlp/yt-dlp#17389). Пока
        # разработчики yt-dlp это не починили, штатный обход — явно указать
        # другой клиент плеера. Пробуем по очереди, пока формат не найдётся.
        def _is_player_reload_error(msg: str) -> bool:
            m = (msg or "").lower()
            return "page needs to be reloaded" in m or "unplayable" in m

        if not raw_info and last_err and _is_player_reload_error(str(last_err)):
            for client in ("web_embedded", "android", "ios", "tv"):
                try:
                    opts = {**base_cookie_opts,
                            "extractor_args": {"youtube": {"player_client": [client]}},
                            "quiet": True, "no_warnings": True, "noplaylist": True}
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        raw_info = ydl.extract_info(clean_url, download=False, process=False)
                        if raw_info:
                            _yt_player_client = client  # запомним для скачивания ниже
                            try:
                                info = ydl.process_ie_result(dict(raw_info), download=False)
                            except Exception:
                                info = ydl.sanitize_info(raw_info)
                            last_err = None
                            break
                except Exception as e3:
                    last_err = e3
            else:
                _yt_player_client = None
        else:
            _yt_player_client = None

        # Strategy 2: check_formats=False skips HTTP HEAD checks that block restricted videos
        if not info or not info.get("formats"):
            for fmt in (None, "bestvideo*+bestaudio/best", "best", "bestaudio"):
                try:
                    o = {**base_cookie_opts, "quiet": True, "no_warnings": True,
                         "check_formats": False, "noplaylist": True}
                    if fmt:
                        o["format"] = fmt
                    with yt_dlp.YoutubeDL(o) as ydl:
                        candidate = ydl.extract_info(clean_url, download=False)
                    if candidate and candidate.get("formats"):
                        info = candidate
                        last_err = None
                        break
                except Exception as e:
                    last_err = e

        # Strategy 3: merge — use raw_info metadata + whatever formats we got
        if raw_info and (not info or not info.get("formats")):
            if info:
                raw_info.update({k: v for k, v in info.items() if v})
            info = raw_info  # at minimum shows title/thumb even if formats empty

        if not info:
            raise (last_err or RuntimeError("Не удалось получить информацию о видео"))

        # Ensure formats list exists (may be empty for restricted videos)
        if "formats" not in info:
            info["formats"] = []

        title   = info.get("title", "Без названия")
        channel = info.get("channel") or info.get("uploader", "")
        duration_s = info.get("duration", 0) or 0
        mins, secs  = divmod(int(duration_s), 60)
        duration_str = f"{mins}:{secs:02d}"
        thumb = info.get("thumbnail", "")

        formats = info.get("formats", [])

        # Build clean format list
        seen_keys: set = set()
        result_fmts: list = []

        # ── Video+Audio combined formats ──
        for f in formats:
            vcodec = f.get("vcodec", "none") or "none"
            acodec = f.get("acodec", "none") or "none"
            if vcodec == "none" or acodec == "none":
                continue
            h = f.get("height") or 0
            fps = f.get("fps") or 0
            fsize = f.get("filesize") or f.get("filesize_approx") or 0
            ext = f.get("ext", "mp4")
            key = f"{h}p_{ext}"
            if key in seen_keys: continue
            seen_keys.add(key)
            result_fmts.append({
                "id":       f["format_id"],
                "type":     "video",
                "label":    f"{h}p{'+' + str(int(fps)) + 'fps' if fps and fps > 30 else ''} · {ext.upper()}",
                "height":   h,
                "fps":      fps,
                "ext":      ext,
                "size_bytes": fsize,
                "size_str": _fmt_bytes(fsize),
                "note":     f.get("format_note",""),
            })

        # ── Best video (no audio) + best audio via merge — added as top options ──
        best_video_ids = {}
        for f in formats:
            vcodec = f.get("vcodec","none") or "none"
            acodec = f.get("acodec","none") or "none"
            if vcodec == "none" or acodec != "none": continue
            h = f.get("height") or 0
            if h not in best_video_ids or (f.get("tbr") or 0) > (best_video_ids[h].get("tbr") or 0):
                best_video_ids[h] = f
        best_audio = None
        for f in formats:
            vcodec = f.get("vcodec","none") or "none"
            acodec = f.get("acodec","none") or "none"
            if vcodec != "none" or acodec == "none": continue
            if best_audio is None or (f.get("abr") or 0) > (best_audio.get("abr") or 0):
                best_audio = f
        for h_val in sorted(best_video_ids.keys(), reverse=True):
            vf = best_video_ids[h_val]
            key = f"{h_val}p_merged"
            if key in seen_keys: continue
            seen_keys.add(key)
            vid_size  = vf.get("filesize") or vf.get("filesize_approx") or 0
            aud_size  = (best_audio.get("filesize") or best_audio.get("filesize_approx") or 0) if best_audio else 0
            total_size = vid_size + aud_size
            aud_id = best_audio["format_id"] if best_audio else "bestaudio"
            fps = vf.get("fps") or 0
            result_fmts.append({
                "id":       f"{vf['format_id']}+{aud_id}",
                "type":     "video",
                "label":    f"{h_val}p{'+' + str(int(fps)) + 'fps' if fps and fps > 30 else ''} · MP4 (лучшее качество)",
                "height":   h_val,
                "fps":      fps,
                "ext":      "mp4",
                "size_bytes": total_size,
                "size_str": _fmt_bytes(total_size) if total_size else "~неизв.",
                "note":     "объединение потоков",
            })

        # ── Audio-only formats ──
        seen_audio: set = set()
        for f in formats:
            vcodec = f.get("vcodec","none") or "none"
            acodec = f.get("acodec","none") or "none"
            if vcodec != "none" or acodec == "none": continue
            abr = int(f.get("abr") or 0)
            ext = f.get("ext","m4a")
            akey = f"{abr}_{ext}"
            if akey in seen_audio: continue
            seen_audio.add(akey)
            fsize = f.get("filesize") or f.get("filesize_approx") or 0
            result_fmts.append({
                "id":       f["format_id"],
                "type":     "audio",
                "label":    f"Только аудио · {abr if abr else '?'} kbps · {ext.upper()}",
                "height":   0,
                "fps":      0,
                "ext":      ext,
                "size_bytes": fsize,
                "size_str": _fmt_bytes(fsize),
                "note":     f.get("format_note",""),
            })

        # Sort: video desc by height, then audio
        result_fmts.sort(key=lambda x: (-x["height"], -x.get("fps",0)))

        # ── Fallback generics when format detection yields nothing ──
        # (happens with restricted/region-locked videos that still allow download)
        if not result_fmts:
            n_raw = len(formats)
            result_fmts = [
                {
                    "id": "bestvideo+bestaudio/best",
                    "type": "video",
                    "label": "Лучшее качество (авто)",
                    "height": 9999, "fps": 0, "ext": "mp4",
                    "size_bytes": 0, "size_str": "",
                    "note": f"yt-dlp выберет сам · {n_raw} потоков найдено" if n_raw else "yt-dlp выберет сам",
                },
                {
                    "id": "bestvideo[height<=?1080]+bestaudio/bestvideo+bestaudio/best",
                    "type": "video",
                    "label": "До 1080p (авто)",
                    "height": 1080, "fps": 0, "ext": "mp4",
                    "size_bytes": 0, "size_str": "", "note": "",
                },
                {
                    "id": "bestvideo[height<=?720]+bestaudio/bestvideo+bestaudio/best",
                    "type": "video",
                    "label": "До 720p (авто)",
                    "height": 720, "fps": 0, "ext": "mp4",
                    "size_bytes": 0, "size_str": "", "note": "",
                },
                {
                    "id": "bestaudio/best",
                    "type": "audio",
                    "label": "Только аудио (авто)",
                    "height": 0, "fps": 0, "ext": "m4a",
                    "size_bytes": 0, "size_str": "", "note": "",
                },
            ]

        return jsonify({
            "ok":       True,
            "title":    title,
            "channel":  channel,
            "duration": duration_str,
            "thumbnail": thumb,
            "formats":  result_fmts,
            "raw_count": len(formats),   # diagnostic
            "player_client": _yt_player_client,  # если понадобился обход "The page needs to be reloaded"
        })
    except Exception as e:
        raw = str(e)
        # Provide a helpful hint while still showing the real error
        if "No module named yt_dlp" in raw or "ImportError" in raw:
            hint = "yt-dlp не установлен — запустите setup.bat. "
        elif "page needs to be reloaded" in raw.lower() or "нужно перезагрузить страницу" in raw.lower():
            hint = ("YouTube обновил логику плеера, а установленная версия yt-dlp с ней ещё "
                    "не справляется (или у yt-dlp завис устаревший закэшированный код "
                    "плеера). Запустите UPDATE_YTDLP.bat в папке программы — он обновит "
                    "yt-dlp и очистит этот кэш. ")
        elif "decrypt" in raw.lower() and ("dpapi" in raw.lower() or "keyring" in raw.lower()):
            hint = ("Не удалось прочитать cookies браузера — Chrome/Edge шифрует их так, "
                    "что стороннее приложение не всегда может их расшифровать (мы уже "
                    "попробовали автоматически другие установленные браузеры, включая "
                    "Firefox, но ни один не подошёл). Самый надёжный вариант — "
                    "«📄 Загрузить cookies.txt», экспортировав файл расширением вроде "
                    "'Get cookies.txt LOCALLY' из браузера, где вы залогинены в YouTube. ")
        elif "Could not copy" in raw and "cookie" in raw.lower():
            hint = ("Chrome заблокировал доступ к cookies пока работает. "
                    "Используйте вместо этого файл cookies.txt — "
                    "нажмите «📄 Загрузить cookies.txt» и следуйте инструкции. ")
        elif "Sign in to confirm" in raw or "not a bot" in raw.lower() or "botguard" in raw.lower():
            hint = ("YouTube требует авторизации. "
                    "Загрузите файл cookies.txt через кнопку «📄 Загрузить cookies.txt». ")
        elif "Sign in" in raw or "confirm your age" in raw.lower():
            hint = "Видео требует авторизации YouTube. Загрузите cookies.txt. "
        elif "Private video" in raw:
            hint = "Приватное видео — доступно только автору. "
        elif "not available" in raw.lower() or "Video unavailable" in raw:
            hint = ("Видео недоступно. Возможно YouTube требует авторизации — "
                    "попробуйте загрузить cookies.txt. ")
        else:
            hint = ""
        return jsonify({"error": hint + raw}), 400


_yt_status: dict = {}   # pid -> {status, progress, stage, error}
_yt_lock = threading.Lock()


@app.route("/api/youtube/download", methods=["POST"])
def api_youtube_download():
    """Start async YouTube download → returns pid to poll."""
    d = request.json or {}
    url       = d.get("url", "").strip()
    fmt_id    = d.get("format_id", "bestvideo+bestaudio/best")
    ext       = d.get("ext", "mp4")
    title     = d.get("title", "youtube")
    if not url:
        return jsonify({"error": "URL не указан"}), 400

    pid = str(uuid.uuid4())
    safe_title = _sanitize_filename(title) or "youtube"
    tmp_dir = UPLOAD_DIR / f"_yt_{pid}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    browser = d.get("browser", "")

    with _yt_lock:
        _yt_status[pid] = {"status": "downloading", "progress": 0, "stage": "Начало...", "error": ""}

    def _run():
        try:
            import yt_dlp  # noqa

            def _hook(h):
                if h.get("status") == "downloading":
                    total   = h.get("total_bytes") or h.get("total_bytes_estimate") or 0
                    dl      = h.get("downloaded_bytes") or 0
                    pct     = int(dl * 100 / total) if total else 0
                    speed   = h.get("_speed_str","").strip()
                    eta     = h.get("_eta_str","").strip()
                    stage   = f"Загрузка: {pct}%"
                    if speed: stage += f"  {speed}"
                    if eta:   stage += f"  ETA {eta}"
                    with _yt_lock:
                        _yt_status[pid].update({"progress": pct, "stage": stage})
                elif h.get("status") == "finished":
                    with _yt_lock:
                        _yt_status[pid].update({"progress": 95, "stage": "Объединение потоков..."})

            out_tmpl = str(tmp_dir / f"{safe_title}.%(ext)s")
            browser    = d.get("browser", "")
            cookie_pid = d.get("cookie_pid", "")
            ydl_opts = {
                "format":     fmt_id,
                "outtmpl":    out_tmpl,
                "merge_output_format": "mp4" if ext in ("mp4","mkv","webm") else ext,
                "ffmpeg_location": str(_local_ff.parent) if _local_ff.exists() else None,
                "progress_hooks": [_hook],
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            if browser:
                ydl_opts["cookiesfrombrowser"] = (browser, None, None, None)
            elif cookie_pid:
                cf = UPLOAD_DIR / f"_cookies_{cookie_pid}.txt"
                if cf.exists():
                    ydl_opts["cookiefile"] = str(cf)
            player_client = d.get("player_client", "")
            if player_client:
                # «Получить форматы» уже выяснил, что клиент плеера по
                # умолчанию упирается в "The page needs to be reloaded" и
                # какой клиент реально работает — используем его сразу,
                # не тратя время на заведомо провальную первую попытку.
                ydl_opts["extractor_args"] = {"youtube": {"player_client": [player_client]}}
            # No explicit cookies — try browsers silently in background
            # (will be used as fallback inside _try_download if needed)
            _auto_browsers = ["firefox", "edge", "chrome"]
            # Ordered fallback chain: try increasingly permissive format selectors
            _FORMAT_CHAIN = [
                None,                        # use originally requested format
                "bestvideo+bestaudio/best",  # standard DASH merge
                "best",                      # single combined stream
                "bestvideo/best",            # video only fallback
                "bestaudio/best",            # audio only fallback
                "-1",                        # worst/any (absolute last resort)
            ]

            def _is_format_error(msg: str) -> bool:
                return ("not available" in msg or "Requested format" in msg
                        or "No video formats" in msg)

            def _is_auth_error(msg: str) -> bool:
                return ("Sign in" in msg or "not a bot" in msg.lower()
                        or "confirm your age" in msg.lower()
                        or "This video is only available" in msg
                        or "Private video" in msg)

            def _is_cookie_decrypt_error(msg: str) -> bool:
                # Chrome/Edge (Chromium) shipped "App-Bound Encryption" for
                # cookies which yt-dlp often can't decrypt from outside the
                # browser's own process — shows up as a DPAPI/keyring error,
                # not an auth error. Firefox doesn't use DPAPI, so it's the
                # most reliable automatic fallback.
                m = (msg or "").lower()
                return "decrypt" in m and ("dpapi" in m or "cookie" in m or "keyring" in m)

            def _try_download(opts: dict, dl_url: str) -> None:
                """Download with progressive fallback:
                1) format chain (permissive selectors)
                2) if auth error and no cookies set — auto-try browser cookies
                """
                last_exc = None

                def _run_chain(chain_opts: dict) -> bool:
                    """Try all formats. Returns True on success, False on format errors."""
                    nonlocal last_exc
                    for i, fmt in enumerate(_FORMAT_CHAIN):
                        attempt_opts = dict(chain_opts)
                        if fmt is not None:
                            attempt_opts["format"] = fmt
                        stage_msg = ("Загрузка..." if i == 0
                                     else f"Повтор {i}/{len(_FORMAT_CHAIN)-1}: {fmt}...")
                        with _yt_lock:
                            _yt_status[pid].update({"stage": stage_msg})
                        try:
                            with yt_dlp.YoutubeDL(attempt_opts) as ydl:
                                ydl.download([dl_url])
                            return True
                        except Exception as exc:
                            msg = str(exc)
                            if _is_format_error(msg):
                                last_exc = exc; continue
                            raise
                    return False

                # First attempt with provided cookies (or none)
                has_cookies = "cookiesfrombrowser" in opts or "cookiefile" in opts
                cookie_decrypt_failed = False
                try:
                    success = _run_chain(opts)
                    if success:
                        return
                except Exception as exc:
                    if has_cookies and _is_cookie_decrypt_error(str(exc)):
                        cookie_decrypt_failed = True
                        last_exc = exc
                    else:
                        raise

                # If no cookies were provided — or the explicitly chosen browser's
                # cookie database couldn't be decrypted — silently try other
                # installed browsers before giving up.
                if not has_cookies or cookie_decrypt_failed:
                    tried_browser = opts.get("cookiesfrombrowser", (None,))[0]
                    for br in _auto_browsers:
                        if br == tried_browser:
                            continue
                        with _yt_lock:
                            _yt_status[pid].update({"stage": f"Авто-cookies: {br}..."})
                        br_opts = dict(opts)
                        br_opts["cookiesfrombrowser"] = (br, None, None, None)
                        try:
                            if _run_chain(br_opts):
                                return
                        except Exception as exc:
                            # Auth/permission/decrypt error with this browser — try next
                            if (_is_auth_error(str(exc)) or _is_format_error(str(exc))
                                    or _is_cookie_decrypt_error(str(exc))):
                                last_exc = exc; continue
                            raise

                raise last_exc  # all fallbacks exhausted

            def _is_player_reload_error(msg: str) -> bool:
                m = (msg or "").lower()
                return "page needs to be reloaded" in m or "unplayable" in m

            try:
                _try_download(ydl_opts, url)
            except Exception as exc:
                if not _is_player_reload_error(str(exc)):
                    raise
                # См. комментарий в api_youtube_info: YouTube иногда
                # принудительно переводит клиент плеера по умолчанию на
                # SABR-стриминг, который текущая версия yt-dlp может не
                # поддерживать. Пробуем другие клиенты плеера по очереди
                # (с тем же набором cookies/браузера), прежде чем сдаться.
                last_client_exc = exc
                succeeded = False
                for client in ("web_embedded", "android", "ios", "tv"):
                    with _yt_lock:
                        _yt_status[pid].update({"stage": f"Пробуем плеер: {client}..."})
                    client_opts = dict(ydl_opts)
                    ea = dict(client_opts.get("extractor_args") or {})
                    ea["youtube"] = {**(ea.get("youtube") or {}), "player_client": [client]}
                    client_opts["extractor_args"] = ea
                    try:
                        _try_download(client_opts, url)
                        succeeded = True
                        break
                    except Exception as exc2:
                        last_client_exc = exc2
                        continue
                if not succeeded:
                    raise last_client_exc

            # Find downloaded file
            downloaded = list(tmp_dir.iterdir())
            if not downloaded:
                raise FileNotFoundError("yt-dlp не создал файл")
            src_file = downloaded[0]
            dl_ext = src_file.suffix.lower()
            is_video = dl_ext in {".mp4",".mkv",".webm",".mov",".avi"}

            with _yt_lock:
                _yt_status[pid].update({"progress": 97, "stage": "Извлечение аудио..."})

            # Create hub project
            proj_pid = str(uuid.uuid4())
            wav_path = UPLOAD_DIR / f"{proj_pid}.wav"

            if is_video:
                vid_dst = UPLOAD_DIR / f"{proj_pid}_video{dl_ext}"
                shutil.copy2(str(src_file), str(vid_dst))
                extract_audio(vid_dst, wav_path)
            else:
                extract_audio(src_file, wav_path)
            try:
                shutil.copy2(str(wav_path), str(UPLOAD_DIR / f"{proj_pid}_original.wav"))
            except Exception:
                pass

            proj = {
                "id": proj_pid, "name": safe_title,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "segments": [], "audio_file": str(wav_path),
                "has_video": is_video, "video_ext": dl_ext if is_video else "",
                "model_key": "", "language": "en",
                "audio_ready": True,
                "hub_original_filename": title + dl_ext,
            }
            save_proj(proj)

            # Cleanup tmp
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

            with _yt_lock:
                _yt_status[pid].update({
                    "status":   "done",
                    "progress": 100,
                    "stage":    "Готово",
                    "proj_pid": proj_pid,
                    "proj_name": safe_title,
                    "has_video": is_video,
                    "filename":  title + dl_ext,
                })
        except Exception as e:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raw = str(e)
            if "No module named yt_dlp" in raw:
                raw = "yt-dlp не установлен — запустите setup.bat."
            elif "not a bot" in raw.lower() or "Sign in to confirm" in raw:
                raw = ("YouTube требует подтверждения. "
                       "Обновите yt-dlp: в папке приложения откройте CMD и выполните: "
                       "python\\Scripts\\yt-dlp.exe -U")
            elif "not available" in raw.lower() or "Requested format" in raw:
                raw = ("Ни один из форматов не сработал. Попробуйте: "
                       "1) обновить yt-dlp (python\\Scripts\\yt-dlp.exe -U), "
                       "2) загрузить cookies.txt из браузера.")
            with _yt_lock:
                _yt_status[pid].update({"status": "error", "error": raw})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "pid": pid})


@app.route("/api/youtube/status/<pid>")
def api_youtube_status(pid):
    with _yt_lock:
        st = dict(_yt_status.get(pid, {"status": "unknown"}))
    return jsonify(st)


def _fmt_bytes(b: int) -> str:
    if not b: return ""
    for unit in ("Б","КБ","МБ","ГБ"):
        if b < 1024: return f"{b:.0f} {unit}"
        b /= 1024
    return f"{b:.1f} ГБ"

@app.route("/api/youtube/cookies_upload", methods=["POST"])
def api_youtube_cookies_upload():
    """Save an uploaded cookies.txt and return a cookie_pid to reference it."""
    if "file" not in request.files:
        return jsonify({"error": "Файл не выбран"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Пустое имя"}), 400
    cookie_pid = str(uuid.uuid4())[:8]
    dst = UPLOAD_DIR / f"_cookies_{cookie_pid}.txt"
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        f.save(str(dst))
        # Quick validation: check it looks like Netscape cookies format
        first = dst.read_text(encoding="utf-8", errors="replace")[:200]
        if "youtube" not in first.lower() and "HTTP Cookie" not in first and "# Netscape" not in first:
            dst.unlink(missing_ok=True)
            return jsonify({"error": "Файл не похож на cookies.txt. Используйте расширение 'Get cookies.txt LOCALLY'."}), 400
        return jsonify({"ok": True, "cookie_pid": cookie_pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hub/projects")
def api_hub_projects():
    """Return ALL projects for the hub — unified pool regardless of origin."""
    out = []
    for pf in PROJECTS_DIR.glob("*.json"):
        try:
            d = json.loads(pf.read_text("utf-8"))
            out.append({
                "id":            d["id"],
                "name":          d.get("name", "Без названия"),
                "has_video":     d.get("has_video", False),
                "created_at":    d.get("created_at", ""),
                "updated_at":    d.get("updated_at", ""),
                "filename":      d.get("hub_original_filename", d.get("name", "")),
                "segments_count": len(d.get("segments", [])),
                "hub_origin":    d.get("hub", False),   # True = uploaded via hub
            })
        except: pass
    out.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
    return jsonify(out)


@app.route("/api/hub/<pid>", methods=["DELETE"])
def api_hub_delete(pid):
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Project not found"}), 404
    # Remove project JSON
    try: (PROJECTS_DIR / f"{pid}.json").unlink(missing_ok=True)
    except: pass
    # Remove uploads
    for f in UPLOAD_DIR.glob(f"{pid}*"):
        try: f.unlink()
        except: pass
    # Remove separated outputs
    for sep_dir in SEP_DIR.glob(f"*/{pid}"):
        try: shutil.rmtree(str(sep_dir))
        except: pass
    return jsonify({"ok": True})


@app.route("/api/hub/upload", methods=["POST"])
def api_hub_upload():
    if "file" not in request.files:
        return jsonify({"error": "Файл не выбран"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Пустое имя файла"}), 400

    ext = Path(f.filename).suffix.lower()
    is_video = ext in {".mp4",".mov",".avi",".mkv",".webm"}
    is_audio = ext in {".wav",".mp3",".m4a",".ogg",".flac",".aac"}
    if not (is_audio or is_video):
        return jsonify({"error": f"Формат {ext} не поддерживается"}), 400

    pid = str(uuid.uuid4())
    original_name = Path(f.filename).stem

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # ВАЖНО: имя временного файла ДОЛЖНО отличаться от wav_path ниже.
        # `f"{pid}{ext}"` для .wav-файла даёт РОВНО ТОТ ЖЕ путь, что и
        # wav_path (`f"{pid}.wav"`), и extract_audio(raw_path, wav_path)
        # превращается в попытку ffmpeg записать файл поверх самого себя же
        # читаемого — "Output ... same as Input #0 - exiting". Суффикс
        # _raw всегда делает путь уникальным независимо от расширения.
        raw_path = UPLOAD_DIR / f"{pid}_raw{ext}"
        f.save(str(raw_path))

        wav_path = UPLOAD_DIR / f"{pid}.wav"
        extract_audio(raw_path, wav_path)
        try:
            shutil.copy2(str(wav_path), str(UPLOAD_DIR / f"{pid}_original.wav"))
        except Exception:
            pass

        if is_video:
            vid_dst = UPLOAD_DIR / f"{pid}_video{ext}"
            shutil.copy2(str(raw_path), str(vid_dst))
        # Временный файл больше не нужен — видео уже сохранено отдельно как vid_dst выше.
        try: raw_path.unlink(missing_ok=True)
        except: pass

        proj = {
            "id": pid, "name": original_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "segments": [], "audio_file": str(wav_path),
            "has_video": is_video, "video_ext": ext if is_video else "",
            "model_key": "", "language": "en",
            "audio_ready": True,
            "hub": True,
            "hub_original_filename": f.filename,
        }
        save_proj(proj)

        return jsonify({"ok": True, "pid": pid,
                        "filename": f.filename,
                        "original_name": original_name,
                        "has_video": is_video})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    return jsonify(_load_settings())

@app.route("/api/settings", methods=["POST"])
def api_settings_set():
    data = request.json or {}
    _save_settings(data)
    return jsonify({"ok": True, "settings": _load_settings()})

def _open_folder_dialog(start_dir: str = "") -> str | None:
    """Open native folder picker via pywebview; returns chosen path or None."""
    try:
        import webview
        wins = webview.windows
        if not wins:
            return None
        start = start_dir or str(_get_downloads_dir())
        result = wins[0].create_file_dialog(webview.FOLDER_DIALOG, directory=start)
        if not result:
            return None
        return str(result[0]) if isinstance(result, (list, tuple)) else str(result)
    except Exception:
        return None


@app.route("/api/choose_folder", methods=["POST"])
def api_choose_folder():
    """Open a native folder-picker dialog via pywebview and return chosen path."""
    d = request.json or {}
    start = d.get("start_dir", "")
    chosen = _open_folder_dialog(start)
    if chosen is None:
        return jsonify({"cancelled": True})
    return jsonify({"ok": True, "path": chosen})


@app.route("/api/hub/<pid>/save_original", methods=["POST"])
def api_hub_save_original(pid):
    """Save the original audio or video file of a hub project to Downloads."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404

    d = request.json or {}
    want_video = d.get("video", False)
    save_dir_raw = d.get("save_dir", "")
    if not save_dir_raw:
        return jsonify({"error": "Папка не выбрана"}), 400
    save_dir = Path(save_dir_raw)

    results = []

    if want_video and p.get("has_video"):
        video_ext = p.get("video_ext", ".mp4")
        video_src = UPLOAD_DIR / f"{pid}_video{video_ext}"
        if not video_src.exists():
            return jsonify({"error": f"Видеофайл не найден: {video_src.name}"}), 404
        safe_name = _sanitize_filename(p["name"]) or "video"
        dst = _save_copy(video_src, f"{safe_name}{video_ext}", folder=save_dir)
        results.append(str(dst))
    else:
        wav_src = UPLOAD_DIR / f"{pid}.wav"
        if not wav_src.exists():
            return jsonify({"error": f"Аудиофайл не найден: {wav_src.name}"}), 404
        safe_name = _sanitize_filename(p["name"]) or "audio"
        dst = _save_copy(wav_src, f"{safe_name}.wav", folder=save_dir)
        results.append(str(dst))

    return jsonify({"ok": True, "paths": results})


@app.route("/api/hub/<pid>/save_sep", methods=["POST"])
def api_hub_save_sep(pid):
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Project not found"}), 404

    d = request.json or {}
    model      = d.get("model", "htdemucs")
    stem       = d.get("stem", "no_vocals")
    save_wav   = d.get("save_wav", True)
    save_video = d.get("save_video", False)
    save_dir_raw = d.get("save_dir", "")
    if not save_dir_raw:
        return jsonify({"error": "Папка не выбрана"}), 400
    base_dir = Path(save_dir_raw)

    if stem not in ("vocals", "no_vocals"):
        return jsonify({"error": "Неверный stem"}), 400

    folder_name = p.get("name", pid)
    label = "голос" if stem == "vocals" else "без_голоса"
    results = []

    if save_wav:
        f_wav = _sep_file(pid, model, stem)
        if not f_wav.exists():
            return jsonify({"error": f"{stem}.wav не найден — запустите разделение"}), 404
        dst = _save_to_hub_folder(folder_name, f"{label}.wav", src_path=f_wav, base_dir=base_dir)
        results.append({"name": f"{label}.wav", "path": str(dst)})

    if save_video:
        if not p.get("has_video"):
            return jsonify({"error": "Исходный файл — аудио, нет видео"}), 400
        audio_wav = _sep_file(pid, model, stem)
        if not audio_wav.exists():
            return jsonify({"error": "Разделённое аудио не найдено"}), 404
        video_ext = p.get("video_ext", ".mp4")
        video_src = UPLOAD_DIR / f"{pid}_video{video_ext}"
        if not video_src.exists():
            return jsonify({"error": "Исходное видео не найдено"}), 404

        tmp_out = EXPORT_DIR / f"{pid}_{stem}_hub_vid.mp4"
        try:
            ff_bin = str(_local_ff) if _local_ff.exists() else "ffmpeg"
            cmd = [ff_bin,
                   "-i", str(video_src), "-i", str(audio_wav),
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                   "-shortest", "-y", str(tmp_out)]
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                err = result.stderr.decode(errors="replace")[-500:]
                return jsonify({"error": f"FFmpeg: {err}"}), 500
            dst = _save_to_hub_folder(folder_name, f"{label}.mp4", src_path=tmp_out, base_dir=base_dir)
            results.append({"name": f"{label}.mp4", "path": str(dst)})
        except subprocess.TimeoutExpired:
            return jsonify({"error": "FFmpeg: превышено время ожидания"}), 500
        finally:
            try: tmp_out.unlink(missing_ok=True)
            except: pass

    root = base_dir if (base_dir and base_dir.exists()) else _get_save_dir()
    folder_path = str(root / _sanitize_filename(folder_name))
    return jsonify({"ok": True, "results": results, "folder": folder_path})


@app.route("/api/hub/<pid>/save_tr", methods=["POST"])
def api_hub_save_tr(pid):
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Project not found"}), 404
    if not p.get("segments"):
        return jsonify({"error": "Нет сегментов — сначала запустите транскрипцию"}), 400

    d = request.json or {}
    formats = d.get("formats", ["srt", "csv"])
    folder_name = p.get("name", pid)
    save_dir_raw = d.get("save_dir", "")
    if not save_dir_raw:
        return jsonify({"error": "Папка не выбрана"}), 400
    base_dir = Path(save_dir_raw)
    results = []

    for fmt in formats:
        try:
            if fmt == "srt":
                dst = _save_to_hub_folder(folder_name, "транскрипция.srt",
                                          data=_gen_srt_bytes(p), base_dir=base_dir)
                results.append({"name": "транскрипция.srt", "path": str(dst)})
            elif fmt == "csv":
                dst = _save_to_hub_folder(folder_name, "транскрипция.csv",
                                          data=_gen_csv_bytes(p), base_dir=base_dir)
                results.append({"name": "транскрипция.csv", "path": str(dst)})
            elif fmt == "json":
                dst = _save_to_hub_folder(folder_name, "транскрипция.json",
                                          data=_gen_json_bytes(p), base_dir=base_dir)
                results.append({"name": "транскрипция.json", "path": str(dst)})
            elif fmt == "docx":
                tmp = EXPORT_DIR / f"{pid}_hub_tr.docx"
                _gen_docx_to_path(p, tmp)
                dst = _save_to_hub_folder(folder_name, "транскрипция.docx",
                                          src_path=tmp, base_dir=base_dir)
                tmp.unlink(missing_ok=True)
                results.append({"name": "транскрипция.docx", "path": str(dst)})
        except Exception as e:
            results.append({"name": fmt, "error": str(e)})

    root = base_dir if (base_dir and base_dir.exists()) else _get_save_dir()
    folder_path = str(root / _sanitize_filename(folder_name))
    return jsonify({"ok": True, "results": results, "folder": folder_path})

@app.route("/api/projects/import", methods=["POST"])
def import_project():
    if "file" not in request.files: return jsonify({"error":"Файл не передан"}), 400
    f = request.files["file"]
    if not f.filename: return jsonify({"error":"Файл не выбран"}), 400
    if Path(f.filename).suffix.lower() not in {".tsproject",".zip"}:
        return jsonify({"error":"Неверный формат. Ожидается .tsproject или .zip"}), 400
    import zipfile, tempfile
    tmp = Path(tempfile.mktemp(suffix=".tsproject", dir=str(EXPORT_DIR)))
    try:
        f.save(str(tmp))
        try: zipfile.ZipFile(str(tmp)).close()
        except zipfile.BadZipFile: return jsonify({"error":"Повреждённый архив"}), 400
        with zipfile.ZipFile(str(tmp)) as zf:
            names = zf.namelist()
            if "project.json" not in names: return jsonify({"error":"Архив не содержит project.json"}), 400
            proj = json.loads(zf.read("project.json").decode("utf-8"))
            new_id = str(uuid.uuid4()); original_name = proj.get("name","Импортированный проект")
            proj["id"] = new_id; proj["name"] = original_name
            proj["imported_at"] = datetime.now().isoformat()
            proj["updated_at"]  = datetime.now().isoformat()
            proj["audio_ready"] = False

            def _extract(arc_name: str, dst: Path):
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(arc_name) as src, open(str(dst), "wb") as out:
                    shutil.copyfileobj(src, out, length=8*1024*1024)

            # ── Audio: new layout (audio/original.wav + audio/transcription_source.wav)
            #    with fallback to the old flat layout ("audio.wav") for older exports
            orig_arc = "audio/original.wav"
            src_arc  = "audio/transcription_source.wav"
            wav_dst  = UPLOAD_DIR / f"{new_id}.wav"
            orig_dst = UPLOAD_DIR / f"{new_id}_original.wav"
            if orig_arc in names:
                _extract(orig_arc, orig_dst)
                proj["audio_ready"] = True
            if src_arc in names:
                _extract(src_arc, wav_dst)
                proj["audio_ready"] = True
            elif orig_arc in names:
                # No separate transcription-source copy saved — original IS the source
                shutil.copy2(str(orig_dst), str(wav_dst))
                proj["audio_ready"] = True
            elif "audio.wav" in names:  # legacy flat layout
                _extract("audio.wav", wav_dst)
                shutil.copy2(str(wav_dst), str(orig_dst))
                proj["audio_ready"] = True

            # ── Video: new layout (video/original{ext}) with legacy fallback
            if proj.get("has_video"):
                ext = proj.get("video_ext", ".mp4")
                vid_dst = UPLOAD_DIR / f"{new_id}_video{ext}"
                new_vid_arc = f"video/original{ext}"
                legacy_vid_arc = f"video{ext}"
                if new_vid_arc in names:
                    _extract(new_vid_arc, vid_dst)
                elif legacy_vid_arc in names:
                    _extract(legacy_vid_arc, vid_dst)
                else:
                    proj["has_video"] = False

            # ── Separated Demucs stems: separated/<model>/<stem>.wav
            restored_stems = 0
            for arc_name in names:
                if not arc_name.startswith("separated/") or not arc_name.endswith(".wav"):
                    continue
                parts = arc_name.split("/")
                if len(parts) != 3:
                    continue
                _, model_name, stem_file = parts
                dst = SEP_DIR / model_name / new_id / stem_file
                _extract(arc_name, dst)
                restored_stems += 1

            # ── Extra manually-uploaded tracks: tracks/<track_id>.wav
            # Metadata (name) is already restored as part of project.json's
            # "extra_tracks" list — just need to also restore each file,
            # dropping entries whose audio didn't make it into the archive.
            valid_track_ids = set()
            for arc_name in names:
                if not arc_name.startswith("tracks/") or not arc_name.endswith(".wav"):
                    continue
                track_id = arc_name[len("tracks/"):-len(".wav")]
                _extract(arc_name, _track_file(new_id, track_id))
                valid_track_ids.add(track_id)
            if proj.get("extra_tracks"):
                proj["extra_tracks"] = [t for t in proj["extra_tracks"] if t["id"] in valid_track_ids]

            save_proj(proj)
            return jsonify({"ok":True,"project":{"id":new_id,"name":original_name,
                "segments_count":len(proj.get("segments",[])),
                "created_at":proj.get("created_at",""),"updated_at":proj["updated_at"],
                "has_video":proj.get("has_video",False),"audio_ready":proj["audio_ready"],
                "restored_stems":restored_stems}})
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500
    finally:
        tmp.unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Разделение аудио (Demucs)
# ─────────────────────────────────────────────────────────────────────────────

def _sep_out_dir(pid: str, model: str) -> Path:
    return SEP_DIR / model / pid

def _find_best_vocals(pid: str) -> Path | None:
    """Find the best available 'vocals only' separated track for a project,
    preferring higher-quality models first."""
    preferred = ["htdemucs_ft", "htdemucs", "mdx23c"]
    for m in preferred:
        f = SEP_DIR / m / pid / "vocals.wav"
        if f.exists():
            return f
    if SEP_DIR.exists():
        for model_dir in SEP_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            f = model_dir / pid / "vocals.wav"
            if f.exists():
                return f
    return None


def _sep_file(pid: str, model: str, stem: str) -> Path:
    return _sep_out_dir(pid, model) / f"{stem}.wav"

def _is_sep_done(pid: str, model: str, operation: str) -> bool:
    stems = _op_stems(operation)
    return all(_sep_file(pid, model, s).exists() for s in stems)

def _op_stems(operation: str) -> list:
    if operation == "remove_vocals":  return ["no_vocals"]
    if operation == "isolate_vocals": return ["vocals"]
    return ["vocals", "no_vocals"]  # both

def _do_separate(pid: str, model: str, operation: str):
    with _sep_lock:
        _sep_status[pid] = {"status":"running","progress":0,"error":None,
                            "model":model,"operation":operation,"stage":"Подготовка..."}
    _sep_t0 = time.time()
    try:
        p = load_proj(pid)

        # Оригинальный файл (лучше качество чем 16kHz WAV)
        source = None
        if p and p.get("has_video"):
            ext = p.get("video_ext", ".mp4")
            c = UPLOAD_DIR / f"{pid}_video{ext}"
            if c.exists():
                source = c
        if source is None:
            source = UPLOAD_DIR / f"{pid}.wav"
        if not source.exists():
            raise FileNotFoundError("Исходный файл не найден: " + str(source))

        # Временная папка для вывода Demucs
        tmp_out = SEP_DIR / "_tmp" / pid
        if tmp_out.exists():
            shutil.rmtree(str(tmp_out))
        tmp_out.mkdir(parents=True, exist_ok=True)

        # Используем demucs_run.py — патчит torchaudio.save перед запуском
        demucs_runner = Path(__file__).parent / "demucs_run.py"
        # Явно передаём устройство — без этого Demucs может не увидеть GPU
        try:
            import torch as _torch
            _device = "cuda" if _torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"
        print(f"[SEP] device={_device}", flush=True)

        cmd = [
            sys.executable, str(demucs_runner),
            "--two-stems", "vocals",
            "-n", model,
            "--device", _device,
            "-o", str(tmp_out),
            str(source),
        ]

        with _sep_lock:
            _sep_status[pid]["stage"] = "Запуск Demucs..."

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )

        pct_re = re.compile(r"([0-9]+)%")
        last_pct = -1
        log_lines = []

        # htdemucs_ft — это "мешок" из 4 отдельных fine-tuned моделей
        # (Demucs прогоняет через них трек по очереди для лучшего качества).
        # Прогресс-бар в его логах поэтому доходит до 100% и НАЧИНАЕТСЯ
        # ЗАНОВО с 0% — четыре раза за один запуск. Раньше наш код при первом
        # же 100% замирал на стадии "Запись файлов" и больше не обновлялся,
        # хотя реально могло остаться ещё 3 таких же прохода (то есть
        # реальное время — примерно в 4 раза больше уже прошедшего). Здесь
        # мы это учитываем и показываем честный общий процент + номер прохода.
        expected_passes = 4 if model == "htdemucs_ft" else 1
        pass_num = 1

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log_lines.append(line)
            if len(log_lines) > 300:
                log_lines.pop(0)
            m = pct_re.search(line)
            if m:
                pct = int(m.group(1))
                # Резкое падение процента говорит о начале нового прохода
                # ансамбля, а не о реальном откате назад
                if pct < last_pct - 50 and pass_num < expected_passes:
                    pass_num += 1
                last_pct = pct
                overall = min(99, int(((pass_num - 1) * 100 + pct) / expected_passes))
                with _sep_lock:
                    _sep_status[pid]["progress"] = overall
                    if pct >= 100 and pass_num >= expected_passes:
                        _sep_status[pid]["stage"] = "Запись файлов..."
                    elif expected_passes > 1:
                        _sep_status[pid]["stage"] = f"Обработка — проход {pass_num} из {expected_passes} (ансамбль моделей)"
                    else:
                        _sep_status[pid]["stage"] = "Обработка"

        with _sep_lock:
            _sep_status[pid]["stage"] = "Финализация..."

        proc.wait()

        if proc.returncode != 0:
            tail = "\n".join(log_lines[-25:])
            raise RuntimeError(
                "Demucs завершился с кодом " + str(proc.returncode) +
                ". Последний вывод:\n" + tail[-1500:]
            )

        # Определяем какие треки нужны по выбранной операции
        if operation == "remove_vocals":
            wanted = ["no_vocals"]
        elif operation == "isolate_vocals":
            wanted = ["vocals"]
        else:
            wanted = ["vocals", "no_vocals"]   # both

        # Ищем файлы рекурсивно — не важно куда Demucs их положил
        final_dir = SEP_DIR / model / pid
        final_dir.mkdir(parents=True, exist_ok=True)
        found = {}
        for stem in wanted:
            matches = list(tmp_out.rglob(stem + ".wav"))
            if matches:
                dst = final_dir / (stem + ".wav")
                # ВАЖНО: Demucs пишет WAV в своём родном формате — 44.1кГц
                # СТЕРЕО, а не 16кГц моно, как все остальные wav-файлы в
                # приложении (см. extract_audio()). Если просто скопировать
                # файл как есть, load_wav() (который не проверяет реальные
                # framerate/channels и всегда предполагает 16кГц моно) читает
                # его неправильно: два интерлived-канала превращаются в
                # мешанину сэмплов одного "канала" — модель распознавания
                # получает искажённый звук вместо речи, а тайминги "растяги-
                # ваются" в ~44100×2/16000 ≈ 5.5 раза относительно реальной
                # длины. Ресемплируем сразу здесь, чтобы ЛЮБОЙ потребитель
                # этого файла (распознавание, плеер, экспорт) получал
                # корректный 16кГц-моно WAV, как и everywhere else.
                try:
                    (ffmpeg.input(str(matches[0]))
                           .output(str(dst), acodec="pcm_s16le", ac=1, ar=16000, loglevel="error")
                           .overwrite_output().run())
                except Exception as _rs_err:
                    print(f"[SEP] Ресемплинг {stem}.wav не удался ({_rs_err}), копирую как есть", flush=True)
                    shutil.copy2(str(matches[0]), str(dst))
                found[stem] = True

        # Убираем временную папку
        shutil.rmtree(str(tmp_out), ignore_errors=True)

        if not found:
            tail = "\n".join(log_lines[-20:])
            raise RuntimeError(
                "Demucs не создал выходные файлы.\n"
                "Содержимое вывода:\n" + tail[-1200:]
            )

        stems_done = list(found.keys())
        try:
            _sep_elapsed = time.time() - _sep_t0
            _sep_audio_dur = len(load_wav(UPLOAD_DIR / f"{pid}.wav")) / 16000.0
            _record_speed("separate", model, _device, _sep_audio_dur, _sep_elapsed)
        except Exception as _rec_err:
            print(f"[SPEED] separate: не удалось записать статистику ({_rec_err})")
        with _sep_lock:
            _sep_status[pid] = {
                "status":"done","progress":100,"error":None,
                "model":model,"operation":operation,"stage":"Готово!",
                "stems":stems_done,
            }
        print("[SEP] Done:", pid, stems_done)

    except Exception as e:
        traceback.print_exc()
        # Очищаем временную папку при ошибке
        tmp_out = SEP_DIR / "_tmp" / pid
        shutil.rmtree(str(tmp_out), ignore_errors=True)
        # Укорачиваем сообщение чтобы не было "длинной ошибки"
        err_short = str(e)[:600]
        with _sep_lock:
            _sep_status[pid] = {
                "status":"error","progress":0,"error":err_short,
                "model":model,"operation":operation,"stage":"Ошибка",
            }


@app.route("/api/system/gpu_status")
def api_gpu_status():
    """Честный ответ на вопрос "использует ли приложение видеокарту?" —
    без необходимости лезть в консоль или разбираться, какой движок
    показывает Диспетчер задач Windows (Compute/CUDA там по умолчанию не
    отображается, в отличие от 3D/Copy/Video Encode/Decode)."""
    import torch
    info = {
        "torch_version": torch.__version__,
        "cuda_available": False,
        "device_name": None,
        "cuda_build": getattr(torch.version, "cuda", None),  # None => CPU-only сборка torch
    }
    try:
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["error"] = str(e)
    return jsonify(info)


@app.route("/api/demucs/models")
def api_demucs_models():
    return jsonify(DEMUCS_MODELS)


@app.route("/api/projects/<pid>/separate", methods=["POST"])
def api_separate(pid):
    p = load_proj(pid)
    if not p: return jsonify({"error": "Проект не найден"}), 404
    if not (UPLOAD_DIR / f"{pid}.wav").exists():
        return jsonify({"error": "Сначала загрузите аудиофайл в проект"}), 400
    d = request.json or {}
    model     = d.get("model", "htdemucs")
    operation = d.get("operation", "remove_vocals")
    if model not in DEMUCS_MODELS:
        return jsonify({"error": "Неизвестная модель"}), 400

    with _sep_lock:
        st = _sep_status.get(pid, {})
    if st.get("status") == "running":
        return jsonify({"error": "Обработка уже выполняется"}), 409

    threading.Thread(target=_do_separate, args=(pid, model, operation), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/projects/<pid>/separate/status")
def api_separate_status(pid):
    with _sep_lock:
        st = dict(_sep_status.get(pid, {"status": None}))
    # Добавляем инфо о уже готовых файлах для всех моделей
    available = {}
    for m in DEMUCS_MODELS:
        stems = []
        for s in ["vocals", "no_vocals"]:
            if _sep_file(pid, m, s).exists():
                stems.append(s)
        if stems:
            available[m] = stems
    st["available"] = available
    return jsonify(st)


@app.route("/api/projects/<pid>/separated/<model>/<stem>")
def api_separated_file(pid, model, stem):
    if stem not in ("vocals", "no_vocals"):
        return jsonify({"error": "Неверный stem"}), 400
    f = _sep_file(pid, model, stem)
    if not f.exists():
        return jsonify({"error": "Файл не найден — запустите обработку"}), 404
    return send_file(str(f), mimetype="audio/wav",
                     as_attachment=False,
                     download_name=f"{stem}.wav")


@app.route("/api/projects/<pid>/reset_transcribe_source", methods=["POST"])
def api_reset_transcribe_source(pid):
    """Switch the project back to automatic transcription-source selection
    (vocals-only if available, otherwise original)."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404
    p["transcribe_source_manual"] = False
    save_proj(p)
    return jsonify({"ok": True})


@app.route("/api/projects/<pid>/prefer_vocals_only", methods=["POST"])
def api_set_prefer_vocals_only(pid):
    """Per-project preference: always prefer the 'только голос' (vocals-only)
    track for transcription when available, even for models (e.g. wav2vec2)
    that don't get this by default. Stored inside this project's own data —
    never affects any other project."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404
    d = request.json or {}
    p["prefer_vocals_only"] = bool(d.get("value", False))
    save_proj(p)
    return jsonify({"ok": True, "prefer_vocals_only": p["prefer_vocals_only"]})


@app.route("/api/projects/<pid>/separated/<model>/<stem>/use_as_audio", methods=["POST"])
def api_separated_use_as_audio(pid, model, stem):
    """Copy a separated stem WAV as the project audio source for transcription."""
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404
    sep_path = SEP_DIR / model / pid / f"{stem}.wav"
    if not sep_path.exists():
        return jsonify({"error": f"Трек не найден: {sep_path.name}"}), 404
    dst = UPLOAD_DIR / f"{pid}.wav"
    try:
        shutil.copy2(str(sep_path), str(dst))
    except Exception as e:
        return jsonify({"error": f"Ошибка копирования: {e}"}), 500
    p["audio_ready"] = True
    p["sep_source"] = {"model": model, "stem": stem}
    p["transcribe_source_manual"] = True  # user explicitly chose — don't auto-override
    save_proj(p)
    return jsonify({"ok": True})


@app.route("/api/projects/<pid>/export_zip_save", methods=["POST"])
def api_project_export_zip(pid):
    """Export everything about a project as a single ZIP and save it to a
    folder the user picked (requires save_dir in the request body, like
    every other save action in this app):
    project data (segments/translation), original + separated audio
    tracks, and ready-made transcript exports (SRT/CSV/DOCX)."""
    import zipfile, io, csv as _csv
    p = load_proj(pid)
    if not p:
        return jsonify({"error": "Проект не найден"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Full project data (segments, translation, settings, etc.)
        zf.writestr("project.json", json.dumps(p, ensure_ascii=False, indent=2))

        # True, untouched original audio
        orig = UPLOAD_DIR / f"{pid}_original.wav"
        if orig.exists():
            zf.write(str(orig), "audio/original.wav")

        # Current working track (whatever was last used for transcription —
        # may differ from original.wav if a separated stem was auto/manually selected)
        wav = UPLOAD_DIR / f"{pid}.wav"
        if wav.exists():
            zf.write(str(wav), "audio/transcription_source.wav")

        # Video file if present
        video_ext = p.get("video_ext", "")
        for ext in ([video_ext] if video_ext else []) + [".mp4", ".mkv", ".webm", ".mov", ".avi"]:
            vid = UPLOAD_DIR / f"{pid}_video{ext}"
            if vid.exists():
                zf.write(str(vid), f"video/original{ext}")
                break

        # All separated Demucs stems for this project, across every model
        # that has been run (correct path layout: SEP_DIR/<model>/<pid>/<stem>.wav)
        if SEP_DIR.exists():
            for model_dir in SEP_DIR.iterdir():
                if not model_dir.is_dir():
                    continue
                proj_dir = model_dir / pid
                if not proj_dir.is_dir():
                    continue
                for f in proj_dir.glob("*.wav"):
                    zf.writestr(f"separated/{model_dir.name}/{f.name}", f.read_bytes())

        # Extra manually-uploaded tracks (e.g. alternate dub/narration).
        # Metadata (id/name) already travels inside project.json's
        # "extra_tracks" list — here we just carry the audio bytes,
        # keyed by track id so import can match them back up.
        for t in p.get("extra_tracks", []):
            tf = _track_file(pid, t["id"])
            if tf.exists():
                zf.writestr(f"tracks/{t['id']}.wav", tf.read_bytes())

        # Ready-made transcript exports
        segs = p.get("segments", [])
        if segs:
            try:
                zf.writestr("transcript/transcript.srt", _gen_srt_bytes(p))
            except Exception:
                pass
            try:
                zf.writestr("transcript/transcript.csv", _gen_csv_bytes(p))
            except Exception:
                pass
            try:
                tmp_docx = EXPORT_DIR / f"{pid}_zipexport.docx"
                _gen_docx_to_path(p, tmp_docx)
                zf.writestr("transcript/transcript.docx", tmp_docx.read_bytes())
                tmp_docx.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                zf.writestr("transcript/transcript.json", _gen_json_bytes(p))
            except Exception:
                pass

    save_dir_raw = (request.json or {}).get("save_dir", "")
    if not save_dir_raw:
        return jsonify({"error": "Папка не выбрана"}), 400
    save_dir = Path(save_dir_raw)

    buf.seek(0)
    safe_name = _sanitize_filename(p.get("name", pid)) or pid
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        dst = _unique_dst(save_dir, f"{safe_name}.zip")
        dst.write_bytes(buf.getvalue())
    except Exception as e:
        return jsonify({"error": f"Ошибка сохранения: {e}"}), 500
    return jsonify({"ok": True, "path": str(dst)})


@app.route("/api/projects/<pid>/separated/<model>/<stem>/download")
def api_separated_download(pid, model, stem):
    p = load_proj(pid)
    proj_name = p["name"] if p else pid
    if stem not in ("vocals", "no_vocals"):
        return jsonify({"error": "Неверный stem"}), 400
    f = _sep_file(pid, model, stem)
    if not f.exists():
        return jsonify({"error": "Файл не найден"}), 404
    label = "голос" if stem == "vocals" else "без_голоса"
    name  = f"{proj_name}_{label}.wav"
    return send_file(str(f), mimetype="audio/wav",
                     as_attachment=True, download_name=name)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # hf_transfer (быстрая загрузка через отдельный Rust-модуль) отключена
    # намеренно: на части сетей (прокси, некоторые провайдеры, антивирусы
    # с SSL-инспекцией) она молча зависает на первых мегабайтах без вменяемой
    # ошибки, в отличие от штатного загрузчика huggingface_hub, который
    # умеет докачивать и корректно сообщает о сетевых сбоях. Пакет даже не
    # входит в requirements.txt — эта строка лишь перестраховка на случай,
    # если кто-то поставит hf_transfer вручную.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # Явный таймаут на сетевые запросы huggingface_hub — без него зависшее
    # TCP-соединение могло висеть неограниченно долго вместо того, чтобы
    # упасть с ошибкой и дать сработать повторным попыткам (_dl_with_retry).
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    PORT = 5000
    print("=" * 50)
    print(f"  TranscribeStudio  ->  http://127.0.0.1:{PORT}")
    print("=" * 50)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
