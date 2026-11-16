"""
Запускает Demucs с патчем torchaudio.save —
обходит ошибку 'No module named torchcodec' в новых версиях torchaudio.
Используется как: python demucs_run.py [аргументы demucs...]
"""
import sys

# ── Патч torchaudio до загрузки Demucs ────────────────────────────────
try:
    import torchaudio as _ta
    import soundfile as _sf

    _orig_save = _ta.save

    def _safe_save(uri, src, sample_rate, *args, **kwargs):
        try:
            _orig_save(uri, src, sample_rate, *args, **kwargs)
        except Exception as e:
            if "torchcodec" in str(e) or "No module" in str(e):
                import numpy as _np
                data = src.numpy()
                if data.ndim == 2:
                    data = data.T   # (C, T) -> (T, C)
                _sf.write(str(uri), data, sample_rate)
            else:
                raise

    _ta.save = _safe_save

    # Патчим _torchcodec модуль напрямую если он существует
    try:
        import torchaudio._torchcodec as _tc
        def _stub(*a, **kw):
            raise ImportError("torchcodec stub — fallback to soundfile")
        _tc.save_with_torchcodec = _stub
    except (ImportError, AttributeError):
        pass

    print("[demucs_run] torchaudio.save patched (soundfile fallback)", flush=True)

except Exception as patch_err:
    print(f"[demucs_run] patch warning: {patch_err}", flush=True)

# ── Запуск Demucs ──────────────────────────────────────────────────────
from demucs.__main__ import main
sys.exit(main())
