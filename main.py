"""
TranscribeStudio — запуск одного окна pywebview
"""

import os, sys, socket, threading, time, traceback, logging, subprocess
from pathlib import Path

_APP_DIR = Path(__file__).parent.resolve()
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

_LOG_FILE = Path(__file__).parent / "start_error.log"
try: _LOG_FILE.unlink(missing_ok=True)
except: pass

def _log(msg):
    # Write to log file ONLY for errors — INFO lines must not trigger START.bat error check
    if "[ERROR]" in msg or "[FATAL]" in msg or "[WARN]" in msg:
        try:
            with _LOG_FILE.open("a", encoding="utf-8") as f: f.write(msg + "\n")
        except: pass
    print(msg, file=sys.stderr)

def _show_error(msg):
    _log(f"[ERROR] {msg}")
    try:
        safe = msg.replace('"', "'").replace("\n", " ").replace("\r", "")
        ps = ('Add-Type -AssemblyName System.Windows.Forms; '
              f'[System.Windows.Forms.MessageBox]::Show("{safe}",'
              f'"TranscribeStudio - Error","OK","Error") | Out-Null')
        subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps], timeout=15)
    except: pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# hf_transfer отключена намеренно — на части сетей молча зависает на первых
# мегабайтах загрузки без вменяемой ошибки (см. комментарий в app.py).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

_log("[INFO] Importing app.py...")
try:
    from app import app as flask_app
    _log("[INFO] app.py imported.")
except Exception as e:
    _show_error(f"Failed to load app.py:\n{e}\n\nRun setup.bat to reinstall dependencies.")
    sys.exit(1)

PORT = 5000

class Api:
    """Exposed to the frontend as window.pywebview.api.*
    pywebview marshals these calls onto the correct (main) GUI thread
    internally, unlike calling webview.windows[0].create_file_dialog()
    directly from a Flask request thread — which is unreliable and can
    silently fail to return the chosen path."""
    def choose_folder(self, start_dir=""):
        try:
            import webview
            wins = webview.windows
            if not wins:
                return None
            start = start_dir or str(Path.home() / "Downloads")
            result = wins[0].create_file_dialog(webview.FOLDER_DIALOG, directory=start)
            if not result:
                return None
            return str(result[0]) if isinstance(result, (list, tuple)) else str(result)
        except Exception as e:
            _log(f"[WARN] choose_folder failed: {e}")
            return None

def _port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0

def _wait_for_flask(port, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3): return True
        except OSError: time.sleep(0.15)
    return False

def _run_flask():
    try:
        flask_app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        _log(f"[ERROR] Flask crashed: {e}\n{traceback.format_exc()}")

def main():
    _log("[INFO] Starting TranscribeStudio...")
    try:
        import webview
    except ImportError as e:
        _show_error(f"pywebview not installed: {e}\n\nRun setup.bat.")
        sys.exit(1)

    if not _port_free(PORT):
        # Не пишем в start_error.log — иначе START.bat зависнет с pause.
        # Просто показываем попап и выходим тихо.
        try:
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "[System.Windows.Forms.MessageBox]::Show("
                "'TranscribeStudio уже запущен.\nЗакройте существующее окно и попробуйте снова.',"
                "'TranscribeStudio','OK','Warning') | Out-Null"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                timeout=10,
            )
        except Exception:
            pass
        sys.exit(1)

    _log("[INFO] Starting Flask...")
    threading.Thread(target=_run_flask, daemon=True).start()
    if not _wait_for_flask(PORT, timeout=20.0):
        _show_error("Flask did not start in 20 seconds.\nSee start_error.log for details.")
        sys.exit(1)
    _log("[INFO] Flask ready.")

    _log("[INFO] Creating window...")
    try:
        window = webview.create_window(
            title="TranscribeStudio",
            url=f"http://127.0.0.1:{PORT}/",
            width=1320, height=840, min_size=(900, 600),
            js_api=Api(),
        )
    except Exception as e:
        _show_error(f"Could not create window: {e}")
        sys.exit(1)

    _log("[INFO] webview.start()...")
    # allow_downloads=True required for WebView2 (Windows) to not silently cancel downloads
    try:
        webview.start(debug=False, allow_downloads=True)
    except TypeError:
        try: webview.start(debug=False)
        except TypeError:
            try: webview.start()
            except Exception as e2: _show_error(f"pywebview failed: {e2}"); os._exit(1)
        except Exception as e2: _show_error(f"pywebview failed: {e2}"); os._exit(1)
    except Exception as e:
        _show_error(f"pywebview error: {e}\n\n{traceback.format_exc()}"); os._exit(1)

    _log("[INFO] Window closed.")
    try: _LOG_FILE.unlink(missing_ok=True)  # no errors — clean up
    except: pass
    os._exit(0)

if __name__ == "__main__":
    try: main()
    except Exception:
        err = traceback.format_exc()
        _log(f"[FATAL] {err}")
        _show_error("Critical startup error. See start_error.log.")
        os._exit(1)
