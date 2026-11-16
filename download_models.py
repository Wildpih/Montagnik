"""
TranscribeStudio — менеджер моделей
Запускается через DOWNLOAD_MODELS.bat

Исправления v4:
  - Whisper Large-v3 скачивается ОДИН РАЗ для всех языков.
    Повторные «языковые» варианты создаются через hard-link
    (ссылки на одни и те же блоки на диске) — без копирования файлов.
  - Удаляется скрытая папка .cache, которую snapshot_download
    создаёт внутри папки модели и дублирует файлы.
  - Удаляется pytorch_model.bin, если рядом уже есть model.safetensors
    (safetensors быстрее загружается и занимает столько же места).
"""

import os, sys, shutil, time, importlib.util
from pathlib import Path

# Embedded Python не добавляет папку скрипта в sys.path автоматически —
# делаем это явно (та же проблема что была в main.py)
_BASE = Path(__file__).parent.resolve()
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# hf_transfer отключена намеренно: на части сетей (прокси, некоторые
# провайдеры, антивирусы с SSL-инспекцией) она молча зависает на первых
# мегабайтах загрузки без вменяемой ошибки. Пакет даже не входит в
# requirements.txt — используется штатный загрузчик huggingface_hub,
# который умеет докачивать и корректно сообщает о сетевых сбоях.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]   = "30"

# ─── Цвета для терминала ────────────────────────────────────────────────────
def bold(s):   return f"\033[1m{s}\033[0m"
def green(s):  return f"\033[92m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def cyan(s):   return f"\033[96m{s}\033[0m"
def gray(s):   return f"\033[90m{s}\033[0m"

# ─── Каталог моделей ────────────────────────────────────────────────────────
# Обрати внимание: whisper-large-v3-ru, -en, -de, -fr, -es, -uk и
# whisper-large-v3 (multi) — это ОДИН И ТОТ ЖЕ файл весов (~3.1 ГБ).
# Язык задаётся при запуске транскрибации, не в весах модели.
# Повторное скачивание заменено hard-link: нет лишних гигабайт.

CATALOG = {
    # ══ РУССКИЙ ══════════════════════════════════════════════════════════════
    "1": {
        "key":   "whisper-large-v3-ru",
        "hf_id": "openai/whisper-large-v3",
        "label": "Whisper Large-v3 RU  ★ WER ~5%  (РЕКОМЕНДУЕТСЯ)",
        "lang":  "Русский", "size": "~3.1 ГБ", "arch": "whisper_hf",
    },
    "2": {
        "key":   "w2v-ru-golos",
        "hf_id": "bond005/wav2vec2-large-ru-golos",
        "label": "wav2vec2 RU Large · Golos — WER ~6%  (быстрее, меньше памяти)",
        "lang":  "Русский", "size": "~1.3 ГБ", "arch": "wav2vec2",
    },
    # ══ АНГЛИЙСКИЙ ═══════════════════════════════════════════════════════════
    "3": {
        "key":   "whisper-large-v3-en",
        "hf_id": "openai/whisper-large-v3",
        "label": "Whisper Large-v3 EN  ★ WER 2.7%  (РЕКОМЕНДУЕТСЯ)",
        "lang":  "Английский", "size": "~3.1 ГБ*", "arch": "whisper_hf",
    },
    "4": {
        "key":   "wav2vec2-en-lv60",
        "hf_id": "facebook/wav2vec2-large-960h-lv60-self",
        "label": "wav2vec2 EN Large LV60 — WER 1.9%  (лучший EN, чистая речь)",
        "lang":  "Английский", "size": "~1.3 ГБ", "arch": "wav2vec2",
    },
    # ══ НЕМЕЦКИЙ ═════════════════════════════════════════════════════════════
    "5": {
        "key":   "whisper-large-v3-de",
        "hf_id": "openai/whisper-large-v3",
        "label": "Whisper Large-v3 DE  ★ WER ~5%",
        "lang":  "Немецкий", "size": "~3.1 ГБ*", "arch": "whisper_hf",
    },
    # ══ ФРАНЦУЗСКИЙ ══════════════════════════════════════════════════════════
    "6": {
        "key":   "whisper-large-v3-fr",
        "hf_id": "openai/whisper-large-v3",
        "label": "Whisper Large-v3 FR  ★ WER ~5%",
        "lang":  "Французский", "size": "~3.1 ГБ*", "arch": "whisper_hf",
    },
    # ══ ИСПАНСКИЙ ════════════════════════════════════════════════════════════
    "7": {
        "key":   "whisper-large-v3-es",
        "hf_id": "openai/whisper-large-v3",
        "label": "Whisper Large-v3 ES  ★ WER ~5%",
        "lang":  "Испанский", "size": "~3.1 ГБ*", "arch": "whisper_hf",
    },
    # ══ УКРАИНСКИЙ ═══════════════════════════════════════════════════════════
    "8": {
        "key":   "whisper-large-v3-uk",
        "hf_id": "openai/whisper-large-v3",
        "label": "Whisper Large-v3 UK  ★ WER ~6%",
        "lang":  "Украинский", "size": "~3.1 ГБ*", "arch": "whisper_hf",
    },
    # ══ МНОГОЯЗЫЧНЫЕ ═════════════════════════════════════════════════════════
    "9": {
        "key":   "whisper-large-v3",
        "hf_id": "openai/whisper-large-v3",
        "label": "Whisper Large-v3 Multi  ★ 100+ языков, WER 2.7-6%",
        "lang":  "Многоязычный", "size": "~3.1 ГБ*", "arch": "whisper_hf",
    },
    "10": {
        "key":   "whisper-medium",
        "hf_id": "whisper-medium",
        "label": "Whisper Medium · 100 языков, WER ~8%  (быстрее, 1.5 ГБ)",
        "lang":  "Многоязычный", "size": "~1.5 ГБ", "arch": "whisper",
    },
    "11": {
        "key":   "whisper-small",
        "hf_id": "whisper-small",
        "label": "Whisper Small · 100 языков, WER ~12%  (самый быстрый, 460 МБ)",
        "lang":  "Многоязычный", "size": "~460 МБ", "arch": "whisper",
    },
}

# Пометка «*» означает: файлы на диске общие — занимает место только
# первая скачанная модель с этим движком; остальные используют hard-link.


# ─── Утилиты ────────────────────────────────────────────────────────────────

def _model_weights_present(local: Path) -> bool:
    """config.json — маленький файл метаданных, который скачивается за
    секунду. Сам файл весов может весить гигабайты и не докачаться (обрыв
    сети, антивирус, закрытие скрипта раньше времени). Без этой проверки
    is_downloaded()/_find_shared_model() считали бы такую папку полностью
    готовой — и вместо реальной докачки просто создавали hard-link на
    битые файлы, из раза в раз воспроизводя одну и ту же ошибку."""
    if (local / "model.safetensors").exists():
        return True
    if (local / "pytorch_model.bin").exists():
        return True
    if (local / "model.safetensors.index.json").exists() and list(local.glob("model-*-of-*.safetensors")):
        return True
    if (local / "pytorch_model.bin.index.json").exists() and list(local.glob("pytorch_model-*-of-*.bin")):
        return True
    return False


def is_downloaded(entry: dict) -> bool:
    arch = entry["arch"]
    key  = entry["key"]
    if arch == "whisper":
        cache = Path(os.path.expanduser("~")) / ".cache" / "whisper"
        name  = entry["hf_id"].replace("whisper-", "")
        return cache.exists() and any(cache.glob(f"{name}*"))
    local = MODELS_DIR / key
    return (local / "config.json").exists() and _model_weights_present(local)


def _clean_local_cache(local: Path):
    """
    Удаляет скрытую папку .cache, которую snapshot_download создаёт
    внутри local_dir. Она дублирует файлы весов и никак не нужна
    приложению — трансформеры читают модель напрямую из local_dir.
    Также удаляет pytorch_model.bin, если рядом есть model.safetensors.
    """
    hf_cache = local / ".cache"
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
        print(yellow(f"  [clean] Удалена папка .cache внутри {local.name}"))

    safe      = local / "model.safetensors"
    pt_bin    = local / "pytorch_model.bin"
    if safe.exists() and pt_bin.exists():
        pt_bin.unlink()
        print(yellow(f"  [clean] Удалён pytorch_model.bin (есть model.safetensors)"))

    # Шарды: pytorch_model-*.bin при наличии model-*.safetensors
    pt_shards = list(local.glob("pytorch_model-*.bin"))
    sf_shards = list(local.glob("model-*.safetensors"))
    if sf_shards and pt_shards:
        for s in pt_shards:
            s.unlink()
        print(yellow(f"  [clean] Удалено {len(pt_shards)} шардов pytorch (есть safetensors)"))


def _find_shared_model(hf_id: str, exclude_key: str) -> Path | None:
    """
    Ищет уже скачанную папку модели с тем же hf_id (кроме текущего ключа).
    Если нашли — можно создать hard-link вместо повторного скачивания.
    """
    for entry in CATALOG.values():
        if entry["key"] == exclude_key:
            continue
        if entry["hf_id"] == hf_id and entry["arch"] in ("whisper_hf", "wav2vec2"):
            candidate = MODELS_DIR / entry["key"]
            if (candidate / "config.json").exists() and _model_weights_present(candidate):
                return candidate
    return None


def _hardlink_dir(src: Path, dst: Path):
    """
    Рекурсивно создаёт hard-link (жёсткую ссылку) для каждого файла из src в dst.
    Hard-link не дублирует данные на диске — оба пути указывают на одни
    и те же блоки. Если hard-link невозможен (разные диски, FAT32),
    выполняется обычное копирование.
    Папка .cache пропускается.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == ".cache":
            continue
        dst_item = dst / item.name
        if item.is_dir():
            _hardlink_dir(item, dst_item)
        else:
            if dst_item.exists():
                continue
            try:
                os.link(item, dst_item)      # hard-link: 0 байт на диске
            except OSError:
                shutil.copy2(item, dst_item)  # fallback: обычная копия


def download_with_retry(fn, max_retries=5, delay=3):
    """Повторяет попытку при сетевых ошибках (до max_retries раз)."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = str(e)
            is_network = any(x in msg.lower() for x in
                ("10054", "10060", "connectionreset", "connection", "timeout",
                 "closed", "ssl", "eof", "remote", "broken"))
            if is_network and attempt < max_retries:
                print(yellow(f"  [!] Сетевая ошибка (попытка {attempt}/{max_retries}): {msg[:80]}"))
                print(f"  Повтор через {delay} сек...")
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise


def do_download(entry: dict) -> bool:
    key   = entry["key"]
    arch  = entry["arch"]
    hf_id = entry["hf_id"]
    local = MODELS_DIR / key

    print()
    print(bold(f"  Скачиваю: {entry['label']}"))
    print(f"  Размер на диске: {entry['size']}")
    print(f"  Источник: huggingface.co/{hf_id}")
    print()

    try:
        # ── Стандартный Whisper через openai-whisper ─────────────────────────
        if arch == "whisper":
            import whisper
            name = hf_id.replace("whisper-", "")
            print(f"  Загрузка Whisper '{name}'...")
            download_with_retry(lambda: whisper.load_model(name))

        # ── Whisper Large-v3 и wav2vec2 через HuggingFace Hub ───────────────
        elif arch in ("whisper_hf", "wav2vec2"):

            # ── КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ──────────────────────────────────────
            # Если у нас уже есть скачанная модель с тем же hf_id,
            # создаём hard-link вместо повторного скачивания ~3 ГБ.
            shared = _find_shared_model(hf_id, key)
            if shared:
                print(green(f"  ✓ Файлы уже скачаны в: {shared.name}"))
                print(f"  Создаю hard-link → {key}  (0 МБ дополнительно)")
                local.mkdir(exist_ok=True)
                _hardlink_dir(shared, local)
                print()
                print(green(bold(f"  ✓ Готово! Папка {key} создана без скачивания.")))
                print(f"  Модель занимает 0 дополнительного места (hard-link).")
                return True

            # ── Первая загрузка этого hf_id — скачиваем по-настоящему ──────
            from huggingface_hub import snapshot_download

            local.mkdir(exist_ok=True)

            # Скачиваем только PyTorch/safetensors + конфиги.
            # Явно исключаем тяжёлые дубликаты других фреймворков:
            ignore = [
                "flax_model*",          # JAX/Flax (~3 ГБ)
                "tf_model*",            # TensorFlow (~3 ГБ)
                "rust_model*",          # Rust
                "*.msgpack",            # Flax веса
                "*.ot",                 # OpenNLP
                "*.h5",                 # TF HDF5
                "pytorch_model.bin",    # старый PyTorch (предпочитаем safetensors)
                "pytorch_model-*.bin",  # шарды старого формата
            ]

            print("  Скачивание через snapshot_download (поддерживает докачку)...")
            print("  Прогресс — ниже:")
            print()

            def _dl():
                snapshot_download(
                    repo_id=hf_id,
                    local_dir=str(local),
                    local_dir_use_symlinks=False,
                    ignore_patterns=ignore,
                )
            download_with_retry(_dl)

            # ── Чистка лишних файлов ─────────────────────────────────────────
            # snapshot_download создаёт .cache внутри local_dir — удаляем.
            # Там лежат копии файлов в blob-формате — лишние ~3 ГБ.
            _clean_local_cache(local)

            # Если safetensors не скачался (старая модель без него) — докачиваем
            # pytorch_model.bin в виде исключения
            if not list(local.glob("model*.safetensors")):
                print(yellow("  Safetensors не найден — скачиваю pytorch_model.bin..."))
                ignore_no_pt = [p for p in ignore if "pytorch_model" not in p]
                def _dl2():
                    snapshot_download(
                        repo_id=hf_id,
                        local_dir=str(local),
                        local_dir_use_symlinks=False,
                        ignore_patterns=ignore_no_pt,
                    )
                download_with_retry(_dl2)
                _clean_local_cache(local)

            # ── Инициализируем processor, чтобы from_pretrained(local) работал
            if arch == "whisper_hf":
                from transformers import AutoProcessor
                print()
                print("  Инициализация processor...")
                proc = AutoProcessor.from_pretrained(str(local))
                proc.save_pretrained(str(local))

            elif arch == "wav2vec2":
                from transformers import AutoProcessor, Wav2Vec2Processor
                print()
                print("  Инициализация processor...")
                try:
                    proc = AutoProcessor.from_pretrained(str(local))
                except Exception as e:
                    if any(x in str(e).lower() for x in ("kenlm", "pyctcdecode", "ctcdecode")):
                        proc = Wav2Vec2Processor.from_pretrained(str(local))
                    else:
                        raise
                proc.save_pretrained(str(local))

        print()
        print(green(bold(f"  ✓ Модель успешно скачана!")))
        print(f"  Сохранена в: {local}")
        return True

    except KeyboardInterrupt:
        print(yellow("\n  Скачивание прервано. Запустите снова — докачка продолжится."))
        return False
    except Exception as e:
        err = str(e)
        print()
        print(red(f"  ✗ Ошибка: {err[:200]}"))
        print()
        print("  Попробуйте:")
        print("  1. Запустить снова — докачка продолжится автоматически")
        print("  2. Проверить интернет-соединение")
        print("  3. Попробовать другую модель")
        return False


def do_delete(entry: dict):
    arch  = entry["arch"]
    key   = entry["key"]
    hf_id = entry["hf_id"]

    if arch == "whisper":
        print(yellow("  Whisper-модели в ~/.cache/whisper — удалите папку вручную."))
        return

    local = MODELS_DIR / key
    if not local.exists():
        print(yellow("  Модель не найдена на диске."))
        return

    # Предупреждаем если другие модели используют hard-link на эти же файлы
    shared_users = [
        e["key"] for e in CATALOG.values()
        if e["key"] != key
        and e.get("hf_id") == hf_id
        and e["arch"] in ("whisper_hf", "wav2vec2")
        and (MODELS_DIR / e["key"] / "config.json").exists()
    ]
    if shared_users:
        print(yellow(f"  Внимание: эти файлы также используются моделями:"))
        for k in shared_users:
            print(yellow(f"    • {k}"))
        print(yellow(f"  Hard-link означает, что удаление {key} НЕ удалит общие данные."))
        print(yellow(f"  Удалятся только ссылки в папке {key}/."))

    confirm = input(f"\n  Удалить {entry['label']}? (да/нет): ").strip().lower()
    if confirm in ("да", "y", "yes", "д"):
        shutil.rmtree(local)
        print(green("  ✓ Удалено."))
    else:
        print("  Отменено.")


def print_catalog():
    os.system("cls" if os.name == "nt" else "clear")
    print(bold("=" * 68))
    print(bold("  TranscribeStudio — Менеджер языковых моделей"))
    print(bold("=" * 68))
    print()
    print(f"  {gray('* — файлы на диске общие (hard-link), дополнительного места не занимает)')}")
    print()
    current_lang = None
    for num, entry in CATALOG.items():
        if entry["lang"] != current_lang:
            current_lang = entry["lang"]
            print(f"  {cyan(bold(current_lang))}")
        dl  = is_downloaded(entry)
        tag = green("✓ скачана  ") if dl else gray("  не скачана")
        print(f"    [{bold(num):>3}]  {tag}  {entry['label']}  {gray(entry['size'])}")
    print()
    print(bold("-" * 68))
    print("  Номер — скачать     d<номер> — удалить     q — выход")
    print(bold("-" * 68))


def check_dependencies():
    missing = []
    for pkg in ["transformers", "torch", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(red(f"Ошибка: не установлены библиотеки: {', '.join(missing)}"))
        print("Запустите: pip install -r requirements.txt")
        input("\nEnter для выхода...")
        sys.exit(1)

    if importlib.util.find_spec("hf_transfer") is not None:
        print(yellow("  hf_transfer установлен, но отключён намеренно (см. комментарий выше) —"))
        print(yellow("  на части сетей он молча зависает без ошибки. Используется штатная загрузка."))


def main():
    if os.name == "nt":
        os.system("color")

    check_dependencies()
    time.sleep(1)

    while True:
        print_catalog()
        try:
            cmd = input("  Ваш выбор: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Выход.")
            break

        if cmd.lower() in ("q", "й", "quit", "exit"):
            print("  До свидания!")
            break

        if cmd.lower().startswith("d") and len(cmd) > 1:
            num = cmd[1:].strip()
            if num in CATALOG:
                do_delete(CATALOG[num])
            else:
                print(red(f"  Неверный номер: {num}"))
            input("\n  Enter для продолжения...")
            continue

        if cmd in CATALOG:
            entry = CATALOG[cmd]
            if is_downloaded(entry):
                print(green(f"\n  Уже скачана: {entry['label']}"))
                ans = input("  Скачать заново? (да/нет): ").strip().lower()
                if ans not in ("да", "y", "д"):
                    input("\n  Enter для продолжения...")
                    continue
            do_download(entry)
            input("\n  Enter для продолжения...")
        else:
            print(red(f"  Введите число от 1 до {len(CATALOG)}."))
            input("  Enter...")


if __name__ == "__main__":
    main()
