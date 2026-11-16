# TranscribeStudio — Автоматическая установка
# Запускается через setup.bat

param()

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'
$ROOT = $PSScriptRoot

# ── Утилиты вывода ─────────────────────────────────────────────────────────
function Hdr($msg)  { Write-Host "`n  $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Err($msg)  {
    Write-Host "`n  ╔═══════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "  ║  ОШИБКА: $msg" -ForegroundColor Red
    Write-Host "  ╚═══════════════════════════════════════════╝" -ForegroundColor Red
}
function Info($msg) { Write-Host "  $msg" -ForegroundColor Gray }

# ── Fix-Pth: ВСЕГДА перезаписывает python*._pth без BOM ───────────────────
# Это главная функция которая исправляет проблему \ufeffpython311.zip.
# Логика: читаем сырые байты → вручную срезаем BOM если есть →
# раскомментируем import site → записываем обратно через System.IO.File
# с явным UTF8Encoding($false) — это единственный способ гарантированно
# записать UTF-8 без BOM в PowerShell 5.
function Fix-Pth($pyDir) {
    $pthFile = Get-Item (Join-Path $pyDir "python*._pth") -ErrorAction SilentlyContinue
    if (-not $pthFile) { Warn "python*._pth не найден."; return }

    # Читаем как байты — Get-Content/Set-Content нельзя доверять с BOM
    $raw = [System.IO.File]::ReadAllBytes($pthFile.FullName)

    # Проверяем и снимаем BOM (EF BB BF) прямо в байтах
    $hasBOM = ($raw.Length -ge 3 -and $raw[0] -eq 0xEF -and
               $raw[1] -eq 0xBB -and $raw[2] -eq 0xBF)
    if ($hasBOM) { $raw = $raw[3..($raw.Length - 1)] }

    # Декодируем — теперь без BOM
    $text = [System.Text.Encoding]::UTF8.GetString($raw)

    # Раскомментируем строку import site (убираем # если есть)
    $fixed = $text -replace '(?m)^#?import site', 'import site'

    # Записываем БЕЗ BOM — ВСЕГДА, даже если содержимое не изменилось.
    # Именно это гарантирует что BOM не останется на диске.
    $enc = New-Object System.Text.UTF8Encoding $false   # $false = NO BOM
    [System.IO.File]::WriteAllText($pthFile.FullName, $fixed, $enc)

    # Проверяем что BOM действительно ушёл
    $verify = [System.IO.File]::ReadAllBytes($pthFile.FullName)
    $bomGone = -not ($verify[0] -eq 0xEF -and $verify[1] -eq 0xBB -and $verify[2] -eq 0xBF)

    if ($bomGone) {
        if ($hasBOM) {
            Ok "$($pthFile.Name) — BOM удалён, import site активирован."
        } else {
            Ok "$($pthFile.Name) — import site активирован."
        }
    } else {
        Warn "Не удалось удалить BOM из $($pthFile.Name) — попробуйте запустить от имени администратора."
    }
}

# ── Скачивание с прогресс-баром ────────────────────────────────────────────
function Get-FileProgress($url, $dest, $label) {
    Info "Скачивание: $label"
    try {
        $wc = New-Object System.Net.WebClient
        $global:dlDone = $false
        $global:dlError = $null
        Register-ObjectEvent -InputObject $wc -EventName DownloadProgressChanged -Action {
            $pct = $EventArgs.ProgressPercentage
            $mb  = [math]::Round($EventArgs.BytesReceived / 1MB, 1)
            Write-Progress -Activity "Загрузка $label" -Status "$mb MB ($pct%)" -PercentComplete $pct
        } | Out-Null
        Register-ObjectEvent -InputObject $wc -EventName DownloadFileCompleted -Action {
            $global:dlDone  = $true
            $global:dlError = $EventArgs.Error
        } | Out-Null
        $wc.DownloadFileAsync([uri]$url, $dest)
        while (-not $global:dlDone) { Start-Sleep -Milliseconds 200 }
        Write-Progress -Activity "Загрузка $label" -Completed
        $wc.Dispose()
        if ($global:dlError) { throw $global:dlError }
    } catch {
        Err "Не удалось скачать $label : $_"
        Read-Host "  Enter для выхода" ; exit 1
    }
}

# ══════════════════════════════════════════════════════════════════════════
# Шапка
# ══════════════════════════════════════════════════════════════════════════
Clear-Host
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║        TranscribeStudio — Первоначальная настройка   ║" -ForegroundColor Cyan
Write-Host "  ║  Запускается ОДИН РАЗ. После этого только START.bat  ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Скрипт скачает и настроит:" -ForegroundColor White
Write-Host "    Python 3.11  — ~25  MB" -ForegroundColor Gray
Write-Host "    FFmpeg       — ~90  MB" -ForegroundColor Gray
Write-Host "    Python-пакеты — ~1.5 GB  (torch, transformers, whisper...)" -ForegroundColor Gray
Write-Host ""
Write-Host "  Нужен интернет. Время: 15-40 минут." -ForegroundColor Yellow
Write-Host "  Не закрывайте это окно!" -ForegroundColor Yellow
Write-Host ""
Read-Host "  Нажмите Enter чтобы начать"

# ══════════════════════════════════════════════════════════════════════════
# Проверка интернета
# ══════════════════════════════════════════════════════════════════════════
Hdr "Проверка подключения к интернету..."
try {
    $null = (New-Object System.Net.WebClient).DownloadString('https://pypi.org')
    Ok "Интернет доступен."
} catch {
    Err "Нет подключения к интернету. Подключитесь к сети и запустите снова."
    Read-Host "`n  Enter для выхода" ; exit 1
}

# ══════════════════════════════════════════════════════════════════════════
# Шаг 1 — Python embeddable
# ══════════════════════════════════════════════════════════════════════════
$pyDir     = Join-Path $ROOT "python"
$pythonExe = Join-Path $pyDir "python.exe"
Hdr "[1/4] Python 3.11..."

if (-not (Test-Path $pythonExe)) {
    $pyZip = Join-Path $ROOT "py_embed.zip"
    Get-FileProgress `
        "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" `
        $pyZip "Python 3.11"
    Info "Распаковка..."
    New-Item -ItemType Directory -Force -Path $pyDir | Out-Null
    Expand-Archive -Path $pyZip -DestinationPath $pyDir -Force
    Remove-Item $pyZip -Force
    Ok "Python распакован."
} else {
    Ok "Python уже скачан."
}

# ── Всегда исправляем .pth — гарантируем отсутствие BOM ──────────────────
Fix-Pth $pyDir

# ── Pip ────────────────────────────────────────────────────────────────────
$pipExe = Join-Path $pyDir "Scripts\pip.exe"
if (-not (Test-Path $pipExe)) {
    Info "Установка pip..."
    $getPip = Join-Path $ROOT "get-pip.py"
    Get-FileProgress "https://bootstrap.pypa.io/get-pip.py" $getPip "get-pip.py"

    $result = & $pythonExe $getPip --quiet --no-warn-script-location 2>&1
    $exitCode = $LASTEXITCODE
    Remove-Item $getPip -Force -ErrorAction SilentlyContinue

    if ($exitCode -ne 0) {
        Err "pip не установился (код $exitCode). Вывод:`n$result"
        Write-Host "  Запустите setup.bat ещё раз." -ForegroundColor Yellow
        Read-Host "  Enter для выхода" ; exit 1
    }
    Ok "pip установлен."
} else {
    Ok "pip уже установлен."
}

# ══════════════════════════════════════════════════════════════════════════
# Шаг 2 — FFmpeg
# ══════════════════════════════════════════════════════════════════════════
$ffDir = Join-Path $ROOT "ffmpeg"
Hdr "[2/4] FFmpeg..."

if (Test-Path (Join-Path $ffDir "ffmpeg.exe")) {
    Ok "FFmpeg уже установлен."
} else {
    $ffZip = Join-Path $ROOT "ffmpeg.zip"
    Get-FileProgress `
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" `
        $ffZip "FFmpeg"

    Info "Распаковка..."
    $ffTmp = Join-Path $ROOT "ffmpeg_tmp"
    New-Item -ItemType Directory -Force -Path $ffTmp | Out-Null
    Expand-Archive -Path $ffZip -DestinationPath $ffTmp -Force
    Remove-Item $ffZip -Force

    New-Item -ItemType Directory -Force -Path $ffDir | Out-Null
    $ffExe = Get-ChildItem -Path $ffTmp -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
    if ($ffExe) {
        Copy-Item $ffExe.FullName (Join-Path $ffDir "ffmpeg.exe") -Force
        Ok "ffmpeg.exe готов."
    } else {
        Warn "ffmpeg.exe не найден в архиве."
    }
    Remove-Item $ffTmp -Recurse -Force
}

# ══════════════════════════════════════════════════════════════════════════
# Шаг 3 — Python-пакеты
# ══════════════════════════════════════════════════════════════════════════
Hdr "[3/4] Установка Python-пакетов..."
Write-Host "  (torch ~2.5 ГБ с поддержкой GPU + transformers + whisper — займёт время)" -ForegroundColor Gray
Write-Host ""

# ── torch/torchaudio ставим ОТДЕЛЬНО от остального requirements.txt ────────
# Обычный "pip install torch" без --index-url на практике устанавливает
# CPU-only сборку даже на машинах с NVIDIA-видеокартой — Demucs и
# распознавание речи в этом случае молча работают на процессоре в разы
# медленнее, хотя GPU всё это время простаивает. Определяем видеокарту и
# ставим подходящую сборку.
$hasNvidia = $false
try {
    $gpus = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop |
            Select-Object -ExpandProperty Name
    $hasNvidia = ($gpus -join " ") -match "NVIDIA"
} catch {
    Warn "Не удалось определить видеокарту — ставлю обычную (CPU) сборку torch."
}

if ($hasNvidia) {
    Ok "Обнаружена видеокарта NVIDIA — ставлю torch с поддержкой CUDA (GPU)."
    & $pythonExe -m pip install "torch>=2.6" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu126 --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        # PyTorch периодически переносит "текущий" индекс на новую версию
        # CUDA — пробуем соседние индексы, прежде чем откатываться на CPU.
        & $pythonExe -m pip install "torch>=2.6" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location
    }
    if ($LASTEXITCODE -ne 0) {
        & $pythonExe -m pip install "torch>=2.6" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu124 --no-warn-script-location
    }
    if ($LASTEXITCODE -ne 0) {
        Warn "Не удалось поставить GPU-сборку torch — пробую обычную (CPU)."
        & $pythonExe -m pip install "torch>=2.6" "torchaudio>=2.6" --no-warn-script-location
    }
} else {
    Write-Host "  NVIDIA-видеокарта не обнаружена — ставлю обычную (CPU) сборку torch." -ForegroundColor Gray
    & $pythonExe -m pip install "torch>=2.6" "torchaudio>=2.6" --no-warn-script-location
}
if ($LASTEXITCODE -ne 0) {
    Err "Ошибка при установке torch."
    Write-Host "  Запустите setup.bat ещё раз — продолжится с места остановки." -ForegroundColor Yellow
    Read-Host "  Enter для выхода" ; exit 1
}

$req = Join-Path $ROOT "requirements.txt"
& $pythonExe -m pip install -r $req --no-warn-script-location
if ($LASTEXITCODE -ne 0) {
    Err "Ошибка при установке пакетов."
    Write-Host "  Запустите setup.bat ещё раз — продолжится с места остановки." -ForegroundColor Yellow
    Read-Host "  Enter для выхода" ; exit 1
}

& $pythonExe -m pip install pywebview --quiet --no-warn-script-location
Ok "Все пакеты установлены."

# ══════════════════════════════════════════════════════════════════════════
# Шаг 4 — Проверка
# ══════════════════════════════════════════════════════════════════════════
Hdr "[4/4] Проверка..."

$check = & $pythonExe -c "import flask, whisper, transformers, torch, webview; print('OK')" 2>&1
if ($check -match "OK") {
    Ok "Все Python-компоненты на месте."
} else {
    Warn "Проблема: $check"
    Warn "Попробуйте запустить setup.bat ещё раз."
}

$cudaCheck = & $pythonExe -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
if ($cudaCheck -match "CUDA") {
    Ok "torch видит видеокарту (GPU) — Demucs и распознавание будут работать на ней."
} elseif ($cudaCheck -match "CPU") {
    Write-Host "  torch будет работать на CPU (видеокарта не используется)." -ForegroundColor Gray
}

$ffBin = Join-Path $ffDir "ffmpeg.exe"
if (Test-Path $ffBin) {
    $ver = & $ffBin -version 2>&1 | Select-Object -First 1
    if ($ver -match "ffmpeg version") { Ok "FFmpeg работает." }
    else { Warn "FFmpeg не отвечает." }
}

# ══════════════════════════════════════════════════════════════════════════
# Финал
# ══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║             Установка успешно завершена!             ║" -ForegroundColor Green
Write-Host "  ║                                                      ║" -ForegroundColor Green
Write-Host "  ║  Дальнейшие шаги:                                    ║" -ForegroundColor Green
Write-Host "  ║  1. Запустите START.bat                              ║" -ForegroundColor Green
Write-Host "  ║  2. Откройте вкладку Модели и скачайте модель        ║" -ForegroundColor Green
Write-Host "  ║  3. Создайте проект и загрузите аудио                ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Read-Host "  Нажмите Enter для выхода"
