# TranscribeStudio — исправление: установить torch с поддержкой GPU (CUDA)
# Запускается через FIX_GPU_TORCH.bat
#
# ВАЖНО: перед запуском полностью закройте TranscribeStudio. Если приложение
# запущено, файлы torch могут быть заблокированы Windows (процесс держит их
# открытыми), и переустановка либо не сработает, либо оставит папку в
# сломанном промежуточном состоянии.

param()

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'
$ROOT = $PSScriptRoot

function Hdr($msg)  { Write-Host "`n  $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Err($msg)  {
    Write-Host "`n  ╔═══════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "  ║  ОШИБКА: $msg" -ForegroundColor Red
    Write-Host "  ╚═══════════════════════════════════════════╝" -ForegroundColor Red
}
function Info($msg) { Write-Host "  $msg" -ForegroundColor Gray }

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║   TranscribeStudio — установка GPU-версии torch      ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan

$pythonExe = Join-Path $ROOT "python\python.exe"
if (-not (Test-Path $pythonExe)) {
    Err "Не найден $pythonExe`nЗапустите этот файл из папки установки TranscribeStudio."
    Read-Host "`n  Enter для выхода"
    exit 1
}

# ── Проверяем, что приложение закрыто ──────────────────────────────────────
# Файлы torch могут быть заблокированы, если процесс ещё жив — останавливаем
# его сами, чтобы не наткнуться на "Access is denied" в середине установки.
$running = Get-Process -Name "python", "pythonw" -ErrorAction SilentlyContinue |
           Where-Object { $_.Path -and $_.Path.StartsWith($ROOT, [System.StringComparison]::OrdinalIgnoreCase) }
if ($running) {
    Warn "TranscribeStudio ещё запущен — закрываю, чтобы освободить файлы..."
    $running | ForEach-Object { try { $_.Kill(); $_.WaitForExit(5000) } catch {} }
    Start-Sleep -Seconds 1
    Ok "Приложение остановлено."
}

Hdr "[1/3] Проверка текущей версии torch..."
$before = & $pythonExe -c "import torch; print(torch.__version__ + ' / cuda_build=' + str(torch.version.cuda))" 2>&1
Info "Сейчас установлено: $before"

Hdr "[2/3] Удаление текущей версии torch..."
& $pythonExe -m pip uninstall torch torchaudio -y
if ($LASTEXITCODE -ne 0) {
    Err "Не удалось удалить старую версию torch."
    Read-Host "`n  Enter для выхода"
    exit 1
}
Ok "Старая версия удалена."

Hdr "[3/3] Установка torch с поддержкой CUDA (GPU)..."
Info "Файл весит ~2.5 ГБ — в зависимости от скорости интернета это может"
Info "занять от пары минут до получаса. Пожалуйста, дождитесь окончания."
Write-Host ""
& $pythonExe -m pip install "torch>=2.6" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) {
    # PyTorch периодически переносит "текущий" индекс на новую версию CUDA
    # (было cu121, потом cu126...) — если основной индекс не сработал,
    # пробуем соседние, прежде чем сдаваться.
    Warn "cu126 не сработал, пробую cu128..."
    & $pythonExe -m pip install "torch>=2.6" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu128
}
if ($LASTEXITCODE -ne 0) {
    Warn "cu128 не сработал, пробую cu124..."
    & $pythonExe -m pip install "torch>=2.6" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/cu124
}
if ($LASTEXITCODE -ne 0) {
    Err "Не удалось установить GPU-версию torch."
    Write-Host ""
    Warn "Частые причины:"
    Warn "  — нет интернета или заблокирован доступ к download.pytorch.org"
    Warn "  — антивирус помешал загрузке"
    Warn "  — PyTorch перенёс актуальные версии на индекс новее, чем cu124/126/128"
    Write-Host ""
    Warn "Можно попробовать запустить этот файл ещё раз, либо вручную"
    Warn "проверить актуальный индекс на https://pytorch.org/get-started/locally/"
    Read-Host "`n  Enter для выхода"
    exit 1
}

Write-Host ""
Hdr "Проверка результата..."
$after = & $pythonExe -c "import torch; print(torch.__version__); print('CUDA' if torch.cuda.is_available() else 'CPU'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1
$afterLines = $after -split "`r?`n"
Info "torch: $($afterLines[0])"

if ($after -match "CUDA") {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "  ║   Готово! Видеокарта обнаружена: $($afterLines[2])" -ForegroundColor Green
    Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Ok "Откройте TranscribeStudio — бейдж в шапке должен стать зелёным (⚡ GPU: ...)."
} else {
    Write-Host ""
    Warn "torch установлен, но видеокарта всё ещё не обнаружена."
    Warn "Возможные причины:"
    Warn "  — видеокарта NVIDIA физически отсутствует"
    Warn "  — устарел драйвер NVIDIA — обновите через nvidia.com/drivers"
    Warn "    или приложение GeForce Experience"
}

Write-Host ""
Read-Host "  Нажмите Enter для выхода"
