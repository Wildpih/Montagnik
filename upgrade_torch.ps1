# TranscribeStudio — обновление torch до последней версии.
# Запускается через UPGRADE_TORCH.bat
#
# Зачем это нужно: начиная с определённой версии transformers, загрузка
# моделей, чьи веса лежат в старом .bin-формате (не safetensors), требует
# PyTorch не старее 2.6 — это защита от уязвимости CVE-2025-32434 в
# torch.load. Если torch был установлен раньше выхода 2.6, такие модели
# (например, wav2vec2) отказываются загружаться с длинной англоязычной
# ошибкой. Этот скрипт просто ставит актуальный torch, СОХРАНЯЯ текущий тип
# сборки — если раньше была GPU-версия (CUDA), останется GPU-версия, если
# CPU-версия — останется CPU-версия. Для смены CPU→GPU используйте
# FIX_GPU_TORCH.bat, а не этот скрипт.
#
# ВАЖНО: перед запуском полностью закройте TranscribeStudio.

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
Write-Host "  ║   TranscribeStudio — обновление PyTorch (torch)      ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan

$pythonExe = Join-Path $ROOT "python\python.exe"
if (-not (Test-Path $pythonExe)) {
    Err "Не найден $pythonExe`nЗапустите этот файл из папки установки TranscribeStudio."
    Read-Host "`n  Enter для выхода"
    exit 1
}

$running = Get-Process -Name "python", "pythonw" -ErrorAction SilentlyContinue |
           Where-Object { $_.Path -and $_.Path.StartsWith($ROOT, [System.StringComparison]::OrdinalIgnoreCase) }
if ($running) {
    Warn "TranscribeStudio ещё запущен — закрываю, чтобы освободить файлы..."
    $running | ForEach-Object { try { $_.Kill(); $_.WaitForExit(5000) } catch {} }
    Start-Sleep -Seconds 1
    Ok "Приложение остановлено."
}

Hdr "[1/4] Проверка текущей версии torch..."
$before = & $pythonExe -c "import torch; print(torch.__version__); print(torch.version.cuda if torch.version.cuda else 'CPU')" 2>&1
$beforeLines = $before -split "`r?`n"
$wasCuda = ($beforeLines.Count -ge 2 -and $beforeLines[1] -ne 'CPU' -and $beforeLines[1] -ne '')
Info "Сейчас установлено: $($beforeLines[0]) / $(if($wasCuda){'GPU-сборка (CUDA ' + $beforeLines[1] + ')'}else{'CPU-сборка'})"

Hdr "[2/4] Удаление текущей версии torch..."
& $pythonExe -m pip uninstall torch torchaudio -y
if ($LASTEXITCODE -ne 0) {
    Err "Не удалось удалить старую версию torch."
    Read-Host "`n  Enter для выхода"
    exit 1
}
Ok "Старая версия удалена."

Hdr "[3/4] Установка актуальной версии torch..."
Info "Файл весит 1-3 ГБ — в зависимости от скорости интернета это может"
Info "занять от пары минут до получаса. Пожалуйста, дождитесь окончания."
Write-Host ""
if ($wasCuda) {
    Info "Сохраняю GPU-сборку (CUDA)..."
    & $pythonExe -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) {
        & $pythonExe -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    }
    if ($LASTEXITCODE -ne 0) {
        & $pythonExe -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    }
} else {
    Info "Сохраняю CPU-сборку..."
    & $pythonExe -m pip install --upgrade torch torchaudio
}
if ($LASTEXITCODE -ne 0) {
    Err "Не удалось установить новую версию torch."
    Write-Host ""
    Warn "Частые причины:"
    Warn "  — нет интернета или заблокирован доступ к download.pytorch.org / pypi.org"
    Warn "  — антивирус помешал загрузке"
    Write-Host ""
    Warn "Можно попробовать запустить этот файл ещё раз."
    Read-Host "`n  Enter для выхода"
    exit 1
}

Write-Host ""
Hdr "[4/4] Проверка результата..."
$after = & $pythonExe -c "import torch; print(torch.__version__); print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
$afterLines = $after -split "`r?`n"
Info "torch теперь: $($afterLines[0])"

if (($wasCuda -and $after -match "CUDA") -or (-not $wasCuda -and $after -notmatch "CUDA")) {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "  ║   Готово! torch обновлён, тип сборки сохранён.       ║" -ForegroundColor Green
    Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
    Ok "Можно снова запускать TranscribeStudio и пробовать распознавание."
} else {
    Write-Host ""
    Warn "torch обновлён, но тип сборки (GPU/CPU) изменился по сравнению с тем,"
    Warn "что было раньше. Если видеокарта пропала — запустите FIX_GPU_TORCH.bat."
}

Write-Host ""
Read-Host "  Нажмите Enter для выхода"
