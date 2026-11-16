# TranscribeStudio — обновление yt-dlp до последней версии.
# Запускается через UPDATE_YTDLP.bat
#
# Зачем это нужно: YouTube регулярно меняет защиту от ботов и способ выдачи
# ссылок на файлы. yt-dlp выпускает обновления вслед за этим очень часто —
# иногда по нескольку раз в месяц. Если скачивание с YouTube вдруг
# перестало работать без каких-либо изменений в самом TranscribeStudio
# (типичная ошибка — "Sign in to confirm your age" или "Sign in to confirm
# you're not a bot"), в 9 из 10 случаев дело именно в устаревшей версии
# yt-dlp, а не в реальном возрастном ограничении видео. Этот скрипт просто
# обновляет yt-dlp до последней версии с PyPI.
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
Write-Host "  ║   TranscribeStudio — обновление yt-dlp                ║" -ForegroundColor Cyan
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

Hdr "[1/2] Проверка текущей версии yt-dlp..."
$before = & $pythonExe -c "import yt_dlp; print(yt_dlp.version.__version__)" 2>&1
Info "Сейчас установлено: $before"

Hdr "[2/2] Установка актуальной версии yt-dlp..."
& $pythonExe -m pip install --upgrade yt-dlp
if ($LASTEXITCODE -ne 0) {
    Err "Не удалось установить новую версию yt-dlp."
    Write-Host ""
    Warn "Частые причины:"
    Warn "  — нет интернета или заблокирован доступ к pypi.org"
    Warn "  — антивирус помешал загрузке"
    Write-Host ""
    Warn "Можно попробовать запустить этот файл ещё раз."
    Read-Host "`n  Enter для выхода"
    exit 1
}

$after = & $pythonExe -c "import yt_dlp; print(yt_dlp.version.__version__)" 2>&1

Hdr "[3/3] Очистка кэша yt-dlp (код плеера YouTube)..."
Info "Нужна, чтобы не осталось устаревшего закэшированного JS-плеера —"
Info "именно он часто вызывает ошибку 'The page needs to be reloaded'"
Info "даже после обновления yt-dlp до последней версии."
& $pythonExe -m yt_dlp --rm-cache-dir 2>&1 | Out-Null
Ok "Кэш очищен."

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║   Готово! yt-dlp обновлён: $before -> $after" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Ok "Можно снова запускать TranscribeStudio и пробовать скачивание с YouTube."
Write-Host ""
Info "Если ошибка 'The page needs to be reloaded' или 'Sign in to confirm...'"
Info "сохранится и после обновления + очистки кэша — подождите день-два:"
Info "иногда YouTube выкатывает изменения быстрее, чем выходит фикс в yt-dlp,"
Info "и тогда помогает только дождаться свежего релиза yt-dlp. Проверить"
Info "вручную: python.exe -m pip install --upgrade yt-dlp"
Write-Host ""
Info "Если же дело в авторизации конкретного видео — используйте кнопку"
Info "«Загрузить cookies.txt», экспортировав файл из браузера, где вы"
Info "залогинены в YouTube (расширение 'Get cookies.txt LOCALLY')."

Write-Host ""
Read-Host "  Нажмите Enter для выхода"
