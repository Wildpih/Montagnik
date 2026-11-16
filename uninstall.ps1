# TranscribeStudio — Удаление
# Запускается через uninstall.bat

param()

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'SilentlyContinue'

$ROOT = $PSScriptRoot

# ── Утилиты ────────────────────────────────────────────────────────────────
function Title($msg) {
    Write-Host ""
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("  " + "─" * ($msg.Length + 2)) -ForegroundColor DarkGray
}
function Ok($msg)     { Write-Host "  [✓] $msg" -ForegroundColor Green }
function Warn($msg)   { Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Info($msg)   { Write-Host "  $msg"     -ForegroundColor Gray }
function Deleted($msg){ Write-Host "  [🗑] $msg" -ForegroundColor DarkYellow }

function Get-FolderSizeMB($path) {
    if (-not (Test-Path $path)) { return 0 }
    $size = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
             Measure-Object -Property Length -Sum).Sum
    return [math]::Round($size / 1MB, 0)
}

function Stop-AppProcesses {
    # Останавливаем процессы приложения (python/pythonw/ffmpeg), которые
    # запущены ИЗ ЭТОЙ папки — иначе они держат файлы открытыми и мешают
    # удалению (в т.ч. после закрытия окна фоновые задачи могут ещё жить).
    Get-Process -Name 'python','pythonw','ffmpeg' -ErrorAction SilentlyContinue |
        Where-Object {
            try { $_.Path -and $_.Path.StartsWith($ROOT, [System.StringComparison]::OrdinalIgnoreCase) }
            catch { $false }
        } |
        ForEach-Object {
            try {
                $_.Kill()
                $_.WaitForExit(3000)
            } catch {}
        }
}

function Remove-FolderSafe($path, $label) {
    if (-not (Test-Path $path)) {
        Info "$label — папка не найдена, пропускаю."
        return 0
    }
    $mb = Get-FolderSizeMB $path

    # Попытка 1 — обычное удаление
    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue

    # Попытка 2 — через cmd (rd иногда справляется там, где Remove-Item — нет)
    if (Test-Path $path) {
        cmd /c "rd /s /q `"$path`"" 2>$null | Out-Null
    }

    # Попытка 3 — robocopy mirror-trick. Это самый надёжный способ снести
    # «неудаляемую» папку в Windows: часто причина в том, что site-packages
    # torch/transformers создают пути глубже 260 символов, и штатное
    # удаление молча пропускает такие файлы, оставляя мусор на диске.
    if (Test-Path $path) {
        Info "Обычное удаление не справилось (возможно, слишком длинные пути) — пробую через robocopy..."
        $empty = Join-Path $env:TEMP ("ts_empty_" + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Force -Path $empty | Out-Null
        robocopy $empty $path /MIR /NFL /NDL /NJH /NJS /NC /NS /NP *> $null
        Remove-Item -LiteralPath $empty -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path $path)) {
        Deleted "$label удалена  (-$mb МБ)"
        return $mb
    } else {
        Warn "Не удалось полностью удалить $label."
        Warn "Закройте приложение (если оно ещё где-то запущено), перезагрузите"
        Warn "компьютер и запустите uninstall.bat ещё раз."
        return 0
    }
}

function Ask($question) {
    Write-Host ""
    Write-Host "  $question" -ForegroundColor White
    $ans = Read-Host "  [д/н]"
    return ($ans.Trim().ToLower() -in @('д','да','y','yes'))
}

Stop-AppProcesses

# ── Шапка ──────────────────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Red
Write-Host "  ║           TranscribeStudio — Удаление                ║" -ForegroundColor Red
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Red
Write-Host ""
Write-Host "  Этот мастер поможет удалить компоненты приложения." -ForegroundColor White
Write-Host "  Вы сами выбираете что удалять — ничего не пропадёт" -ForegroundColor White
Write-Host "  без вашего разрешения." -ForegroundColor White
Write-Host ""

# ── Сводка того что установлено ─────────────────────────────────────────────
Title "Что установлено на этом компьютере"

$dirs = @{
    "Python + пакеты  (python/)"         = Join-Path $ROOT "python"
    "FFmpeg           (ffmpeg/)"          = Join-Path $ROOT "ffmpeg"
    "Языковые модели  (models/)"          = Join-Path $ROOT "models"
    "Проекты          (projects/)"        = Join-Path $ROOT "projects"
    "Загруженные файлы(uploads/)"         = Join-Path $ROOT "uploads"
    "Экспорт          (exports/)"         = Join-Path $ROOT "exports"
}

$totalMB = 0
foreach ($kv in $dirs.GetEnumerator()) {
    $mb = Get-FolderSizeMB $kv.Value
    $totalMB += $mb
    if ($mb -gt 0) {
        $sizeStr = if ($mb -ge 1000) { "$([math]::Round($mb/1000,1)) ГБ" } else { "$mb МБ" }
        Info ("  {0,-40} {1}" -f $kv.Key, $sizeStr)
    } else {
        Info ("  {0,-40} не найдена" -f $kv.Key)
    }
}
$totalStr = if ($totalMB -ge 1000) { "$([math]::Round($totalMB/1000,1)) ГБ" } else { "$totalMB МБ" }
Write-Host ""
Write-Host "  Итого занято: $totalStr" -ForegroundColor White
Write-Host ""
Read-Host "  Нажмите Enter чтобы продолжить"

# ══════════════════════════════════════════════════════════════════════════
# БЛОК 1 — Программные файлы (Python + FFmpeg)
# ══════════════════════════════════════════════════════════════════════════
Title "Блок 1 / 4 — Программные файлы"
Write-Host ""
Write-Host "  Папки python/ и ffmpeg/ были созданы при установке." -ForegroundColor Gray
Write-Host "  Без них приложение не запустится." -ForegroundColor Gray

$pyMB  = Get-FolderSizeMB (Join-Path $ROOT "python")
$ffMB  = Get-FolderSizeMB (Join-Path $ROOT "ffmpeg")
$progMB = $pyMB + $ffMB
$progStr = if ($progMB -ge 1000) { "$([math]::Round($progMB/1000,1)) ГБ" } else { "$progMB МБ" }

Write-Host "  Будет освобождено: $progStr" -ForegroundColor Yellow

$delProg = Ask "Удалить Python и FFmpeg? ($progStr)"

$freedMB = 0

if ($delProg) {
    Write-Host ""
    $freedMB += Remove-FolderSafe (Join-Path $ROOT "python") "Python + пакеты"
    $freedMB += Remove-FolderSafe (Join-Path $ROOT "ffmpeg") "FFmpeg"
} else {
    Ok "Python и FFmpeg оставлены."
}

# ══════════════════════════════════════════════════════════════════════════
# БЛОК 2 — Языковые модели
# ══════════════════════════════════════════════════════════════════════════
Title "Блок 2 / 4 — Языковые модели"
Write-Host ""

$modelsMB = Get-FolderSizeMB (Join-Path $ROOT "models")
if ($modelsMB -eq 0) {
    Info "Папка models/ пуста или не существует — нечего удалять."
    $delModels = $false
} else {
    $modelsStr = if ($modelsMB -ge 1000) { "$([math]::Round($modelsMB/1000,1)) ГБ" } else { "$modelsMB МБ" }
    Write-Host "  Скачанные модели занимают $modelsStr." -ForegroundColor Gray
    Write-Host "  После удаления их придётся скачивать заново ($modelsStr интернет-трафика)." -ForegroundColor Gray
    $delModels = Ask "Удалить языковые модели? ($modelsStr)"
}

if ($delModels) {
    $freedMB += Remove-FolderSafe (Join-Path $ROOT "models") "Языковые модели"
} elseif ($modelsMB -gt 0) {
    Ok "Модели оставлены. При повторной установке они уже будут готовы."
}

# ══════════════════════════════════════════════════════════════════════════
# БЛОК 3 — Данные проектов
# ══════════════════════════════════════════════════════════════════════════
Title "Блок 3 / 4 — Ваши проекты и файлы"
Write-Host ""

$projMB    = Get-FolderSizeMB (Join-Path $ROOT "projects")
$uploadsMB = Get-FolderSizeMB (Join-Path $ROOT "uploads")
$exportsMB = Get-FolderSizeMB (Join-Path $ROOT "exports")
$dataMB    = $projMB + $uploadsMB + $exportsMB

if ($dataMB -eq 0) {
    Info "Пользовательских данных нет — нечего удалять."
    $delData = $false
} else {
    $dataStr = if ($dataMB -ge 1000) { "$([math]::Round($dataMB/1000,1)) ГБ" } else { "$dataMB МБ" }
    Write-Host "  Здесь хранятся ваши транскрипты, загруженные аудио/видео и экспорты." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  ┌──────────────────────────────────────────────┐" -ForegroundColor Red
    Write-Host "  │  ⚠  ВНИМАНИЕ — ЭТО НЕЛЬЗЯ ВОССТАНОВИТЬ  ⚠  │" -ForegroundColor Red
    Write-Host "  │  Все проекты и транскрипты будут удалены.    │" -ForegroundColor Red
    Write-Host "  │  Сделайте резервные копии заранее!           │" -ForegroundColor Red
    Write-Host "  └──────────────────────────────────────────────┘" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Будет удалено:" -ForegroundColor Gray
    if ($projMB -gt 0)    { Info "  • Проекты (projects/)       — $projMB МБ" }
    if ($uploadsMB -gt 0) { Info "  • Загруженные файлы (uploads/) — $uploadsMB МБ" }
    if ($exportsMB -gt 0) { Info "  • Файлы экспорта (exports/)  — $exportsMB МБ" }
    Write-Host ""

    $delData = Ask "Удалить ВСЕ проекты и файлы? ($dataStr) — НЕОБРАТИМО"

    if ($delData) {
        # Второе подтверждение для пользовательских данных
        Write-Host ""
        Write-Host "  Вы уверены? Восстановить транскрипты будет невозможно." -ForegroundColor Red
        $confirm2 = Ask "Подтвердите: удалить данные безвозвратно?"
        if (-not $confirm2) {
            $delData = $false
            Ok "Данные проектов оставлены."
        }
    }
}

if ($delData) {
    $freedMB += Remove-FolderSafe (Join-Path $ROOT "projects") "Проекты"
    $freedMB += Remove-FolderSafe (Join-Path $ROOT "uploads")  "Загруженные файлы"
    $freedMB += Remove-FolderSafe (Join-Path $ROOT "exports")  "Файлы экспорта"
} elseif ($dataMB -gt 0 -and -not $delData) {
    Ok "Проекты и файлы оставлены."
}

# ══════════════════════════════════════════════════════════════════════════
# БЛОК 4 — Полное удаление программы (файлы, ярлыки, реестр)
# ══════════════════════════════════════════════════════════════════════════
# Предлагается только если пользователь согласился удалить ВСЁ содержимое
# (python/ffmpeg + модели + проекты). Если хоть что-то решено оставить —
# этот блок пропускается, и uninstall.bat остаётся в папке нетронутым,
# чтобы можно было завершить удаление позже.
Title "Блок 4 / 4 — Файлы программы и ярлыки"
Write-Host ""

$delAll = $false
if ($delProg -and $delModels -and $delData) {
    Write-Host "  Вы удалили все компоненты (Python, FFmpeg, модели, проекты)." -ForegroundColor Gray
    Write-Host "  Можно также удалить оставшиеся файлы программы (~1 МБ)," -ForegroundColor Gray
    Write-Host "  ярлыки на рабочем столе / в меню «Пуск» и запись в реестре." -ForegroundColor Gray
    $delAll = Ask "Удалить программу полностью (включая ярлыки)?"
} else {
    Info "Пропущено — вы решили оставить часть данных (см. блоки выше)."
}

if ($delAll) {
    Write-Host ""
    Info "Удаляю файлы программы..."
    $appFiles = @(
        "app.py", "main.py", "requirements.txt", "setup.bat", "setup.ps1",
        "START.bat", "START_DEBUG.bat", "DOWNLOAD_MODELS.bat",
        "download_models.py", "demucs_run.py", "ИНСТРУКЦИЯ.txt",
        "FIX_GPU_TORCH.bat", "fix_gpu_torch.ps1"
    )
    foreach ($f in $appFiles) {
        Remove-Item (Join-Path $ROOT $f) -Force -ErrorAction SilentlyContinue
    }
    Remove-Item (Join-Path $ROOT "templates") -Recurse -Force -ErrorAction SilentlyContinue
    Ok "Файлы программы удалены."

    Info "Удаляю ярлыки..."
    $desktopLnk = Join-Path ([Environment]::GetFolderPath('Desktop')) "TranscribeStudio.lnk"
    Remove-Item $desktopLnk -Force -ErrorAction SilentlyContinue
    $startMenuDir = Join-Path ([Environment]::GetFolderPath('StartMenu')) "Programs\TranscribeStudio"
    Remove-Item $startMenuDir -Recurse -Force -ErrorAction SilentlyContinue
    Ok "Ярлыки удалены."

    Info "Очищаю запись в реестре..."
    Remove-Item "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\TranscribeStudio" -Force -ErrorAction SilentlyContinue
    Remove-Item "HKCU:\Software\TranscribeStudio" -Force -ErrorAction SilentlyContinue
    Ok "Реестр очищен."
}

# ══════════════════════════════════════════════════════════════════════════
# Итог
# ══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║                    Готово                            ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$freedStr = if ($freedMB -ge 1000) { "$([math]::Round($freedMB/1000,1)) ГБ" } else { "$freedMB МБ" }

if ($freedMB -gt 0) {
    Ok "Освобождено: $freedStr"
} else {
    Info "Ничего не удалено."
}

Write-Host ""

if ($delAll) {
    Write-Host "  Программа удалена полностью." -ForegroundColor Green
    Write-Host "  В папке остались только uninstall.bat / uninstall.ps1" -ForegroundColor Gray
    Write-Host "  (и Uninstall.exe, если приложение ставилось через установщик —" -ForegroundColor Gray
    Write-Host "  он снесёт и их, и саму папку сразу после этого окна)." -ForegroundColor Gray
    Write-Host "  Если вы запускали uninstall.bat напрямую — удалите эту" -ForegroundColor Gray
    Write-Host "  небольшую оставшуюся папку вручную через проводник." -ForegroundColor Gray
} else {
    Write-Host "  Файл uninstall.bat остался в папке приложения —" -ForegroundColor Gray
    Write-Host "  запустите его снова позже, чтобы удалить оставшееся." -ForegroundColor Gray
}

Write-Host ""
Read-Host "  Нажмите Enter для выхода"
