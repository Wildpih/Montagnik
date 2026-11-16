@echo off
chcp 65001 >nul
title TranscribeStudio — Менеджер моделей
color 0A
cd /d "%~dp0"

REM ── Выбираем Python ───────────────────────────────────────────────────────
if exist "%~dp0python\python.exe" (
    set "PYTHON=%~dp0python\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        color 0C
        echo.
        echo  ОШИБКА: Python не найден.
        echo  Запустите setup.bat для автоматической установки.
        echo.
        pause & exit /b 1
    )
    set "PYTHON=python"
)

echo.
echo  ============================================
echo   TranscribeStudio — Менеджер языковых моделей
echo   Скачивание и удаление моделей
echo  ============================================
echo.

REM Абсолютный путь — embedded Python не видит текущую папку
"%PYTHON%" "%~dp0download_models.py"
echo.
pause
