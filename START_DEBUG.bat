@echo off
chcp 65001 >nul
title TranscribeStudio — Диагностика
cd /d "%~dp0"

if exist "%~dp0python\python.exe" (
    set "PYTHON=%~dp0python\python.exe"
) else (
    set "PYTHON=python"
)
if exist "%~dp0ffmpeg\ffmpeg.exe" set "PATH=%~dp0ffmpeg;%PATH%"

echo Запуск TranscribeStudio (режим диагностики)...
echo Если ниже появятся ошибки — скопируйте их и сообщите разработчику.
echo.

"%PYTHON%" main.py

echo.
echo Приложение завершилось. Код ошибки: %errorlevel%
pause
