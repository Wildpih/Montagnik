@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist "%~dp0python\pythonw.exe" (
    set "PYTHON=%~dp0python\pythonw.exe"
) else if exist "%~dp0python\python.exe" (
    set "PYTHON=%~dp0python\python.exe"
) else (
    where pythonw >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=pythonw"
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo.
            echo  ERROR: Python not found. Run setup.bat to install.
            echo.
            pause & exit /b 1
        )
        set "PYTHON=python"
    )
)

if exist "%~dp0ffmpeg\ffmpeg.exe" set "PATH=%~dp0ffmpeg;%PATH%"
if exist "%~dp0start_error.log" del /f /q "%~dp0start_error.log"

start "" "%PYTHON%" "%~dp0main.py"

timeout /t 6 /nobreak >nul

if exist "%~dp0start_error.log" (
    echo.
    echo  TranscribeStudio failed to start:
    echo  ----------------------------------------
    type "%~dp0start_error.log"
    echo  ----------------------------------------
    echo.
    echo  If you see 'No module named' - run setup.bat again.
    echo  For diagnostics use START_DEBUG.bat
    echo.
    pause
)
