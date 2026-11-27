@echo off
setlocal
title Smart Shoe Monitor - Lab Teknik Informatika
echo ============================================================
echo   SMART SHOE MONITOR - LAB TEKNIK INFORMATIKA (REAL-TIME)
echo ============================================================
echo.

cd /d "%~dp0"

if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_EXE=py -3"
    ) else (
        set "PY_EXE=python"
    )
)

echo Menjalankan aplikasi deteksi video real-time...
echo - Kamera default: Webcam 0
echo - Tekan 'q' pada jendela video untuk berhenti.
echo - Tekan 's' untuk snapshot manual.
echo.

%PY_EXE% realtime_detect.py --source 0

pause
