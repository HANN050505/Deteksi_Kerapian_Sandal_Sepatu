@echo off
setlocal
title Smart Shoe Monitor - Lab Teknik Informatika (Real-Time)
echo ============================================================
echo   SMART SHOE MONITOR - LAB TEKNIK INFORMATIKA (REAL-TIME)
echo ============================================================
echo.

cd /d "%~dp0"

echo Menjalankan aplikasi deteksi video real-time...
echo - Kamera default: Webcam 0
echo - Tekan 'q' pada jendela video untuk berhenti.
echo - Tekan 's' untuk snapshot manual.
echo.

if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" realtime_detect.py --source 0
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py -3.9 realtime_detect.py --source 0
    ) else (
        python realtime_detect.py --source 0
    )
)

pause
