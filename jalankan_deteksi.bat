@echo off
title Smart Shoe Monitor - Lab Teknik Informatika
echo ============================================================
echo   SMART SHOE MONITOR - LAB TEKNIK INFORMATIKA (REAL-TIME)
echo ============================================================
echo.
echo Menjalankan aplikasi deteksi video real-time...
echo Kamera default: Webcam 0
echo Tekan tombol 'q' pada jendela video untuk berhenti.
echo.

cd /d "%~dp0"
where python >nul 2>&1
if %errorlevel% equ 0 (
    python realtime_detect.py --source 0
) else (
    "C:\Users\Raihan aryanova\AppData\Local\Programs\Python\Python39\python.exe" realtime_detect.py --source 0
)

pause
