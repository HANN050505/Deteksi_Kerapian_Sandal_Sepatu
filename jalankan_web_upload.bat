@echo off
title Smart Shoe Monitor - Web Dashboard Lab Teknik Informatika
echo ============================================================
echo   SMART SHOE MONITOR - WEB DASHBOARD LAB TEKNIK INFORMATIKA
echo ============================================================
echo.
echo Menjalankan aplikasi web Streamlit...
echo - Di Laptop: http://localhost:8501
echo - Di HP (iOS / Android):
echo   * Buka 'Network URL' yang tertera otomatis oleh Streamlit di bawah ini!
echo   * Contoh saat ini: http://10.5.6.249:8501 (atau IP Wi-Fi/Hotspot yang aktif)
echo   * CATATAN: HP dan Laptop WAJIB terhubung ke Wi-Fi / Hotspot yang sama.
echo Tekan Ctrl+C di terminal ini jika ingin menghentikan server.
echo.

cd /d "%~dp0"
where python >nul 2>&1
if %errorlevel% equ 0 (
    python -m streamlit run app_upload.py --server.address 0.0.0.0
) else (
    "C:\Users\Raihan aryanova\AppData\Local\Programs\Python\Python39\python.exe" -m streamlit run app_upload.py --server.address 0.0.0.0
)

pause
