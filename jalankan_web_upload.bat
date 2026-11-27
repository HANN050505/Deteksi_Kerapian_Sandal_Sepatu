@echo off
setlocal
title Smart Shoe Monitor - Web Dashboard Lab Teknik Informatika
echo ============================================================
echo   SMART SHOE MONITOR - WEB DASHBOARD LAB TEKNIK INFORMATIKA
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/2] Menyiapkan pembukaan browser otomatis ke http://localhost:8501 ...
start "" cmd /c "ping 127.0.0.1 -n 3 >nul & start http://localhost:8501"

echo [2/2] Menjalankan server Streamlit...
echo.
echo ============================================================
echo   LINK AKSES APLIKASI:
echo   - Laptop (Otomatis Terbuka): http://localhost:8501
echo   - HP / Smartphone (Wi-Fi)  : http://localhost:8501 (atau cek IP Network di terminal)
echo ============================================================
echo.
echo * Catatan: Tekan Ctrl+C di terminal ini jika ingin menghentikan aplikasi.
echo.

if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" -m streamlit run app_upload.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py -3.9 -m streamlit run app_upload.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
    ) else (
        python -m streamlit run app_upload.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
    )
)

pause
