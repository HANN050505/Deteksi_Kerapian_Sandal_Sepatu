@echo off
title Smart Shoe Monitor - Uji Foto Cepat
echo ============================================================
echo   SMART SHOE MONITOR - UJI FOTO SANDAL / SEPATU
echo ============================================================
echo.
echo Membuka jendela pemilihan foto...
echo Pilih file gambar (.jpg, .jpeg, .png) pada dialog yang muncul.
echo.

cd /d "%~dp0"
where python >nul 2>&1
if %errorlevel% equ 0 (
    python predict_image.py
) else (
    "C:\Users\Raihan aryanova\AppData\Local\Programs\Python\Python39\python.exe" predict_image.py
)

pause
