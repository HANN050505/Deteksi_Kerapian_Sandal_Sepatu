@echo off
setlocal
title Smart Shoe Monitor - Uji Foto Cepat
echo ============================================================
echo   SMART SHOE MONITOR - UJI FOTO SANDAL / SEPATU
echo ============================================================
echo.

cd /d "%~dp0"

echo Membuka jendela pemilihan foto...
echo Pilih file gambar (.jpg, .jpeg, .png, .webp) pada dialog yang muncul.
echo.

if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" predict_image.py
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py -3.9 predict_image.py
    ) else (
        python predict_image.py
    )
)

pause
