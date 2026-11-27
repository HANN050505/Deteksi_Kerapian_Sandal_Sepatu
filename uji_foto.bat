@echo off
setlocal
title Smart Shoe Monitor - Uji Foto Cepat
echo ============================================================
echo   SMART SHOE MONITOR - UJI FOTO SANDAL / SEPATU
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

echo Membuka jendela pemilihan foto...
echo Pilih file gambar (.jpg, .jpeg, .png, .webp) pada dialog yang muncul.
echo.

%PY_EXE% predict_image.py

pause
