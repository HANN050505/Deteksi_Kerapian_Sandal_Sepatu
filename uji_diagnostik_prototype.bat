@echo off
setlocal
title Smart Shoe Monitor - Diagnostik Mandiri Prototype
echo ============================================================
echo   SMART SHOE MONITOR - DIAGNOSTIK MANDIRI PROTOTYPE
echo ============================================================
echo.

cd /d "%~dp0"

echo Menjalankan uji diagnostik prototype mandiri...
echo.

if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" diagnostik_prototype.py
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py -3.9 diagnostik_prototype.py
    ) else (
        python diagnostik_prototype.py
    )
)

echo.
pause
