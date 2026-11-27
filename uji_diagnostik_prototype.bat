@echo off
setlocal
title Smart Shoe Monitor - Diagnostik Mandiri Prototype
echo ============================================================
echo   SMART SHOE MONITOR - DIAGNOSTIK MANDIRI PROTOTYPE
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

echo Menjalankan uji diagnostik prototype mandiri...
echo.

%PY_EXE% diagnostik_prototype.py

echo.
pause
