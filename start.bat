@echo off
REM start.bat — One-click launcher for libaix (Windows).
REM
REM Usage:
REM   start.bat              (Install deps, train if needed, start server)
REM   start.bat --port 8080  (Custom port)
REM
REM Drop the libaix folder anywhere and double-click this file.

cd /d "%~dp0"

echo ==========================================
echo          libaix - AI launcher
echo ==========================================
echo.

REM Try python, then python3, then py
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python start.py %*
    goto :end
)

where python3 >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python3 start.py %*
    goto :end
)

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -3 start.py %*
    goto :end
)

echo ERROR: Python 3 not found. Install it from https://www.python.org
echo.
pause

:end
