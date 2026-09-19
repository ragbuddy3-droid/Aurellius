@echo off
setlocal

cd /d "%~dp0"

if not exist "rag.index" if not exist "ipas.index" (
    echo FAISS index belum ditemukan.
    echo Jalankan dulu: python build_index.py
    pause
    exit /b 1
)

if not exist "rag_store.json" if not exist "ipas_store.json" (
    echo Metadata store belum ditemukan.
    echo Jalankan dulu: python build_index.py
    pause
    exit /b 1
)

if "%GEMINI_API_KEY%"=="" (
    echo GEMINI_API_KEY belum diset.
    echo Set dulu environment variable GEMINI_API_KEY sebelum menjalankan chatbot.
    pause
    exit /b 1
)

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" chatbot.py
    goto :end
)

where py >nul 2>nul
if %errorlevel%==0 (
    py chatbot.py
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    python chatbot.py
    goto :end
)

echo Python tidak ditemukan.
echo Install Python atau aktifkan virtual environment terlebih dahulu.

:end
pause
endlocal
