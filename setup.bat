@echo off
rem One-time setup on a new Windows computer: Python packages and the AI model.
rem Needs Python 3.11+ and Ollama installed first (see README.md).
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH". & pause & exit /b 1)
where ollama >nul 2>nul || (echo Ollama is not installed. Install it from https://ollama.com/download and run this again. & pause & exit /b 1)
if not exist venv\Scripts\python.exe (
  echo Creating the Python environment...
  python -m venv venv || (pause & exit /b 1)
)
echo Installing packages...
venv\Scripts\python.exe -m pip install --upgrade pip >nul
venv\Scripts\python.exe -m pip install -r eval\requirements.txt || (pause & exit /b 1)
echo Downloading the AI model (about 6 GB, only the first time)...
ollama pull qwen2.5vl:7b || (pause & exit /b 1)
echo.
echo Setup finished. Double-click ScoreSense_App.bat to start ScoreSense.
pause
