@echo off
rem Start ScoreSense: opens the upload page in the browser. Keep this window open while using it.
cd /d "%~dp0eval"
echo ScoreSense is running at http://localhost:8000  (close this window to stop it)
"%~dp0venv\Scripts\python.exe" app.py
pause
