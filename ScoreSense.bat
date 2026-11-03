@echo off
rem Mark one test: drag the test folder onto this file, or double-click and paste its path.
cd /d "%~dp0"
set "FOLDER=%~1"
if "%FOLDER%"=="" set /p "FOLDER=Drag the test folder into this window and press Enter: "
set "FOLDER=%FOLDER:"=%"
venv\Scripts\python.exe eval\run.py "%FOLDER%" %2
echo.
pause
