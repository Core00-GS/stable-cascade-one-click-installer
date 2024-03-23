@echo off
setlocal

REM Capture the current script's directory
set "script_dir=%~dp0"

REM Change to the directory above
cd ..
call venv\Scripts\activate.bat

REM Change back to tests directory
cd tests
python test_forge.py
pause
