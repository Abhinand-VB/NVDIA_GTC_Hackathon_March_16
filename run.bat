@echo off
cd /d "%~dp0"
set PYTHON=c:\users\1abhi\appdata\local\programs\python\python39\python.exe
if not exist "%PYTHON%" set PYTHON=python
echo Installing dependencies...
"%PYTHON%" -m pip install -q -r requirements.txt
if errorlevel 1 (echo pip install failed. & pause & exit /b 1)
echo Starting Streamlit...
"%PYTHON%" -m streamlit run app.py
pause
