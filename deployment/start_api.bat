@echo off
REM Laptop 1 (Windows): start the FastAPI backend. Edit the two paths below or set them beforehand.
cd /d %~dp0\..
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if "%PATHSNAP_EVAL_DATA_ROOT%"=="" set PATHSNAP_EVAL_DATA_ROOT=data\raw\student_test
if "%PATHSNAP_DATASET_LABEL%"=="" set PATHSNAP_DATASET_LABEL=student_test (rehearsal)
python scripts\validate_artifacts.py || exit /b 1
ipconfig | findstr IPv4
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
