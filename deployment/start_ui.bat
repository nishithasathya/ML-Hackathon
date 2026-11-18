@echo off
REM Laptop 2 (Windows): start the UI. Usage: deployment\start_ui.bat http://<laptop-1-ip>:8000
cd /d %~dp0\..
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
if not "%1"=="" set PATHSNAP_API_URL=%1
if "%PATHSNAP_API_URL%"=="" (echo Usage: start_ui.bat http://LAPTOP1_IP:8000 & exit /b 1)
python deployment\check_connection.py %PATHSNAP_API_URL%
streamlit run frontend\app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
