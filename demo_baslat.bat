@echo off
rem GraphAML canli demo: iki defe klikleyin. Brauzerde http://127.0.0.1:8090 acilir.
rem Dayandirmaq ucun bu qara pencereni baglayin.
cd /d "%~dp0"
echo Demo baslayir... (ilk acilis 10-15 saniye cekir)
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:8090/"
".venv\Scripts\python.exe" -m uvicorn graphaml.api:app --app-dir src --host 127.0.0.1 --port 8090
pause
