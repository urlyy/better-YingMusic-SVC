@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo 请先按 README.md 安装依赖，创建项目的 .venv 环境。
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0app\server.py"
if errorlevel 1 pause
