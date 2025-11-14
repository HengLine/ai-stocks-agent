@echo off

REM AI Stocks Agent 启动脚本
REM 此脚本将激活虚拟环境并启动交互式界面

echo AI Stocks Agent 启动脚本

REM 检查虚拟环境是否存在
if not exist .venv (
    echo 错误: 未找到虚拟环境。请先运行 install.bat 安装项目。
    pause
    exit /b 1
)

REM 激活虚拟环境
call .venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo 错误: 无法激活虚拟环境。
    pause
    exit /b 1
)

echo 虚拟环境已激活

REM 检查.env文件是否存在
if not exist .env (
    echo 警告: 未找到.env文件。请根据.env.example创建.env文件并配置必要的环境变量。
    echo 程序将继续运行，但某些功能可能无法正常工作。
    echo.
)

REM 设置Streamlit调试模式环境变量
set STREAMLIT_DEBUG=1

REM 启动主程序
python main.py --dashboard --debug
if %ERRORLEVEL% neq 0 (
    echo 错误: 程序运行失败。
    pause
    exit /b 1
)

pause