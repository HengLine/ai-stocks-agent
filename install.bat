@echo off

REM AI Stocks Agent 安装脚本
REM 此脚本将安装项目所需的所有依赖

echo AI Stocks Agent 安装脚本

REM 检查Python是否已安装
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到Python。请先安装Python 3.8或更高版本。
    pause
    exit /b 1
)

echo 正在使用Python: %PYTHON_VERSION%

REM 创建虚拟环境
if not exist .venv (
    echo 创建Python虚拟环境...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo 错误: 无法创建虚拟环境。
        pause
        exit /b 1
    )
)

REM 激活虚拟环境
call .venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo 错误: 无法激活虚拟环境。
    pause
    exit /b 1
)

echo 虚拟环境已激活

REM 更新pip
pip install --upgrade pip

REM 安装项目依赖
if exist requirements.txt (
    echo 安装项目依赖...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo 错误: 安装依赖失败。
        pause
        exit /b 1
    )
)

echo 项目依赖安装完成

REM 提示用户关于环境变量的配置
if not exist .env (
    echo 注意: 未找到.env文件。请根据.env.example创建.env文件并配置必要的环境变量。
)

REM 提示用户项目已安装完成
echo.
echo ================================
echo AI Stocks Agent 安装已完成！
echo ================================
echo 要使用项目，请运行:
if %0 == "%~f0" (
    echo   .venv\Scripts\activate.bat
)

echo  python main.py  --dashboard

echo.
echo 或者直接运行 start.bat 启动交互式界面
echo ================================

pause