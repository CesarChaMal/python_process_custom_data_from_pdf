@echo off
setlocal enabledelayedexpansion
REM PDF to Q&A Dataset Generator Launcher for Windows
REM 
REM Usage Examples:
REM   run.bat                           - Use default settings (Ollama)
REM   set AI_PROVIDER=openai && run.bat - Use OpenAI for this run
REM   set AI_MODEL=gpt-4 && run.bat     - Use specific model
REM
REM Configuration:
REM   Edit .env file to set:
REM   - AI_PROVIDER (ollama/openai)
REM   - AI_MODEL (optional)
REM   - OPENAI_API_KEY (for OpenAI)
REM   - HUGGING_FACE_HUB_TOKEN (optional)
REM   - TRAIN_MODEL (true/false)
REM   - BASE_MODEL (for fine-tuning)

echo [INFO] Starting PDF to Q&A Dataset Generator...

REM Check if Python is installed (try multiple variants)
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
) else (
    python3 --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=python3
    ) else (
        py --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON_CMD=py
        ) else (
            echo ❌ Python is not installed. Please install Python 3.8+ first.
            pause
            exit /b 1
        )
    )
)

echo [INFO] Using Python: %PYTHON_CMD%

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if .env exists
if not exist ".env" (
    echo [WARNING] .env file not found. Please create one with your configuration
    echo You can copy from .env.example and update the values.
)

REM Interactive configuration
echo [INFO] Configuration Setup
echo 1. Select AI Provider:
echo    1^) Ollama ^(default^)
echo    2^) OpenAI
set /p provider_choice="Choose provider (1-2) [1]: "
if "%provider_choice%"=="" set provider_choice=1

if "%provider_choice%"=="2" (
    set AI_PROVIDER=openai
    echo Enter OpenAI model ^(gpt-4o-mini, gpt-3.5-turbo, gpt-4^) [gpt-4o-mini]:
    set /p ai_model="Model: "
    if "!ai_model!"=="" set ai_model=gpt-4o-mini
) else (
    set AI_PROVIDER=ollama
    echo Enter Ollama model [cesarchamal/qa-expert]:
    set /p ai_model="Model: "
    if "!ai_model!"=="" set ai_model=cesarchamal/qa-expert
)

echo 2. Dataset handling:
echo    1^) Use existing dataset if found ^(default^)
echo    2^) Always overwrite existing dataset
set /p dataset_choice="Choose option (1-2) [1]: "
if "%dataset_choice%"=="" set dataset_choice=1

if "%dataset_choice%"=="2" (
    set OVERWRITE_DATASET=true
) else (
    set OVERWRITE_DATASET=false
)

echo 3. Model training:
echo    1^) Skip model training ^(default^)
echo    2^) Train model after dataset creation
set /p train_choice="Choose option (1-2) [1]: "
if "%train_choice%"=="" set train_choice=1

if "%train_choice%"=="2" (
    set TRAIN_MODEL=true
    echo Select base model:
    echo    1^) microsoft/DialoGPT-small ^(fast, lightweight^)
    echo    2^) microsoft/DialoGPT-medium ^(balanced^)
    echo    3^) microsoft/DialoGPT-large ^(best quality, slow^)
    echo    4^) distilgpt2 ^(very fast, basic^)
    echo    5^) gpt2 ^(standard^)
    echo    6^) Custom model
    set /p model_choice="Choose model (1-6) [1]: "
    if "!model_choice!"=="" set model_choice=1
    
    if "!model_choice!"=="1" set base_model=microsoft/DialoGPT-small
    if "!model_choice!"=="2" set base_model=microsoft/DialoGPT-medium
    if "!model_choice!"=="3" set base_model=microsoft/DialoGPT-large
    if "!model_choice!"=="4" set base_model=distilgpt2
    if "!model_choice!"=="5" set base_model=gpt2
    if "!model_choice!"=="6" (
        echo Enter custom model name:
        set /p base_model="Model: "
    )
) else (
    set TRAIN_MODEL=false
    set base_model=microsoft/DialoGPT-small
)

REM Update .env file
echo [INFO] Updating .env configuration...
powershell -Command "(Get-Content .env) -replace '^AI_PROVIDER=.*', 'AI_PROVIDER=%AI_PROVIDER%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^AI_MODEL=.*', 'AI_MODEL=%ai_model%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^OVERWRITE_DATASET=.*', 'OVERWRITE_DATASET=%OVERWRITE_DATASET%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^TRAIN_MODEL=.*', 'TRAIN_MODEL=%TRAIN_MODEL%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^BASE_MODEL=.*', 'BASE_MODEL=%base_model%' | Set-Content .env"

echo [INFO] Using AI provider: %AI_PROVIDER%
echo [INFO] Model: %ai_model%
echo [INFO] Overwrite dataset: %OVERWRITE_DATASET%
echo [INFO] Train model: %TRAIN_MODEL%
if "%TRAIN_MODEL%"=="true" (
    echo [INFO] Base model: %base_model%
)

REM Check if PDF exists
if not exist "jvm_troubleshooting_guide.pdf" (
    echo ⚠️  PDF file 'jvm_troubleshooting_guide.pdf' not found in current directory
    echo Please place your PDF file with this name to continue.
    pause
    exit /b 1
)

REM Check AI provider requirements
if "%AI_PROVIDER%"=="ollama" (
    echo [INFO] Checking Ollama connection...
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Ollama is not running on localhost:11434
        echo Please start Ollama first: ollama serve
        pause
        exit /b 1
    )
    
    echo [INFO] Checking available models...
    curl -s http://localhost:11434/api/tags | findstr "cesarchamal/qa-expert" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Model 'cesarchamal/qa-expert' not found. Pulling model...
        ollama pull cesarchamal/qa-expert
    )
) else if "%AI_PROVIDER%"=="openai" (
    echo [INFO] Checking OpenAI configuration...
    for /f "tokens=2 delims==" %%i in ('findstr "^OPENAI_API_KEY=" .env 2^>nul') do set OPENAI_KEY=%%i
    if "%OPENAI_KEY%"=="" (
        echo [ERROR] OpenAI API key not configured in .env file
        echo Please set OPENAI_API_KEY in your .env file
        pause
        exit /b 1
    )
    if "%OPENAI_KEY%"=="your_openai_key_here" (
        echo [ERROR] Please update OPENAI_API_KEY in your .env file
        pause
        exit /b 1
    )
    echo [SUCCESS] OpenAI configuration found
) else (
    echo [ERROR] Unsupported AI provider: %AI_PROVIDER%
    echo Please set AI_PROVIDER to 'ollama' or 'openai' in .env file
    pause
    exit /b 1
)

REM Pre-download base model if training is enabled
if "%TRAIN_MODEL%"=="true" (
    echo [INFO] Pre-downloading base model: %base_model%
    %PYTHON_CMD% -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('%base_model%'); AutoModelForCausalLM.from_pretrained('%base_model%'); print('[SUCCESS] Base model downloaded')"
)

echo [SUCCESS] All checks passed. Starting PDF processing...
%PYTHON_CMD% main.py

echo [SUCCESS] Process completed successfully!

REM Check if model exists and offer testing
if exist "models\jvm_troubleshooting_model" (
    echo.
    set /p test_model="Do you want to test the trained model? (y/N): "
    if /i "!test_model!"=="y" (
        echo [INFO] Starting model testing...
        %PYTHON_CMD% test_model.py
    )
) else if "%TRAIN_MODEL%"=="true" (
    echo [WARNING] Model training was enabled but no model found at models\jvm_troubleshooting_model
)

pause