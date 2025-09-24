@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM PDF to Q&A Dataset Generator - Windows Batch Launcher Script
REM ============================================================================
REM
REM This Windows batch script provides comprehensive setup and execution for the
REM PDF to Q&A Dataset Generator ML pipeline. It handles environment detection,
REM dependency management, interactive configuration, and automated execution.
REM
REM FEATURES:
REM - Windows-native Python detection (python/python3/py commands)
REM - Automatic virtual environment management
REM - Interactive AI provider configuration (Ollama/OpenAI)
REM - Model training setup with multiple base model options
REM - Dependency validation and installation
REM - Environment file (.env) management
REM - Pre-flight checks for AI services
REM - Post-execution model testing options
REM
REM MACHINE LEARNING PIPELINE:
REM 1. PDF Text Extraction - Extract content from PDF documents
REM 2. AI-Powered Q&A Generation - Create training datasets using LLMs
REM 3. Dataset Management - Structure and validate training data
REM 4. Model Fine-tuning - Train custom conversational models
REM 5. Model Testing - Interactive validation of trained models
REM 6. Model Deployment - Upload to Hugging Face Hub
REM
REM SUPPORTED ENVIRONMENTS:
REM - Windows 10/11 Command Prompt
REM - Windows PowerShell
REM - Windows Terminal
REM - Cross-platform Python detection
REM
REM USAGE EXAMPLES:
REM   run.bat                           - Interactive setup with defaults
REM   set AI_PROVIDER=openai && run.bat - Override AI provider for this run
REM   set AI_MODEL=gpt-4 && run.bat     - Use specific model temporarily
REM
REM CONFIGURATION OPTIONS:
REM   Edit .env file to set persistent configuration:
REM   - AI_PROVIDER: ollama (local) or openai (cloud)
REM   - AI_MODEL: Specific model name (optional)
REM   - OPENAI_API_KEY: Required for OpenAI provider
REM   - HUGGING_FACE_HUB_TOKEN: Optional for model/dataset upload
REM   - TRAIN_MODEL: Enable/disable model fine-tuning
REM   - BASE_MODEL: Pre-trained model for fine-tuning
REM   - FINETUNE_METHOD: full or lora (Parameter Efficient Fine-Tuning)
REM   - OVERWRITE_DATASET: Force dataset regeneration
REM
REM TECHNICAL REQUIREMENTS:
REM - Python 3.8+ with pip (Windows installation)
REM - For Ollama: Local Ollama server running on port 11434
REM - For OpenAI: Valid API key with sufficient credits
REM - For training: GPU recommended but not required
REM - For uploads: Hugging Face account and token
REM
REM Author: Generated from PDF to Q&A Dataset Generator Pipeline
REM Purpose: Automated ML pipeline setup and execution for Windows
REM ============================================================================

echo ============================================================================
echo PDF to Q&A Dataset Generator - Windows ML Pipeline Launcher
echo ============================================================================
echo Starting automated setup and execution...

REM ============================================================================
REM PYTHON ENVIRONMENT DETECTION FOR WINDOWS
REM ============================================================================
REM Windows-specific Python detection to handle different installation methods:
REM - 'python' - Standard Windows Python installation
REM - 'python3' - Explicit Python 3 command (less common on Windows)
REM - 'py' - Windows Python Launcher (recommended method)
REM This ensures compatibility across different Windows Python setups

echo [1/8] Detecting Python installation...
set PYTHON_CMD=

REM Try 'python' command first (most common on Windows)
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    echo ✓ Found Python via 'python' command
) else (
    REM Try 'python3' command (less common on Windows)
    python3 --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=python3
        echo ✓ Found Python via 'python3' command
    ) else (
        REM Try 'py' command (Windows Python Launcher - recommended)
        py --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON_CMD=py
            echo ✓ Found Python via 'py' command (Windows Launcher)
        ) else (
            echo ❌ [ERROR] Python is not installed or not working.
            echo Please install Python 3.8+ from https://python.org
            echo Make sure to check "Add Python to PATH" during installation
            echo Supported commands: python, python3, py
            pause
            exit /b 1
        )
    )
)

REM Display detected Python version for verification
echo Using Python command: %PYTHON_CMD%
echo Python version:
%PYTHON_CMD% --version

REM ============================================================================
REM VIRTUAL ENVIRONMENT MANAGEMENT FOR WINDOWS
REM ============================================================================
REM Clean virtual environment setup to ensure consistent dependencies
REM This prevents conflicts with system-wide Python packages

echo.
echo [2/8] Setting up virtual environment...

REM Remove existing virtual environment to ensure clean state
REM This prevents issues with corrupted or outdated environments
if exist ".venv" (
    echo 🗑️  Removing existing virtual environment for clean setup...
    rmdir /s /q .venv
)

REM Create new virtual environment
echo 📦 Creating fresh virtual environment...
%PYTHON_CMD% -m venv .venv

REM Validate virtual environment creation
if not exist ".venv" (
    echo ❌ [ERROR] Failed to create virtual environment
    echo This might be due to:
    echo - Insufficient disk space
    echo - Permission issues (try running as Administrator)
    echo - Python venv module not installed
    echo - Antivirus software blocking file creation
    pause
    exit /b 1
)

echo ✓ Virtual environment created successfully

REM ============================================================================
REM VIRTUAL ENVIRONMENT ACTIVATION FOR WINDOWS
REM ============================================================================
REM Windows-specific virtual environment activation
REM Uses the Scripts directory (Windows) instead of bin (Unix)

echo.
echo [3/8] Activating virtual environment...
echo 🪟 Activating Windows virtual environment...

REM Activate virtual environment using Windows-specific path
call .venv\Scripts\activate.bat

echo ✓ Virtual environment activated

REM ============================================================================
REM DEPENDENCY INSTALLATION
REM ============================================================================
REM Install required Python packages for the ML pipeline

echo.
echo [4/8] Installing ML pipeline dependencies...
echo 📥 Installing packages from requirements.txt...

REM Install dependencies with progress indication
REM --upgrade ensures latest compatible versions
pip install --upgrade pip
pip install -r requirements.txt

echo ✓ Dependencies installed successfully

REM ============================================================================
REM ENVIRONMENT CONFIGURATION CHECK
REM ============================================================================
REM Validate and prepare environment configuration

echo.
echo [5/8] Checking environment configuration...

REM Check if .env file exists for persistent configuration
if not exist ".env" (
    echo ⚠️  [WARNING] .env file not found
    echo Creating .env file from template...
    
    REM Create basic .env file if .env.example exists
    if exist ".env.example" (
        copy .env.example .env >nul
        echo ✓ Created .env from .env.example template
    ) else (
        REM Create minimal .env file
        type nul > .env
        echo ✓ Created empty .env file
    )
    
    echo You can manually edit .env file later for persistent configuration
) else (
    echo ✓ Found existing .env configuration file
)

REM ============================================================================
REM INTERACTIVE CONFIGURATION SETUP
REM ============================================================================
REM Guide user through ML pipeline configuration options
REM This ensures proper setup for different use cases and environments

echo.
echo [6/8] Interactive Configuration Setup
echo ============================================================================
echo Configure your ML pipeline settings:
echo.
echo 🤖 1. AI Provider Selection:
echo    1^) Ollama ^(Local LLM - Free, Private, Requires local setup^)
echo    2^) OpenAI ^(Cloud API - Paid, High quality, Easy setup^)
echo.
set /p provider_choice="Choose AI provider (1-2) [1]: "
if "%provider_choice%"=="" set provider_choice=1

REM Configure AI provider based on user selection
if "%provider_choice%"=="2" (
    set AI_PROVIDER=openai
    echo.
    echo 📡 OpenAI Configuration:
    echo Available models:
    echo    - gpt-4o-mini: Fast, cost-effective, good quality
    echo    - gpt-3.5-turbo: Balanced speed and quality
    echo    - gpt-4: Highest quality, slower, more expensive
    echo.
    set /p ai_model="Enter OpenAI model [gpt-4o-mini]: "
    if "!ai_model!"=="" set ai_model=gpt-4o-mini
) else (
    set AI_PROVIDER=ollama
    echo.
    echo 🏠 Ollama Configuration:
    echo Using local Ollama server for private, offline processing
    echo Recommended model: cesarchamal/qa-expert ^(optimized for Q&A generation^)
    echo.
    set /p ai_model="Enter Ollama model [cesarchamal/qa-expert]: "
    if "!ai_model!"=="" set ai_model=cesarchamal/qa-expert
)

echo.
echo 📊 2. Dataset Management:
echo    1^) Use existing dataset if found ^(Faster, reuse previous work^)
echo    2^) Always overwrite existing dataset ^(Fresh generation, slower^)
echo.
set /p dataset_choice="Choose dataset option (1-2) [1]: "
if "%dataset_choice%"=="" set dataset_choice=1

REM Set dataset overwrite behavior
if "%dataset_choice%"=="2" (
    set OVERWRITE_DATASET=true
    echo ✓ Will regenerate dataset from scratch
) else (
    set OVERWRITE_DATASET=false
    echo ✓ Will reuse existing dataset if available
)

echo.
echo 🧠 3. Model Training Configuration:
echo    1^) Skip model training ^(Dataset generation only^)
echo    2^) Train custom model after dataset creation ^(Full ML pipeline^)
echo.
echo Note: Training requires significant computational resources and time
set /p train_choice="Choose training option (1-2) [1]: "
if "%train_choice%"=="" set train_choice=1

REM Configure model training if selected
if "%train_choice%"=="2" (
    set TRAIN_MODEL=true
    echo ✓ Model training enabled
    
    echo.
    echo 🔧 Fine-tuning Method Selection:
    echo    1^) Full fine-tuning ^(Updates all model parameters^)
    echo       - Best quality results
    echo       - Requires more memory and time
    echo       - Recommended for production use
    echo.
    echo    2^) LoRA fine-tuning ^(Parameter Efficient Fine-Tuning^)
    echo       - Faster training with less memory
    echo       - Good quality with efficiency
    echo       - Recommended for experimentation
    echo.
    set /p finetune_method="Choose fine-tuning method (1-2) [1]: "
    if "!finetune_method!"=="" set finetune_method=1
    
    if "!finetune_method!"=="2" (
        set FINETUNE_METHOD=lora
        echo ✓ Using LoRA ^(Parameter Efficient Fine-Tuning^)
    ) else (
        set FINETUNE_METHOD=full
        echo ✓ Using full fine-tuning
    )
    
    echo.
    echo 🏗️  Base Model Selection:
    echo Choose the pre-trained model to fine-tune:
    echo.
    echo    1^) microsoft/DialoGPT-small ^(117M params^)
    echo       - Fast training and inference
    echo       - Lower memory requirements
    echo       - Good for prototyping
    echo.
    echo    2^) microsoft/DialoGPT-medium ^(345M params^) [RECOMMENDED]
    echo       - Balanced performance and quality
    echo       - Moderate resource requirements
    echo       - Best overall choice
    echo.
    echo    3^) microsoft/DialoGPT-large ^(762M params^)
    echo       - Highest quality responses
    echo       - Requires significant resources
    echo       - Best for production deployment
    echo.
    echo    4^) distilgpt2 ^(82M params^)
    echo       - Very fast, minimal resources
    echo       - Basic conversational ability
    echo.
    echo    5^) gpt2 ^(124M params^)
    echo       - Standard GPT-2 model
    echo       - General purpose
    echo.
    echo    6^) Custom model ^(Enter your own^)
    echo.
    set /p model_choice="Choose base model (1-6) [2]: "
    if "!model_choice!"=="" set model_choice=2
    
    REM Set base model based on user selection
    if "!model_choice!"=="1" (
        set base_model=microsoft/DialoGPT-small
        echo ✓ Selected DialoGPT-small ^(fast, lightweight^)
    ) else if "!model_choice!"=="2" (
        set base_model=microsoft/DialoGPT-medium
        echo ✓ Selected DialoGPT-medium ^(recommended balance^)
    ) else if "!model_choice!"=="3" (
        set base_model=microsoft/DialoGPT-large
        echo ✓ Selected DialoGPT-large ^(highest quality^)
    ) else if "!model_choice!"=="4" (
        set base_model=distilgpt2
        echo ✓ Selected DistilGPT-2 ^(very fast^)
    ) else if "!model_choice!"=="5" (
        set base_model=gpt2
        echo ✓ Selected GPT-2 ^(standard^)
    ) else if "!model_choice!"=="6" (
        echo.
        echo Enter custom model name ^(e.g., 'microsoft/DialoGPT-medium'^):
        set /p base_model="Custom model: "
        echo ✓ Selected custom model: !base_model!
    ) else (
        set base_model=microsoft/DialoGPT-medium
        echo ✓ Using default: DialoGPT-medium
    )
) else (
    set TRAIN_MODEL=false
    set FINETUNE_METHOD=full
    set base_model=microsoft/DialoGPT-medium
    echo ✓ Model training disabled - dataset generation only
)

REM ============================================================================
REM ENVIRONMENT FILE CONFIGURATION
REM ============================================================================
REM Update .env file with user selections for persistent configuration

echo.
echo [7/8] Updating configuration file...
echo 💾 Saving settings to .env file for future use...

REM Update .env file using PowerShell for reliable text replacement
powershell -Command "(Get-Content .env -ErrorAction SilentlyContinue) -replace '^AI_PROVIDER=.*', 'AI_PROVIDER=%AI_PROVIDER%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^AI_MODEL=.*', 'AI_MODEL=%ai_model%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^OVERWRITE_DATASET=.*', 'OVERWRITE_DATASET=%OVERWRITE_DATASET%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^TRAIN_MODEL=.*', 'TRAIN_MODEL=%TRAIN_MODEL%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^FINETUNE_METHOD=.*', 'FINETUNE_METHOD=%FINETUNE_METHOD%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace '^BASE_MODEL=.*', 'BASE_MODEL=%base_model%' | Set-Content .env"

REM Display final configuration summary
echo.
echo ============================================================================
echo 📋 CONFIGURATION SUMMARY
echo ============================================================================
echo 🤖 AI Provider: %AI_PROVIDER%
echo 🧠 AI Model: %ai_model%
echo 📊 Overwrite Dataset: %OVERWRITE_DATASET%
echo 🏋️  Train Model: %TRAIN_MODEL%
if "%TRAIN_MODEL%"=="true" (
    echo 🏗️  Base Model: %base_model%
    echo 🔧 Fine-tuning Method: %FINETUNE_METHOD%
)
echo ============================================================================

REM ============================================================================
REM PRE-FLIGHT VALIDATION
REM ============================================================================
REM Validate required files and services before starting the ML pipeline

echo.
echo [8/8] Performing pre-flight checks...

REM Check if source PDF file exists
echo 📄 Checking for PDF input file...
if not exist "jvm_troubleshooting_guide.pdf" (
    echo ❌ [ERROR] PDF file 'jvm_troubleshooting_guide.pdf' not found
    echo.
    echo Required: Place your PDF file in the current directory with the name:
    echo          'jvm_troubleshooting_guide.pdf'
    echo.
    echo The PDF should contain the content you want to convert into Q&A format.
    pause
    exit /b 1
)
echo ✓ Found PDF input file

REM Validate AI provider setup and connectivity
if "%AI_PROVIDER%"=="ollama" (
    echo 🏠 Validating Ollama setup...
    
    REM Check if Ollama server is running
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo ❌ [ERROR] Ollama server is not running on localhost:11434
        echo.
        echo To fix this:
        echo 1. Install Ollama from https://ollama.ai
        echo 2. Start the server: ollama serve
        echo 3. Run this script again
        echo.
        echo Ollama provides local, private AI processing without API costs.
        pause
        exit /b 1
    )
    echo ✓ Ollama server is running
    
    REM Check if required model is available
    echo 🔍 Checking for required model...
    curl -s http://localhost:11434/api/tags | findstr "%ai_model%" >nul 2>&1
    if errorlevel 1 (
        echo ⬇️  Model '%ai_model%' not found locally. Downloading...
        echo This may take several minutes depending on model size.
        ollama pull "%ai_model%"
        echo ✓ Model downloaded successfully
    ) else (
        echo ✓ Model '%ai_model%' is available
    )
) else if "%AI_PROVIDER%"=="openai" (
    echo 📡 Validating OpenAI setup...
    
    REM Check if API key is configured
    for /f "tokens=2 delims==" %%i in ('findstr "^OPENAI_API_KEY=" .env 2^>nul') do set OPENAI_KEY=%%i
    if "%OPENAI_KEY%"=="" (
        echo ❌ [ERROR] OpenAI API key not configured
        echo.
        echo To fix this:
        echo 1. Get an API key from https://platform.openai.com/api-keys
        echo 2. Add it to your .env file: OPENAI_API_KEY=sk-your-key-here
        echo 3. Ensure you have sufficient credits in your OpenAI account
        echo.
        echo Note: OpenAI charges per API call. Check pricing at https://openai.com/pricing
        pause
        exit /b 1
    )
    if "%OPENAI_KEY%"=="your_openai_key_here" (
        echo ❌ [ERROR] Please update OPENAI_API_KEY in your .env file
        echo Replace 'your_openai_key_here' with your actual API key
        pause
        exit /b 1
    )
    echo ✓ OpenAI API key configured
    echo ✓ Using model: %ai_model%
) else (
    echo ❌ [ERROR] Unsupported AI provider: %AI_PROVIDER%
    echo Supported providers: 'ollama' ^(local^) or 'openai' ^(cloud^)
    echo Please check your .env configuration.
    pause
    exit /b 1
)

REM Pre-download base model for training if enabled
if "%TRAIN_MODEL%"=="true" (
    echo.
    echo 🏗️  Preparing base model for fine-tuning...
    echo 📥 Pre-downloading: %base_model%
    echo This ensures the model is available before training starts.
    
    REM Download tokenizer and model to local cache
    %PYTHON_CMD% -c "from transformers import AutoTokenizer, AutoModelForCausalLM; print('Downloading tokenizer...'); AutoTokenizer.from_pretrained('%base_model%'); print('Downloading model...'); AutoModelForCausalLM.from_pretrained('%base_model%'); print('✓ Base model ready for fine-tuning')"
)

REM ============================================================================
REM ML PIPELINE EXECUTION
REM ============================================================================
REM Execute the main ML pipeline with validated configuration

echo.
echo ============================================================================
echo 🚀 STARTING ML PIPELINE EXECUTION
echo ============================================================================
echo ✅ All pre-flight checks passed
echo 🔄 Executing PDF to Q&A Dataset Generator pipeline...
echo.

REM Execute the main Python script with all configuration ready
%PYTHON_CMD% main.py

echo.
echo ============================================================================
echo 🎉 ML PIPELINE COMPLETED SUCCESSFULLY!
echo ============================================================================

REM ============================================================================
REM POST-EXECUTION MODEL TESTING
REM ============================================================================
REM Offer interactive model testing if a trained model is available

echo.
echo 🧪 Post-Execution Options:

REM Check if trained model exists and offer testing
if exist "models\jvm_troubleshooting_model" (
    echo ✅ Trained model found at: models\jvm_troubleshooting_model
    echo.
    echo Available testing options:
    echo 1. Interactive testing with conversation memory ^(test_model.py^)
    echo 2. Quick batch testing ^(quick_test.py^)
    echo 3. Skip testing
    echo.
    set /p test_choice="Choose testing option (1-3) [3]: "
    if "!test_choice!"=="" set test_choice=3
    
    if "!test_choice!"=="1" (
        echo 🚀 Starting interactive model testing...
        %PYTHON_CMD% test_model.py
    ) else if "!test_choice!"=="2" (
        echo 🚀 Running quick batch validation...
        %PYTHON_CMD% quick_test.py
    ) else (
        echo ✓ Skipping model testing
        echo You can test the model later using:
        echo   python test_model.py              # Interactive testing with memory
        echo   python quick_test.py              # Quick validation
    )
) else if "%TRAIN_MODEL%"=="true" (
    echo ⚠️  [WARNING] Model training was enabled but no model found
    echo Check the training logs above for any errors.
    echo You can try running 'python model_utils.py recover' to restore the model.
) else (
    echo ℹ️  No model training was performed ^(dataset generation only^)
    echo To train a model, run this script again and select training option.
)

echo.
echo ============================================================================
echo 📚 NEXT STEPS:
echo ============================================================================
echo • Dataset: Check .\dataset\jvm_troubleshooting_guide\ for generated Q&A pairs
if "%TRAIN_MODEL%"=="true" (
    echo • Model: Check .\models\jvm_troubleshooting_model\ for trained model files
)
echo • Testing: Use test_model.py for interactive testing with conversation memory
echo • Upload: Models and datasets can be uploaded to Hugging Face Hub
echo • Documentation: See README.md for detailed usage instructions
echo ============================================================================
echo Thank you for using the PDF to Q&A Dataset Generator! 🎯

pause