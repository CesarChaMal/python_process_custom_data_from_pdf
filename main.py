#!/usr/bin/env python3
"""
PDF to Q&A Dataset Generator - Main Processing Pipeline

This script converts PDF documents into question-answer datasets and optionally
trains custom conversational AI models. It demonstrates end-to-end ML workflow
from data extraction to model deployment.

Key Features:
- PDF text extraction using PyMuPDF
- AI-powered Q&A generation (Ollama/OpenAI)
- Dataset creation with train/test splits
- Model fine-tuning (full or LoRA)
- Hugging Face Hub integration

Author: CesarChaMal
License: MIT
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import logging
import fitz  # PyMuPDF for PDF processing
import ollama  # Local AI model interface
from openai import OpenAI  # OpenAI API client
from dotenv import load_dotenv  # Environment variable management
from datasets import Dataset, DatasetDict, load_from_disk  # HuggingFace datasets
from tqdm import tqdm  # Progress bars
from huggingface_hub import HfApi, create_repo, delete_repo  # HF Hub operations
from huggingface_hub.utils import HfHubHTTPError  # HF error handling
from transformers import (  # Transformer models and training
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
import torch  # PyTorch deep learning framework

# Optional PEFT (Parameter Efficient Fine-Tuning) support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =============================================================================
# PDF PROCESSING FUNCTIONS
# =============================================================================

def read_pdf_content(pdf_path: str) -> list[str]:
    """
    Extract text content from PDF file using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list[str]: List of text content from each page
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be processed
    """
    content_list = []
    
    try:
        # Open PDF document using PyMuPDF (fitz)
        with fitz.open(pdf_path) as doc:
            logging.info(f"Processing PDF with {len(doc)} pages")
            
            # Extract text from each page
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                
                # Only add non-empty pages
                if page_text.strip():
                    content_list.append(page_text)
                    logging.debug(f"Extracted {len(page_text)} characters from page {page_num + 1}")
                else:
                    logging.warning(f"Page {page_num + 1} is empty or contains no text")
                    
    except Exception as e:
        logging.error(f"Failed to process PDF {pdf_path}: {e}")
        raise
        
    logging.info(f"Successfully extracted content from {len(content_list)} pages")
    return content_list

# =============================================================================
# AI PROVIDER INTERFACE FUNCTIONS
# =============================================================================

def call_ai(query: str, provider: str = "ollama", model: str = None) -> str:
    """
    Universal AI provider interface for generating responses.
    
    Args:
        query (str): The prompt/question to send to AI
        provider (str): AI provider ('ollama' or 'openai')
        model (str): Specific model name (optional)
        
    Returns:
        str: AI-generated response
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider == "ollama":
        # Use local Ollama server
        model = model or "cesarchamal/qa-expert"  # Default Q&A optimized model
        return call_ollama(query, model)
    elif provider == "openai":
        # Use OpenAI cloud API
        model = model or "gpt-4o-mini"  # Default cost-effective model
        return call_openai(query, model)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}. Use 'ollama' or 'openai'")

def call_ollama(query: str, model: str) -> str:
    """
    Call local Ollama server for AI response generation.
    
    Args:
        query (str): The prompt to send
        model (str): Ollama model name
        
    Returns:
        str: Generated response or error message
    """
    logging.debug(f"Calling Ollama with model '{model}' for query length: {len(query)}")
    
    # Format messages for chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in creating educational Q&A content."},
        {"role": "user", "content": query},
    ]
    
    try:
        # Call Ollama API (assumes server running on localhost:11434)
        response = ollama.chat(model=model, messages=messages, stream=False)
        
        # Extract response content
        if 'message' in response and 'content' in response['message']:
            content = response['message']['content']
            logging.debug(f"Ollama response length: {len(content)}")
            return content
        else:
            logging.error(f"Unexpected Ollama response structure: {response}")
            return "No response or unexpected response structure."
            
    except Exception as e:
        logging.error(f"Ollama API error: {str(e)}")
        return f"Error occurred while calling Ollama API: {str(e)}"

def call_openai(query: str, model: str) -> str:
    """
    Call OpenAI API for AI response generation.
    
    Args:
        query (str): The prompt to send
        model (str): OpenAI model name (gpt-4o-mini, gpt-3.5-turbo, gpt-4)
        
    Returns:
        str: Generated response or error message
    """
    logging.debug(f"Calling OpenAI with model '{model}' for query length: {len(query)}")
    
    try:
        # Initialize OpenAI client with API key from environment
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create chat completion
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in creating educational Q&A content."},
                {"role": "user", "content": query}
            ],
            temperature=0.7,  # Balanced creativity vs consistency
            max_tokens=1000   # Reasonable response length
        )
        
        content = response.choices[0].message.content
        logging.debug(f"OpenAI response length: {len(content)}")
        return content
        
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        return f"Error occurred while calling OpenAI API: {str(e)}"

# =============================================================================
# Q&A GENERATION FUNCTIONS
# =============================================================================

def prompt_engineered_api(text: str, provider: str = "ollama", model: str = None) -> str:
    """
    Generate question-answer pairs from text content using AI.
    
    This function uses prompt engineering to instruct the AI to create
    educational Q&A content in a specific format for training conversational models.
    
    Args:
        text (str): Source text content to generate Q&A from
        provider (str): AI provider to use
        model (str): Specific model name
        
    Returns:
        str: Generated Q&A in format "### Human: ... ### Assistant: ..."
    """
    # Carefully crafted prompt for consistent Q&A generation
    prompt = f"""
    Based on the following technical content, create a high-quality question-answer pair that would be useful for training a troubleshooting assistant.

    Content: {text}

    Requirements:
    1. Generate ONE clear, specific question that someone might ask about this topic
    2. Provide a comprehensive, helpful answer
    3. Use EXACTLY this format:

    ### Human:
    [Your question here]
    ### Assistant:
    [Your detailed answer here]

    Focus on practical, actionable information that would help someone solve real problems.
    """
    
    return call_ai(prompt, provider, model)

# =============================================================================
# MODEL TRAINING AND FINE-TUNING FUNCTIONS
# =============================================================================

def train_and_upload_model(dataset_dict: DatasetDict, auth_token: str, username: str) -> None:
    """
    Train a custom conversational model using the generated Q&A dataset.
    
    This function implements both full fine-tuning and LoRA (Low-Rank Adaptation)
    methods for parameter-efficient training. It handles the complete training
    pipeline from data preprocessing to model upload.
    
    Args:
        dataset_dict (DatasetDict): Training and test datasets
        auth_token (str): Hugging Face authentication token
        username (str): Hugging Face username for model upload
    """
    print("[INFO] Starting model fine-tuning...")
    
    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    
    # Get configuration from environment variables with sensible defaults
    base_model = os.getenv('BASE_MODEL')
    if not base_model:
        # Auto-select based on available hardware
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            available = total_mem - reserved
            
            print(f"\n[INFO] GPU Memory Analysis:")
            print(f"  Total: {total_mem:.1f} GB")
            print(f"  Available: {available:.1f} GB")
            
            if available >= 10:
                print("\n[INFO] Model Selection (GPU):")
                print("  1. DialoGPT-small (117M params) - Fast, 2-4GB VRAM")
                print("  2. DialoGPT-medium (345M params) - Balanced, 4-6GB VRAM")
                print("  3. DialoGPT-large (774M params) - Best quality, 8-12GB VRAM")
                
                import sys
                if sys.stdin.isatty():
                    for attempt in range(3):
                        try:
                            choice = input("\nSelect model (1, 2, or 3) [default: 2]: ").strip()
                            if not choice:  # Default to medium
                                choice = '2'
                            if choice == '1':
                                base_model = 'microsoft/DialoGPT-small'
                                break
                            elif choice == '2':
                                base_model = 'microsoft/DialoGPT-medium'
                                break
                            elif choice == '3':
                                base_model = 'microsoft/DialoGPT-large'
                                break
                            else:
                                print("Please enter 1, 2, or 3")
                        except (EOFError, KeyboardInterrupt):
                            break
                
                if not base_model:
                    base_model = 'microsoft/DialoGPT-medium'
                    print("[INFO] Using default: DialoGPT-medium")
            else:
                base_model = 'microsoft/DialoGPT-small'
                print(f"[INFO] Auto-selected DialoGPT-small (limited GPU memory: {available:.1f}GB)")
        else:
            print("\n[INFO] CPU Mode Detected")
            print("\n[INFO] Model Selection (CPU):")
            print("  1. DialoGPT-small (117M params) - Recommended for CPU")
            print("  2. DialoGPT-medium (345M params) - Slower on CPU")
            
            import sys
            if sys.stdin.isatty():
                for attempt in range(3):
                    try:
                        choice = input("\nSelect model (1 or 2) [default: 1]: ").strip()
                        if not choice:  # Default to small
                            choice = '1'
                        if choice == '1':
                            base_model = 'microsoft/DialoGPT-small'
                            break
                        elif choice == '2':
                            base_model = 'microsoft/DialoGPT-medium'
                            print("[WARNING] Medium model will be slow on CPU")
                            break
                        else:
                            print("Please enter 1 or 2")
                    except (EOFError, KeyboardInterrupt):
                        break
            
            if not base_model:
                base_model = 'microsoft/DialoGPT-small'
                print("[INFO] Using default: DialoGPT-small")
    
    finetune_method = os.getenv('FINETUNE_METHOD', 'full')  # Full fine-tuning by default
    model_name = "jvm_troubleshooting_model"
    model_id = f"{username}/{model_name}"
    
    print(f"[INFO] Base model: {base_model}")
    print(f"[INFO] Fine-tuning method: {finetune_method}")
    print(f"[INFO] Target model ID: {model_id}")
    
    # Create local model directory
    model_dir = f"./models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # =============================================================================
    # GPU MEMORY OPTIMIZATION
    # =============================================================================
    
    if torch.cuda.is_available():
        # Aggressive GPU memory clearing for RTX 4090
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Force garbage collection
        import gc
        gc.collect()
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = total_mem - reserved
        
        print(f"[INFO] GPU Total: {total_mem:.1f} GB")
        print(f"[INFO] GPU Allocated: {allocated:.1f} GB")
        print(f"[INFO] GPU Reserved: {reserved:.1f} GB")
        print(f"[INFO] GPU Available: {free:.1f} GB")
    
    # =============================================================================
    # MODEL AND TOKENIZER LOADING
    # =============================================================================
    
    print("[INFO] Loading base model and tokenizer...")
    
    try:
        # Load tokenizer with proper configuration
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Set padding token (required for batch processing)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load base model with hardware-specific optimization
        if torch.cuda.is_available():
            # GPU configuration
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "6GB"}  # Very conservative for large model
            )
        else:
            # CPU configuration - optimized for memory efficiency
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_cache=False  # Reduce memory usage on CPU
            )
        
        print(f"[SUCCESS] Loaded model with {model.num_parameters():,} parameters")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Failed to load base model: {error_msg}")
        
        # Handle model loading errors with comprehensive options
        if "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
            print("\n[INFO] GPU Memory Issue During Model Loading!")
            
            # Show current memory status
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                available = total - reserved
                print(f"[INFO] GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {available:.1f}GB available of {total:.1f}GB total")
            
            def clear_gpu_memory():
                """Clear GPU memory and delete model"""
                nonlocal model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    import gc
                    gc.collect()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            def reload_model_with_config(config):
                """Reload model with specific configuration"""
                nonlocal model
                clear_gpu_memory()
                model = AutoModelForCausalLM.from_pretrained(base_model, **config)
                return model
            
            import sys
            if sys.stdin.isatty():
                # Temporarily disable debug logging
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)
                sys.stdout.flush()
                sys.stderr.flush()
                
                while True:
                    print("\n📋 Model Loading Failed - Available Options:")
                    print("  1. Try smaller memory limit (2GB)")
                    print("  2. Switch to CPU loading")
                    print("  3. Run GPU cleanup utility")
                    print("  4. Change base model (go back to model selection)")
                    print("  5. Change training configuration (go back to training mode)")
                    print("  6. Skip training and test existing model")
                    print("  7. Exit and manually free GPU memory")
                    sys.stdout.flush()
                    
                    try:
                        choice = input("\nSelect option (1-7) [4]: ").strip() or '4'
                        
                        if choice == '1':
                            print("[INFO] Applying smaller memory limit (2GB)...")
                            try:
                                model = reload_model_with_config({
                                    'torch_dtype': torch.float16,
                                    'device_map': "auto",
                                    'low_cpu_mem_usage': True,
                                    'max_memory': {0: "2GB"}
                                })
                                print("[SUCCESS] Model loaded with 2GB memory limit")
                                break
                            except Exception as retry_e:
                                print(f"[ERROR] 2GB limit failed: {retry_e}")
                                continue
                        elif choice == '2':
                            print("[INFO] Switching to CPU loading...")
                            try:
                                model = reload_model_with_config({
                                    'torch_dtype': torch.float32,
                                    'low_cpu_mem_usage': True,
                                    'use_cache': False
                                })
                                model = model.cpu()
                                print("[SUCCESS] Model loaded on CPU")
                                break
                            except Exception as retry_e:
                                print(f"[ERROR] CPU loading failed: {retry_e}")
                                continue
                        elif choice == '3':
                            print("[INFO] Running GPU cleanup utility...")
                            import subprocess
                            try:
                                subprocess.run(["python", "gpu_cleanup.py"], check=True)
                                print("[INFO] GPU cleanup completed. Retrying model loading...")
                                continue
                            except Exception as cleanup_e:
                                print(f"[ERROR] GPU cleanup failed: {cleanup_e}")
                                print("[TIP] Run manually: python gpu_cleanup.py")
                                continue
                        elif choice == '4':
                            print("[INFO] Returning to base model selection...")
                            return train_and_upload_model(dataset_dict, auth_token, username)
                        elif choice == '5':
                            print("[INFO] Returning to training configuration...")
                            if 'TRAINING_MODE' in os.environ:
                                del os.environ['TRAINING_MODE']
                            return train_and_upload_model(dataset_dict, auth_token, username)
                        elif choice == '6':
                            print("[INFO] Skipping training and proceeding to model testing...")
                            model_dir = "./models/jvm_troubleshooting_model"
                            if os.path.exists(model_dir):
                                print(f"[INFO] Found existing model at {model_dir}")
                                print("\n🧪 Model Testing Options:")
                                print("  1. Interactive testing with conversation memory (test_model.py)")
                                print("  2. Quick batch testing (quick_test.py)")
                                print("  3. Skip testing")
                                
                                try:
                                    test_choice = input("\nChoose testing option (1-3) [1]: ").strip() or '1'
                                    if test_choice == '1':
                                        print("[INFO] Starting interactive testing...")
                                        import subprocess
                                        subprocess.run(["python", "test_model.py"])
                                    elif test_choice == '2':
                                        print("[INFO] Starting quick batch testing...")
                                        import subprocess
                                        subprocess.run(["python", "quick_test.py"])
                                    elif test_choice == '3':
                                        print("[INFO] Skipping testing")
                                except (EOFError, KeyboardInterrupt):
                                    print("\n[INFO] Testing skipped")
                            else:
                                print("[WARNING] No existing model found to test")
                                print("[INFO] You can try downloading from Hugging Face:")
                                print("  python model_utils.py recover")
                            return
                        elif choice == '7':
                            print("[INFO] Exiting. Please free GPU memory manually and retry.")
                            return
                        else:
                            print("Please enter 1, 2, 3, 4, 5, 6, or 7")
                            continue
                            
                    except (EOFError, KeyboardInterrupt):
                        print("\n[INFO] Using default option 4 (change model)")
                        logging.getLogger().setLevel(original_level)
                        return train_and_upload_model(dataset_dict, auth_token, username)
                
                logging.getLogger().setLevel(original_level)
                print("[INFO] Continuing with selected option...")
                return
            else:
                print("[INFO] Non-interactive mode: Model loading failed")
                return
        else:
            print("[INFO] Model loading failed due to other reasons")
            return
    
    # =============================================================================
    # LORA CONFIGURATION (IF SELECTED)
    # =============================================================================
    
    if finetune_method == "lora":
        if not PEFT_AVAILABLE:
            print("[ERROR] PEFT library not available for LoRA fine-tuning")
            print("[INFO] Install with: pip install peft")
            return
        
        print("[INFO] Applying LoRA (Low-Rank Adaptation) configuration...")
        
        # LoRA configuration for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Causal language modeling task
            inference_mode=False,          # Training mode
            r=16,                         # Rank of adaptation (lower = more efficient)
            lora_alpha=32,                # LoRA scaling parameter
            lora_dropout=0.1,             # Dropout for regularization
            target_modules=["c_attn", "c_proj", "c_fc"]  # DialoGPT attention modules
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Show parameter efficiency
        
    else:
        print("[INFO] Using full fine-tuning (all parameters will be updated)")
    
    # =============================================================================
    # TRAINING CONFIGURATION
    # =============================================================================
    
    def get_training_config():
        """Interactive training configuration selection with fallback"""
        import sys
        
        if torch.cuda.is_available():
            print("\n[INFO] Training Configuration (GPU):")
            print("  1. Conservative - Safe for any GPU (2GB+ VRAM)")
            print("  2. Balanced - Good performance/memory trade-off (6GB+ VRAM)")
            print("  3. Aggressive - High performance (10GB+ VRAM)")
            print("  4. Extreme - Maximum memory efficiency (any VRAM)")
            
            # Check if running in interactive mode
            if sys.stdin.isatty():
                for attempt in range(3):
                    try:
                        choice = input("\nSelect training mode (1-4) [default: 2]: ").strip()
                        if not choice:  # Default to balanced
                            choice = '2'
                        if choice in ['1', '2', '3', '4']:
                            return f"gpu_{choice}"
                        print("Please enter 1, 2, 3, or 4")
                    except (EOFError, KeyboardInterrupt):
                        break
            
            # Fallback to balanced mode
            print("[INFO] Using default: Balanced mode (gpu_2)")
            return "gpu_2"
        else:
            print("\n[INFO] Training Configuration (CPU):")
            print("  1. Fast - Minimal training for quick results")
            print("  2. Quality - Better training with more time")
            
            if sys.stdin.isatty():
                for attempt in range(3):
                    try:
                        choice = input("\nSelect training mode (1-2) [default: 1]: ").strip()
                        if not choice:  # Default to fast
                            choice = '1'
                        if choice in ['1', '2']:
                            return f"cpu_{choice}"
                        print("Please enter 1 or 2")
                    except (EOFError, KeyboardInterrupt):
                        break
            
            # Fallback to fast mode
            print("[INFO] Using default: Fast mode (cpu_1)")
            return "cpu_1"
    
    # Get training configuration early
    training_mode = os.getenv('TRAINING_MODE')
    if not training_mode:
        training_mode = get_training_config()
    print(f"[INFO] Training mode: {training_mode}")
    
    # =============================================================================
    # DATA PREPROCESSING
    # =============================================================================
    
    def preprocess_function(examples):
        """
        Preprocess text examples for causal language modeling.
        
        This function tokenizes the Q&A text and prepares it for training
        by creating input_ids and labels for the model.
        """
        texts = []
        
        # Process each text example in the batch
        for text in examples['text']:
            if isinstance(text, str) and text.strip():
                # Clean and format the text
                formatted_text = text.strip()
                
                # Ensure text ends with EOS token for proper sequence termination
                if not formatted_text.endswith(tokenizer.eos_token):
                    formatted_text += tokenizer.eos_token
                    
                texts.append(formatted_text)
            else:
                # Fallback for invalid entries
                fallback_text = "### Human: What is JVM?\n### Assistant: JVM stands for Java Virtual Machine." + tokenizer.eos_token
                texts.append(fallback_text)
                logging.warning("Used fallback text for invalid entry")
        
        # Tokenize all texts with consistent padding and truncation
        model_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512 if training_mode in ['gpu_1', 'gpu_4'] else 768,  # Adaptive context length
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids (shifted internally by model)
        model_inputs["labels"] = model_inputs["input_ids"][:]
        
        return model_inputs
    
    # Apply preprocessing to datasets
    print("[INFO] Preprocessing training data...")
    train_dataset = dataset_dict['train'].map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset_dict['train'].column_names,
        desc="Tokenizing training data"
    )
    
    print("[INFO] Preprocessing evaluation data...")
    eval_dataset = dataset_dict['test'].map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset_dict['test'].column_names,
        desc="Tokenizing evaluation data"
    )
    
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Evaluation samples: {len(eval_dataset)}")
    
    # Configure training arguments based on mode
    if finetune_method == "lora":
        # LoRA configuration
        training_args = TrainingArguments(
            output_dir=model_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=3e-4,
            warmup_steps=100,
            logging_steps=25,
            save_steps=250,
            eval_strategy="steps",
            eval_steps=250,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=False,
            bf16=torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported(),
            dataloader_num_workers=0,
            report_to=None,
        )
    else:
        # Full fine-tuning configurations
        if training_mode == "gpu_1":  # Conservative
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=2,
                per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=5e-5, warmup_steps=50, logging_steps=20, save_steps=200,
                eval_strategy="steps", eval_steps=200, save_total_limit=1, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=False, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None, gradient_checkpointing=True, optim="adafactor"
            )
        elif training_mode == "gpu_2":  # Balanced
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=3,
                per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
                learning_rate=3e-5, warmup_steps=100, logging_steps=10, save_steps=100,
                eval_strategy="steps", eval_steps=100, save_total_limit=1, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=False, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None, gradient_checkpointing=True
            )
        elif training_mode == "gpu_3":  # Aggressive
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=3,
                per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=2,
                learning_rate=3e-5, warmup_steps=100, logging_steps=10, save_steps=100,
                eval_strategy="steps", eval_steps=100, save_total_limit=2, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=False, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None
            )
        elif training_mode == "gpu_4":  # Extreme
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=3,
                per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=16,
                learning_rate=3e-5, warmup_steps=100, logging_steps=10, save_steps=100,
                eval_strategy="steps", eval_steps=100, save_total_limit=1, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=False, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None, gradient_checkpointing=True, optim="adafactor"
            )
        elif training_mode == "cpu_1":  # Fast CPU
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=1,
                per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
                learning_rate=5e-5, warmup_steps=25, logging_steps=50, save_steps=500,
                eval_strategy="steps", eval_steps=500, save_total_limit=1, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=False, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None, use_cpu=True
            )
        else:  # cpu_2 - Quality CPU
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=2,
                per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=5e-5, warmup_steps=50, logging_steps=20, save_steps=200,
                eval_strategy="steps", eval_steps=200, save_total_limit=1, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=False, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None, use_cpu=True
            )
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not masked language modeling
        pad_to_multiple_of=8,  # Optimize for tensor cores
    )
    
    # =============================================================================
    # TRAINER SETUP AND VALIDATION
    # =============================================================================
    
    # Initialize Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Validate data before training
    print("[INFO] Validating training setup...")
    print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
    print(f"[DEBUG] Sample train item keys: {list(train_dataset[0].keys())}")
    print(f"[DEBUG] Sample input_ids length: {len(train_dataset[0]['input_ids'])}")
    
    try:
        # Test data loading
        sample_batch = next(iter(trainer.get_train_dataloader()))
        print(f"[INFO] Sample batch input_ids shape: {sample_batch['input_ids'].shape}")
        print(f"[INFO] Sample batch labels shape: {sample_batch['labels'].shape}")
        print("[SUCCESS] Data validation passed")
    except Exception as e:
        print(f"[ERROR] Data validation failed: {e}")
        return
    
    # =============================================================================
    # MODEL TRAINING
    # =============================================================================
    
    print("[INFO] Starting model training...")
    print(f"[INFO] This may take 15-60 minutes depending on your hardware")
    
    try:
        # Start training process
        trainer.train()
        print("[SUCCESS] Training completed successfully!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Training failed: {error_msg}")
        
        # Handle CUDA out of memory specifically
        if "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
            print("\n[INFO] GPU Memory Issue Detected!")
            
            # Analyze system and provide intelligent recommendations
            def analyze_and_recommend():
                """Analyze system state and recommend best option"""
                recommendations = []
                
                if torch.cuda.is_available():
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    available = total - reserved
                    
                    print(f"[INFO] GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {available:.1f}GB available of {total:.1f}GB total")
                    
                    # Analyze memory patterns
                    memory_usage_ratio = allocated / total
                    available_ratio = available / total
                    
                    # Check if this is a repeated failure
                    failure_count = getattr(analyze_and_recommend, 'failure_count', 0)
                    analyze_and_recommend.failure_count = failure_count + 1
                    
                    # Analyze error message for specific issues
                    fragmentation_issue = "fragmentation" in error_msg.lower()
                    small_allocation = "20.00 MiB" in error_msg or "30.00 MiB" in error_msg
                    
                    print(f"\n🔍 System Analysis:")
                    print(f"  • GPU: {torch.cuda.get_device_name(0)}")
                    print(f"  • Memory Usage: {memory_usage_ratio*100:.1f}% of total capacity")
                    print(f"  • Available Memory: {available:.1f}GB ({available_ratio*100:.1f}%)")
                    print(f"  • Failure Attempt: #{failure_count}")
                    print(f"  • Model Size: {base_model} (~{model.num_parameters()/1e6:.0f}M params)")
                    
                    # Generate recommendations based on analysis
                    if failure_count == 1:
                        if available >= 8 and total >= 12:
                            recommendations.append((1, "Smart memory optimization - You have sufficient VRAM"))
                            recommendations.append((4, "GPU cleanup utility - Clear any background processes first"))
                        elif available >= 4:
                            recommendations.append((2, "Extreme memory efficiency - Conservative approach"))
                            recommendations.append((1, "Smart memory optimization - Try if cleanup helps"))
                        else:
                            recommendations.append((5, "Change base model - Current model too large"))
                            recommendations.append((2, "Extreme memory efficiency - Last GPU attempt"))
                    
                    elif failure_count == 2:
                        if "DialoGPT-large" in base_model:
                            recommendations.append((5, "Change to DialoGPT-medium - Large model too demanding"))
                            recommendations.append((6, "Change training config - Try conservative mode"))
                        else:
                            recommendations.append((2, "Extreme memory efficiency - More aggressive settings"))
                            recommendations.append((3, "Switch to CPU training - GPU may be unstable"))
                    
                    else:  # failure_count >= 3
                        recommendations.append((7, "Skip training and test existing model - Avoid further issues"))
                        recommendations.append((3, "Switch to CPU training - GPU training problematic"))
                        recommendations.append((5, "Change to smallest model - DialoGPT-small"))
                    
                    # Special cases
                    if fragmentation_issue:
                        recommendations.insert(0, (4, "GPU cleanup utility - Memory fragmentation detected"))
                    
                    if small_allocation and available > 2:
                        recommendations.insert(0, (1, "Smart memory optimization - Small allocation failure with available memory"))
                    
                    if "DialoGPT-large" in base_model and total < 16:
                        recommendations.insert(0, (5, "Change base model - Large model needs 16GB+ VRAM"))
                    
                else:
                    recommendations.append((3, "Switch to CPU training - No GPU available"))
                
                return recommendations
            
            recommendations = analyze_and_recommend()
            
            import sys
            if sys.stdin.isatty():
                while True:  # Unlimited retries
                    print("\n🤖 AI Recommendations (based on system analysis):")
                    for i, (option_num, reason) in enumerate(recommendations[:3], 1):
                        print(f"  {i}. Option {option_num}: {reason}")
                    
                    print("\n📋 All Available Options:")
                    print("  1. Smart memory optimization (recommended for high-VRAM GPUs)")
                    print("  2. Extreme memory efficiency (smallest possible batch)")
                    print("  3. Switch to CPU training")
                    print("  4. Run GPU cleanup utility")
                    print("  5. Change base model (go back to model selection)")
                    print("  6. Change training configuration (go back to training mode)")
                    print("  7. Skip training and test existing model")
                    print("  8. Exit and manually free GPU memory")
                    
                    # Show recommended option
                    if recommendations:
                        recommended_option = recommendations[0][0]
                        print(f"\n💡 Recommended: Option {recommended_option}")
                    
                    try:
                        default_choice = str(recommendations[0][0]) if recommendations else '1'
                        choice = input(f"\nSelect option (1-8) [{default_choice}]: ").strip()
                        if not choice:
                            choice = default_choice
                        
                        if choice == '1':
                            print("[INFO] Applying smart memory optimization...")
                            # Clear GPU memory first
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                import gc
                                gc.collect()
                            
                            # Smart optimization for high-VRAM GPUs
                            training_args.per_device_train_batch_size = 1
                            training_args.gradient_accumulation_steps = 16
                            training_args.gradient_checkpointing = True
                            training_args.optim = "adafactor"
                            training_args.dataloader_pin_memory = False
                            training_args.dataloader_num_workers = 0
                            
                            # Recreate model with memory constraints
                            del model
                            torch.cuda.empty_cache()
                            
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                max_memory={0: "4GB"}  # Limit model to 4GB
                            )
                            
                            trainer = Trainer(
                                model=model,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                            )
                            
                            try:
                                trainer.train()
                                print("[SUCCESS] Training completed with smart optimization!")
                                return  # Success, exit the retry loop
                            except Exception as retry_e:
                                print(f"[ERROR] Smart optimization failed: {retry_e}")
                                print("[INFO] Try a different option or adjust settings.")
                                continue
                                
                        elif choice == '2':
                            print("[INFO] Applying extreme memory efficiency...")
                            # Clear GPU memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                import gc
                                gc.collect()
                            
                            # Extreme settings
                            training_args.per_device_train_batch_size = 1
                            training_args.gradient_accumulation_steps = 64
                            training_args.gradient_checkpointing = True
                            training_args.optim = "adafactor"
                            training_args.dataloader_pin_memory = False
                            training_args.dataloader_num_workers = 0
                            
                            # Recreate model with extreme constraints
                            del model
                            torch.cuda.empty_cache()
                            
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                max_memory={0: "2GB"}  # Very conservative
                            )
                            
                            trainer = Trainer(
                                model=model,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                            )
                            
                            try:
                                trainer.train()
                                print("[SUCCESS] Training completed with extreme efficiency!")
                                return  # Success, exit the retry loop
                            except Exception as retry_e:
                                print(f"[ERROR] Extreme efficiency failed: {retry_e}")
                                print("[INFO] Try a different option or consider CPU training.")
                                continue
                                
                        elif choice == '3':
                            print("[INFO] Switching to CPU training...")
                            # Move model to CPU
                            model = model.cpu()
                            training_args.use_cpu = True
                            training_args.fp16 = False
                            training_args.per_device_train_batch_size = 1
                            training_args.gradient_accumulation_steps = 8
                            
                            trainer = Trainer(
                                model=model,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                            )
                            
                            try:
                                trainer.train()
                                print("[SUCCESS] Training completed on CPU!")
                                return  # Success, exit
                            except Exception as cpu_e:
                                print(f"[ERROR] CPU training failed: {cpu_e}")
                                return
                                
                        elif choice == '4':
                            print("[INFO] Running GPU cleanup utility...")
                            import subprocess
                            try:
                                subprocess.run(["python", "gpu_cleanup.py"], check=True)
                            except:
                                print("[ERROR] Could not run gpu_cleanup.py")
                                print("[TIP] Run manually: python gpu_cleanup.py")
                            continue  # Go back to options
                            
                        elif choice == '5':
                            print("[INFO] Returning to base model selection...")
                            # Clear current model from memory
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Recursively call the training function with model selection
                            return train_and_upload_model(dataset_dict, auth_token, username)
                            
                        elif choice == '6':
                            print("[INFO] Returning to training configuration...")
                            # Clear current model from memory
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Reset training mode and restart
                            if 'TRAINING_MODE' in os.environ:
                                del os.environ['TRAINING_MODE']
                            return train_and_upload_model(dataset_dict, auth_token, username)
                            
                        elif choice == '7':
                            print("[INFO] Skipping training and proceeding to model testing...")
                            # Check if there's an existing model to test
                            model_dir = "./models/jvm_troubleshooting_model"
                            if os.path.exists(model_dir):
                                print(f"[INFO] Found existing model at {model_dir}")
                                # Show testing options
                                print("\n🧪 Model Testing Options:")
                                print("  1. Interactive testing with conversation memory (test_model.py)")
                                print("  2. Quick batch testing (quick_test.py)")
                                print("  3. Skip testing")
                                
                                try:
                                    test_choice = input("\nChoose testing option (1-3) [1]: ").strip()
                                    if not test_choice:
                                        test_choice = '1'
                                    
                                    if test_choice == '1':
                                        print("[INFO] Starting interactive testing...")
                                        import subprocess
                                        subprocess.run(["python", "test_model.py"])
                                    elif test_choice == '2':
                                        print("[INFO] Starting quick batch testing...")
                                        import subprocess
                                        subprocess.run(["python", "quick_test.py"])
                                    elif test_choice == '3':
                                        print("[INFO] Skipping testing")
                                    
                                except (EOFError, KeyboardInterrupt):
                                    print("\n[INFO] Testing skipped")
                            else:
                                print("[WARNING] No existing model found to test")
                                print("[INFO] You can try downloading from Hugging Face:")
                                print("  python model_utils.py recover")
                            return
                            
                        elif choice == '8':
                            print("[INFO] Exiting. Please free GPU memory manually and retry.")
                            print("[TIP] Run: nvidia-smi to check GPU processes")
                            print("[TIP] Kill processes with: kill -9 <PID>")
                            print("[TIP] Run: python gpu_cleanup.py for automated cleanup")
                            return
                        else:
                            print("Please enter 1, 2, 3, 4, 5, 6, 7, or 8")
                            continue
                            
                    except (EOFError, KeyboardInterrupt):
                        print("\n[INFO] Using default option 1 (smart optimization)")
                        choice = '1'
                        continue
                # This line will never be reached due to infinite loop
            else:
                # Non-interactive fallback - try smart optimization first
                print("[INFO] Non-interactive mode: Trying smart memory optimization...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    import gc
                    gc.collect()
                
                # Smart optimization settings
                training_args.per_device_train_batch_size = 1
                training_args.gradient_accumulation_steps = 16
                training_args.gradient_checkpointing = True
                training_args.optim = "adafactor"
                training_args.dataloader_pin_memory = False
                training_args.dataloader_num_workers = 0
                
                # Recreate model with memory limit
                del model
                torch.cuda.empty_cache()
                
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "4GB"}
                )
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                )
                
                try:
                    trainer.train()
                    print("[SUCCESS] Training completed with smart optimization!")
                except Exception as retry_e:
                    print(f"[ERROR] Smart optimization failed: {retry_e}")
                    print("[INFO] Run interactively for more options: python main.py")
                    return
        else:
            # Handle any other training error with same comprehensive options
            print("\n[INFO] Training Error Detected!")
            
            import sys
            if sys.stdin.isatty():
                while True:  # Unlimited retries for any error
                    print("\n📋 All Available Options:")
                    print("  1. Smart memory optimization (recommended for high-VRAM GPUs)")
                    print("  2. Extreme memory efficiency (smallest possible batch)")
                    print("  3. Switch to CPU training")
                    print("  4. Run GPU cleanup utility")
                    print("  5. Change base model (go back to model selection)")
                    print("  6. Change training configuration (go back to training mode)")
                    print("  7. Skip training and test existing model")
                    print("  8. Exit and manually free GPU memory")
                    
                    try:
                        choice = input("\nSelect option (1-8) [5]: ").strip()
                        if not choice:
                            choice = '5'  # Default to changing model for general errors
                        
                        if choice == '5':
                            print("[INFO] Returning to base model selection...")
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            return train_and_upload_model(dataset_dict, auth_token, username)
                        elif choice == '6':
                            print("[INFO] Returning to training configuration...")
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if 'TRAINING_MODE' in os.environ:
                                del os.environ['TRAINING_MODE']
                            return train_and_upload_model(dataset_dict, auth_token, username)
                        elif choice == '7':
                            print("[INFO] Skipping training and proceeding to model testing...")
                            model_dir = "./models/jvm_troubleshooting_model"
                            if os.path.exists(model_dir):
                                print(f"[INFO] Found existing model at {model_dir}")
                                print("\n🧪 Model Testing Options:")
                                print("  1. Interactive testing with conversation memory (test_model.py)")
                                print("  2. Quick batch testing (quick_test.py)")
                                print("  3. Skip testing")
                                
                                try:
                                    test_choice = input("\nChoose testing option (1-3) [1]: ").strip()
                                    if not test_choice:
                                        test_choice = '1'
                                    
                                    if test_choice == '1':
                                        print("[INFO] Starting interactive testing...")
                                        import subprocess
                                        subprocess.run(["python", "test_model.py"])
                                    elif test_choice == '2':
                                        print("[INFO] Starting quick batch testing...")
                                        import subprocess
                                        subprocess.run(["python", "quick_test.py"])
                                    elif test_choice == '3':
                                        print("[INFO] Skipping testing")
                                    
                                except (EOFError, KeyboardInterrupt):
                                    print("\n[INFO] Testing skipped")
                            else:
                                print("[WARNING] No existing model found to test")
                                print("[INFO] You can try downloading from Hugging Face:")
                                print("  python model_utils.py recover")
                            return
                        elif choice == '8':
                            print("[INFO] Exiting. Check the error message above for details.")
                            return
                        elif choice == '1':
                            print("[INFO] Applying smart memory optimization...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                import gc
                                gc.collect()
                            training_args.per_device_train_batch_size = 1
                            training_args.gradient_accumulation_steps = 16
                            training_args.gradient_checkpointing = True
                            training_args.optim = "adafactor"
                            training_args.dataloader_pin_memory = False
                            training_args.dataloader_num_workers = 0
                            del model
                            torch.cuda.empty_cache()
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model, torch_dtype=torch.float16, device_map="auto",
                                low_cpu_mem_usage=True, max_memory={0: "4GB"}
                            )
                            trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                                             train_dataset=train_dataset, eval_dataset=eval_dataset)
                            try:
                                trainer.train()
                                print("[SUCCESS] Training completed with smart optimization!")
                                return
                            except Exception as retry_e:
                                print(f"[ERROR] Smart optimization failed: {retry_e}")
                                continue
                        elif choice == '2':
                            print("[INFO] Applying extreme memory efficiency...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                import gc
                                gc.collect()
                            training_args.per_device_train_batch_size = 1
                            training_args.gradient_accumulation_steps = 64
                            training_args.gradient_checkpointing = True
                            training_args.optim = "adafactor"
                            training_args.dataloader_pin_memory = False
                            training_args.dataloader_num_workers = 0
                            del model
                            torch.cuda.empty_cache()
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model, torch_dtype=torch.float16, device_map="auto",
                                low_cpu_mem_usage=True, max_memory={0: "2GB"}
                            )
                            trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                                             train_dataset=train_dataset, eval_dataset=eval_dataset)
                            try:
                                trainer.train()
                                print("[SUCCESS] Training completed with extreme efficiency!")
                                return
                            except Exception as retry_e:
                                print(f"[ERROR] Extreme efficiency failed: {retry_e}")
                                continue
                        elif choice == '3':
                            print("[INFO] Switching to CPU training...")
                            model = model.cpu()
                            training_args.use_cpu = True
                            training_args.fp16 = False
                            training_args.per_device_train_batch_size = 1
                            training_args.gradient_accumulation_steps = 8
                            trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                                             train_dataset=train_dataset, eval_dataset=eval_dataset)
                            try:
                                trainer.train()
                                print("[SUCCESS] Training completed on CPU!")
                                return
                            except Exception as cpu_e:
                                print(f"[ERROR] CPU training failed: {cpu_e}")
                                return
                        elif choice == '4':
                            print("[INFO] Running GPU cleanup utility...")
                            import subprocess
                            try:
                                subprocess.run(["python", "gpu_cleanup.py"], check=True)
                                print("[INFO] GPU cleanup completed. Retrying...")
                                continue
                            except Exception as cleanup_e:
                                print(f"[ERROR] GPU cleanup failed: {cleanup_e}")
                                print("[TIP] Run manually: python gpu_cleanup.py")
                                continue
                        else:
                            print("Please enter 1, 2, 3, 4, 5, 6, 7, or 8")
                            continue
                            
                    except (EOFError, KeyboardInterrupt):
                        print("\n[INFO] Using default option 5 (change model)")
                        choice = '5'
                        continue
            else:
                print("[INFO] Non-interactive mode: This might be due to:")
                print("  - Corrupted training data")
                print("  - Hardware compatibility issues")
                print("  - Model configuration problems")
                print("[INFO] Run interactively for more options: python main.py")
                return
    
    # Save final model (moved outside try block)
    print("[INFO] Saving trained model...")
    trainer.save_model()
    tokenizer.save_pretrained(model_dir)
    
    # Save training metrics
    if hasattr(trainer.state, 'log_history'):
        import json
        with open(f"{model_dir}/training_log.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
    
    print(f"[SUCCESS] Model saved to {model_dir}")


    # =============================================================================
    # MODEL UPLOAD TO HUGGING FACE HUB
    # =============================================================================
    
    if auth_token:
        try:
            print(f"[INFO] Uploading model to Hugging Face Hub as {model_id}...")
            
            # Initialize Hugging Face API
            api = HfApi()
            
            # Create model repository
            try:
                create_repo(repo_id=model_id, token=auth_token, repo_type="model")
                print(f"[SUCCESS] Repository {model_id} created!")
            except HfHubHTTPError as e:
                if "already exists" in str(e):
                    print(f"[INFO] Repository {model_id} already exists, updating...")
                else:
                    raise e
            
            # Upload all model files
            api.upload_folder(
                folder_path=model_dir,
                repo_id=model_id,
                token=auth_token,
                repo_type="model"
            )
            
            print(f"[SUCCESS] Model uploaded to: https://huggingface.co/{model_id}")
            
        except Exception as e:
            print(f"[ERROR] Failed to upload model to Hugging Face: {e}")
            print("[INFO] Model is still available locally for testing")
    else:
        print("[INFO] No Hugging Face token provided - model saved locally only")

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def main():
    """
    Main processing pipeline that orchestrates the entire workflow:
    1. PDF text extraction
    2. Q&A generation using AI
    3. Dataset creation and upload
    4. Optional model training
    """
    
    # =============================================================================
    # CONFIGURATION AND SETUP
    # =============================================================================
    
    pdf_path = "jvm_troubleshooting_guide.pdf"
    dataset_name = "jvm_troubleshooting_guide"
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file '{pdf_path}' not found in current directory")
        print("[INFO] Please add your PDF file and run again")
        return
    
    # Load configuration from environment variables
    ai_provider = os.getenv('AI_PROVIDER', 'ollama')
    ai_model = os.getenv('AI_MODEL')
    overwrite_dataset = os.getenv('OVERWRITE_DATASET', 'false').lower() == 'true'
    train_model = os.getenv('TRAIN_MODEL', 'false').lower() == 'true'
    auth_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    print(f"[INFO] Configuration:")
    print(f"  - AI Provider: {ai_provider}")
    print(f"  - AI Model: {ai_model or 'default'}")
    print(f"  - Overwrite Dataset: {overwrite_dataset}")
    print(f"  - Train Model: {train_model}")
    print(f"  - HF Token: {'configured' if auth_token else 'not provided'}")
    
    # =============================================================================
    # DATASET PROCESSING
    # =============================================================================
    
    dataset_path = f"./dataset/{dataset_name}"
    
    # Check if dataset already exists
    if os.path.exists(dataset_path) and not overwrite_dataset:
        print(f"[INFO] Loading existing dataset from {dataset_path}")
        try:
            dataset_dict = load_from_disk(dataset_path)
            print(f"[SUCCESS] Loaded dataset:")
            print(f"  - Training examples: {len(dataset_dict['train'])}")
            print(f"  - Test examples: {len(dataset_dict['test'])}")
        except Exception as e:
            print(f"[ERROR] Failed to load existing dataset: {e}")
            print("[INFO] Will regenerate dataset...")
            overwrite_dataset = True
    
    # Generate new dataset if needed
    if not os.path.exists(dataset_path) or overwrite_dataset:
        if os.path.exists(dataset_path):
            print(f"[INFO] Overwriting existing dataset at {dataset_path}")
        
        # Step 1: Extract PDF content
        print(f"[INFO] Extracting text from {pdf_path}...")
        try:
            content_list = read_pdf_content(pdf_path)
            print(f"[SUCCESS] Extracted content from {len(content_list)} pages")
        except Exception as e:
            print(f"[ERROR] Failed to extract PDF content: {e}")
            return
        
        # Step 2: Generate Q&A pairs
        print("[INFO] Generating question-answer pairs using AI...")
        qa_pairs = []
        
        # Process each page with progress tracking
        for i, content in enumerate(tqdm(content_list, desc="Generating Q&A")):
            if content.strip():  # Skip empty pages
                try:
                    qa_response = prompt_engineered_api(content, ai_provider, ai_model)
                    
                    # Validate Q&A format
                    if (qa_response and 
                        "### Human:" in qa_response and 
                        "### Assistant:" in qa_response):
                        qa_pairs.append(qa_response)
                        logging.debug(f"Generated Q&A for page {i+1}")
                    else:
                        logging.warning(f"Invalid Q&A format for page {i+1}")
                        
                except Exception as e:
                    logging.error(f"Failed to generate Q&A for page {i+1}: {e}")
        
        # Validate we have sufficient data
        if not qa_pairs:
            print("[ERROR] No valid Q&A pairs generated!")
            print("[INFO] This might be due to:")
            print("  - AI service connectivity issues")
            print("  - Invalid API keys")
            print("  - Poor PDF text extraction")
            return
        
        print(f"[SUCCESS] Generated {len(qa_pairs)} Q&A pairs")
        
        # Step 3: Create structured dataset
        print("[INFO] Creating structured dataset...")
        try:
            # Create HuggingFace dataset
            dataset = Dataset.from_dict({"text": qa_pairs})
            
            # Split into train/test (80/20 split)
            train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
            dataset_dict = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })
            
            # Save dataset locally
            os.makedirs("./dataset", exist_ok=True)
            dataset_dict.save_to_disk(dataset_path)
            print(f"[SUCCESS] Dataset saved to {dataset_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create dataset: {e}")
            return
        
        # Step 4: Upload dataset to Hugging Face Hub
        if auth_token:
            try:
                # Get username from token
                username = HfApi(token=auth_token).whoami()["name"]
                repo_id = f"{username}/{dataset_name}"
                
                print(f"[INFO] Uploading dataset to {repo_id}...")
                
                # Create repository
                try:
                    create_repo(repo_id=repo_id, token=auth_token, repo_type="dataset")
                    print(f"[SUCCESS] Repository {repo_id} created!")
                except HfHubHTTPError as e:
                    if "already exists" in str(e):
                        print(f"[INFO] Repository {repo_id} already exists, updating...")
                    else:
                        raise e
                
                # Push dataset
                dataset_dict.push_to_hub(repo_id, token=auth_token)
                print(f"[SUCCESS] Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
                
            except Exception as e:
                print(f"[ERROR] Failed to upload dataset: {e}")
                print("[INFO] Dataset is still available locally")
        else:
            print("[INFO] No Hugging Face token - dataset saved locally only")
    
    # =============================================================================
    # MODEL TRAINING (OPTIONAL)
    # =============================================================================
    
    if train_model:
        print("[INFO] Starting model training pipeline...")
        
        if auth_token:
            try:
                username = HfApi(token=auth_token).whoami()["name"]
                train_and_upload_model(dataset_dict, auth_token, username)
            except Exception as e:
                print(f"[ERROR] Failed to get username from token: {e}")
                print("[INFO] Training locally without upload...")
                train_and_upload_model(dataset_dict, None, "local")
        else:
            print("[INFO] Training locally (no Hugging Face token provided)")
            train_and_upload_model(dataset_dict, None, "local")
    else:
        print("[INFO] Model training skipped")
        print("[INFO] To enable training, set TRAIN_MODEL=true in .env file")
    
    # =============================================================================
    # COMPLETION SUMMARY
    # =============================================================================
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"✅ Dataset: {len(dataset_dict['train'])} train + {len(dataset_dict['test'])} test examples")
    
    if train_model and os.path.exists("./models/jvm_troubleshooting_model"):
        print("✅ Model: Trained and saved locally")
        print("\n🧪 Test your model:")
        print("   python test_model.py    # Interactive testing")
        print("   python quick_test.py    # Batch validation")
    
    print("\n📚 Next steps:")
    print("   • Review generated Q&A pairs in dataset/")
    print("   • Test model performance with sample questions")
    print("   • Fine-tune training parameters if needed")
    print("   • Deploy model for production use")
    
    if not train_model or not os.path.exists("./models/jvm_troubleshooting_model"):
        print("\n🔧 Training Issues?")
        print("   • Run: python gpu_cleanup.py (for GPU memory issues)")
        print("   • Set TRAIN_MODEL=true in .env to enable training")
    
    print("\n✨ Thank you for using the PDF to Q&A Generator!")

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Script entry point with error handling and logging setup.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error in main pipeline: {e}")
        print(f"[ERROR] Pipeline failed: {e}")
        print("[INFO] Check logs for detailed error information")