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
                
                while True:
                    choice = input("\nSelect model (1, 2, or 3): ").strip()
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
            else:
                base_model = 'microsoft/DialoGPT-small'
                print(f"[INFO] Auto-selected DialoGPT-small (limited GPU memory: {available:.1f}GB)")
        else:
            print("\n[INFO] CPU Mode Detected")
            print("\n[INFO] Model Selection (CPU):")
            print("  1. DialoGPT-small (117M params) - Recommended for CPU")
            print("  2. DialoGPT-medium (345M params) - Slower on CPU")
            
            while True:
                choice = input("\nSelect model (1 or 2): ").strip()
                if choice == '1':
                    base_model = 'microsoft/DialoGPT-small'
                    break
                elif choice == '2':
                    base_model = 'microsoft/DialoGPT-medium'
                    print("[WARNING] Medium model will be slow on CPU")
                    break
                else:
                    print("Please enter 1 or 2")
    
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
        print(f"[ERROR] Failed to load base model: {e}")
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
    
    # =============================================================================
    # TRAINING CONFIGURATION
    # =============================================================================
    
    def get_training_config():
        """Interactive training configuration selection"""
        if torch.cuda.is_available():
            print("\n[INFO] Training Configuration (GPU):")
            print("  1. Conservative - Safe for any GPU (2GB+ VRAM)")
            print("  2. Balanced - Good performance/memory trade-off (6GB+ VRAM)")
            print("  3. Aggressive - High performance (10GB+ VRAM)")
            print("  4. Extreme - Maximum memory efficiency (any VRAM)")
            
            while True:
                choice = input("\nSelect training mode (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return f"gpu_{choice}"
                print("Please enter 1, 2, 3, or 4")
        else:
            print("\n[INFO] Training Configuration (CPU):")
            print("  1. Fast - Minimal training for quick results")
            print("  2. Quality - Better training with more time")
            
            while True:
                choice = input("\nSelect training mode (1-2): ").strip()
                if choice in ['1', '2']:
                    return f"cpu_{choice}"
                print("Please enter 1 or 2")
    
    # Get training configuration
    training_mode = os.getenv('TRAINING_MODE') or get_training_config()
    
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
                dataloader_pin_memory=False, fp16=True, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None, gradient_checkpointing=True, optim="adafactor"
            )
        elif training_mode == "gpu_2":  # Balanced
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=3,
                per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
                learning_rate=3e-5, warmup_steps=100, logging_steps=10, save_steps=100,
                eval_strategy="steps", eval_steps=100, save_total_limit=1, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=True, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None, gradient_checkpointing=True
            )
        elif training_mode == "gpu_3":  # Aggressive
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=3,
                per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=2,
                learning_rate=3e-5, warmup_steps=100, logging_steps=10, save_steps=100,
                eval_strategy="steps", eval_steps=100, save_total_limit=2, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=True, dataloader_num_workers=0, weight_decay=0.01,
                max_grad_norm=1.0, report_to=None
            )
        elif training_mode == "gpu_4":  # Extreme
            training_args = TrainingArguments(
                output_dir=model_dir, overwrite_output_dir=True, num_train_epochs=3,
                per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=16,
                learning_rate=3e-5, warmup_steps=100, logging_steps=10, save_steps=100,
                eval_strategy="steps", eval_steps=100, save_total_limit=1, remove_unused_columns=False,
                dataloader_pin_memory=False, fp16=True, dataloader_num_workers=0, weight_decay=0.01,
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
    
    print(f"[INFO] Training mode: {training_mode}")
    
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
        print(f"[ERROR] Training failed: {e}")
        print("[INFO] This might be due to:")
        print("  - Insufficient GPU/CPU memory")
        print("  - Corrupted training data")
        print("  - Hardware compatibility issues")
        return
    
        # Save final model
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