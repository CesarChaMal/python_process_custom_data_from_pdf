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
    base_model = os.getenv('BASE_MODEL', 'microsoft/DialoGPT-medium')  # Balanced performance
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
    # MODEL AND TOKENIZER LOADING
    # =============================================================================
    
    print("[INFO] Loading base model and tokenizer...")
    
    try:
        # Load tokenizer with proper configuration
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Set padding token (required for batch processing)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load base model for causal language modeling
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
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
            max_length=768,  # Increased context length for better understanding
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
    
    # Configure training arguments based on fine-tuning method
    if finetune_method == "lora":
        # LoRA: More aggressive training since fewer parameters are updated
        training_args = TrainingArguments(
            output_dir=model_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,              # More epochs for LoRA
            per_device_train_batch_size=4,   # Larger batch size possible
            per_device_eval_batch_size=4,
            learning_rate=3e-4,              # Higher learning rate for LoRA
            warmup_steps=100,
            logging_steps=25,
            save_steps=250,
            eval_strategy="steps",
            eval_steps=250,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
            report_to=None,                  # Disable wandb/tensorboard
        )
    else:
        # Full fine-tuning: Conservative approach for stability
        training_args = TrainingArguments(
            output_dir=model_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,              # Sufficient epochs for convergence
            per_device_train_batch_size=1,   # Small batch for memory efficiency
            per_device_eval_batch_size=1,
            learning_rate=3e-5,              # Lower learning rate for stability
            warmup_steps=100,                # Gradual learning rate increase
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,   # Effective batch size = 4
            weight_decay=0.01,               # L2 regularization
            max_grad_norm=1.0,               # Gradient clipping for stability
            report_to=None,
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