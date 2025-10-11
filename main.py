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
from training_health_monitor import TrainingHealthMonitor
from training_utils import handle_training_error_menu, recreate_model_with_config
from create_model_card import generate_and_upload_model_card
from check_hf import check_hf_connection
from check_gpu import check_gpu
from upload_model import upload_model_to_hf
from upload_dataset import upload_dataset_to_hf

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
    # Enhanced prompt for high-quality technical training data
    prompt = f"""
    You are a senior JVM performance engineer creating high-quality training data. Based on the technical content below, create ONE comprehensive question-answer pair.

    Content: {text}

    STRICT Requirements:
    1. Question must be specific, realistic, and commonly asked by developers
    2. Answer must be technically accurate with NO fabricated information
    3. Use ONLY real JVM tools: jstat, jmap, jconsole, VisualVM, JProfiler, Eclipse MAT, GCViewer
    4. Use ONLY real JVM parameters: -Xmx, -Xms, -XX:NewRatio, -XX:MaxMetaspaceSize, -XX:+UseG1GC, etc.
    5. Provide complete, actionable solutions with specific commands
    6. Answer must be 150-300 characters for comprehensive coverage
    7. Include practical examples with realistic values
    8. Use EXACTLY this format:

    ### Human:
    [Specific, realistic question]
    ### Assistant:
    [Complete, accurate answer with examples and commands]

    GOOD Example:
    ### Human:
    How do I identify memory leaks in a Java application?
    ### Assistant:
    To identify memory leaks: 1) Use jmap -histo <pid> to see object counts over time, 2) Generate heap dumps with jmap -dump:format=b,file=heap.hprof <pid>, 3) Analyze with Eclipse MAT or VisualVM to find objects not being garbage collected, 4) Look for growing collections or static references, 5) Monitor with -XX:+PrintGCDetails to see if heap usage keeps increasing after GC.

    BAD Examples to AVOID:
    - Fake tools: "AnalyzingGarbage collected Logs", "JVMProfiler Pro"
    - Fake parameters: "tuned ForThreadExecutionPatterns", "-XX:CustomMemoryTuning"
    - Fake experts: "urologist David Carr", "Dr. Smith recommends"
    - Incomplete answers: "Use jstat to check" (too short)
    """
    
    response = call_ai(prompt, provider, model)
    
    # Enhanced response validation
    if response and "### Human:" in response and "### Assistant:" in response:
        assistant_part = response.split("### Assistant:")[-1].strip()
        
        # Quality checks
        quality_issues = []
        if len(assistant_part) < 100:
            quality_issues.append("too short")
        if "urologist David Carr" in response or "Dr. Smith" in response:
            quality_issues.append("fake expert names")
        if "tuned ForThreadExecutionPatterns" in response or "AnalyzingGarbage collected Logs" in response:
            quality_issues.append("fake tools/parameters")
        if not any(tool in response.lower() for tool in ['jstat', 'jmap', 'jconsole', 'visualvm', 'gc', 'heap', 'memory']):
            quality_issues.append("missing JVM tools/concepts")
        
        # Retry if quality issues found
        if quality_issues:
            logging.warning(f"Quality issues detected: {', '.join(quality_issues)}. Retrying...")
            retry_prompt = prompt + f"\n\nIMPORTANT: Avoid these issues: {', '.join(quality_issues)}. Provide a detailed, accurate answer with real JVM tools and at least 150 characters."
            response = call_ai(retry_prompt, provider, model)
            
            # Final validation
            if response and "### Assistant:" in response:
                final_assistant = response.split("### Assistant:")[-1].strip()
                if len(final_assistant) >= 100:
                    logging.info("Retry successful - quality improved")
                else:
                    logging.warning("Retry still produced short answer")
    
    return response


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
    
    def select_base_model():
        """Interactive base model selection with memory analysis"""
        # Temporarily disable debug logging to avoid interference
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        
        try:
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                available = total_mem - reserved
                
                print(f"\n[INFO] GPU Memory Analysis:")
                print(f"  Total: {total_mem:.1f} GB")
                print(f"  Available: {available:.1f} GB")
                
                print("\n[INFO] Model Selection (GPU):")
                print("  1. DialoGPT-small (117M params) - Fast, 2-4GB VRAM")
                print("  2. DialoGPT-medium (345M params) - Balanced, 4-6GB VRAM")
                print("  3. DialoGPT-large (774M params) - Best quality, 8-12GB VRAM")
                
                import sys
                sys.stdout.flush()
                
                if sys.stdin.isatty():
                    for attempt in range(3):
                        try:
                            choice = input("\nSelect model (1, 2, or 3) [default: 2]: ").strip()
                            if not choice:
                                choice = '2'
                            if choice == '1':
                                return 'microsoft/DialoGPT-small'
                            elif choice == '2':
                                return 'microsoft/DialoGPT-medium'
                            elif choice == '3':
                                if available >= 8:
                                    return 'microsoft/DialoGPT-large'
                                else:
                                    print(f"Large model requires 8GB+ available memory (you have {available:.1f}GB)")
                                    continue
                            else:
                                print("Please enter 1, 2, or 3")
                        except (EOFError, KeyboardInterrupt):
                            break
                
                print("[INFO] Using default: DialoGPT-medium")
                return 'microsoft/DialoGPT-medium'
            else:
                print("\n[INFO] CPU Mode Detected")
                print("\n[INFO] Model Selection (CPU):")
                print("  1. DialoGPT-small (117M params) - Recommended for CPU")
                print("  2. DialoGPT-medium (345M params) - Slower on CPU")
                
                import sys
                sys.stdout.flush()
                
                if sys.stdin.isatty():
                    for attempt in range(3):
                        try:
                            choice = input("\nSelect model (1 or 2) [default: 1]: ").strip()
                            if not choice:
                                choice = '1'
                            if choice == '1':
                                return 'microsoft/DialoGPT-small'
                            elif choice == '2':
                                print("[WARNING] Medium model will be slow on CPU")
                                return 'microsoft/DialoGPT-medium'
                            else:
                                print("Please enter 1 or 2")
                        except (EOFError, KeyboardInterrupt):
                            break
                
                print("[INFO] Using default: DialoGPT-small")
                return 'microsoft/DialoGPT-small'
        finally:
            # Always restore logging level
            logging.getLogger().setLevel(original_level)
    
    # Get configuration from environment variables with sensible defaults
    base_model = os.getenv('BASE_MODEL')
    if not base_model:
        base_model = select_base_model()
    
    finetune_method = os.getenv('FINETUNE_METHOD', 'full')  # Full fine-tuning by default
    
    # Use consistent model directory name for testing
    model_name = "jvm_troubleshooting_model"
    
    # Create descriptive model ID for Hugging Face upload with postfix
    base_model_short = base_model.split('/')[-1].lower()  # e.g., "dialogpt-medium"
    method_suffix = "lora" if finetune_method == "lora" else "full"
    hf_model_name = f"jvm_troubleshooting_{base_model_short}_{method_suffix}"
    model_id = f"{username}/{hf_model_name}"
    
    print(f"[INFO] Base model: {base_model}")
    print(f"[INFO] Fine-tuning method: {finetune_method}")
    print(f"[INFO] Target model ID: {model_id}")
    print(f"[INFO] Model name format: jvm_troubleshooting_<base>_<method>")
    
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
    
    # Initialize model variable to avoid scope issues
    model = None
    
    try:
        # Load tokenizer with proper configuration
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Set padding token (required for batch processing)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Enhanced automatic environment detection and configuration
        def get_optimal_device_config():
            """Automatically detect hardware and return optimal configuration"""
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_name = torch.cuda.get_device_name(0)
                
                print(f"[INFO] GPU Detected: {gpu_name} ({total_mem:.1f}GB)")
                
                # Smart memory allocation based on GPU size
                if total_mem >= 12:  # High-end GPU (RTX 4090, etc.)
                    max_memory = "8GB"
                    batch_size = 2
                elif total_mem >= 8:   # Mid-range GPU
                    max_memory = "6GB"
                    batch_size = 2
                else:  # Low-end GPU
                    max_memory = "4GB"
                    batch_size = 1
                
                return {
                    'device_type': 'gpu',
                    'config': {
                        'torch_dtype': torch.float32,  # Universal compatibility
                        'device_map': "auto",
                        'low_cpu_mem_usage': True,
                        'max_memory': {0: max_memory}
                    },
                    'training': {
                        'per_device_train_batch_size': batch_size,
                        'gradient_accumulation_steps': 8 // batch_size,
                        'fp16': False,  # Always use FP32 for stability
                        'use_cpu': False
                    }
                }
            else:
                print("[INFO] CPU Mode Detected - Optimizing for CPU training")
                return {
                    'device_type': 'cpu',
                    'config': {
                        'torch_dtype': torch.float32,
                        'low_cpu_mem_usage': True,
                        'use_cache': False
                    },
                    'training': {
                        'per_device_train_batch_size': 1,
                        'gradient_accumulation_steps': 8,
                        'fp16': False,  # CPU doesn't support FP16
                        'use_cpu': True
                    }
                }
        
        # Get optimal configuration for current hardware
        device_config = get_optimal_device_config()
        print(f"[INFO] Using {device_config['device_type'].upper()} optimized configuration")
        
        # Load model with enhanced stability for large models
        if "large" in base_model.lower():
            # Special handling for DialoGPT-large to prevent segfaults
            os.makedirs("./offload", exist_ok=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={0: "10GB"} if torch.cuda.is_available() else None,
                offload_folder="./offload" if torch.cuda.is_available() else None
            )
        else:
            # Standard loading for smaller models
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map=None
            )
            if torch.cuda.is_available():
                model = model.cuda()
        
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
                if model is not None:
                    del model
                    model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            def reload_model_with_config(config):
                """Reload model with specific configuration"""
                nonlocal model
                clear_gpu_memory()
                model = AutoModelForCausalLM.from_pretrained(base_model, **config)
                return model
            
            # Use unified error handling menu
            result = handle_training_error_menu(model, base_model, None, None)
            
            import sys
            if sys.stdin.isatty():
                # Temporarily disable ALL logging to avoid interference
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.CRITICAL)
                
                # Flush all output streams
                sys.stdout.flush()
                sys.stderr.flush()
                
                result = handle_error_menu()
                
                # Restore logging level
                logging.getLogger().setLevel(original_level)
                
                if result == "restart":
                    return train_and_upload_model(dataset_dict, auth_token, username)
                elif result == True:
                    print("[INFO] Continuing with selected option...")
                    # Continue with training setup below
                else:
                    return  # Exit training
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
        
        # Temporarily disable debug logging to avoid interference
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        
        try:
            if torch.cuda.is_available():
                print("\n[INFO] Training Configuration (GPU):")
                print("  1. Conservative - Safe for any GPU (2GB+ VRAM)")
                print("  2. Balanced - Good performance/memory trade-off (6GB+ VRAM)")
                print("  3. Aggressive - High performance (10GB+ VRAM)")
                print("  4. Extreme - Maximum memory efficiency (any VRAM)")
                
                # Flush output before input
                sys.stdout.flush()
                
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
                
                sys.stdout.flush()
                
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
        finally:
            # Always restore logging level
            logging.getLogger().setLevel(original_level)
    
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
            max_length=512 if device_config['device_type'] == 'cpu' else 768,  # Adaptive context length
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

    # Best Practices Summary

    # Learning Rate Guidelines by Model Size:
    # 1.- Small models( < 200M params): 1e-5 to 5e-5
    # 2.- Medium models(200M - 500M params): 5e-6 to 1e-5
    # 3.- Large models( > 500M params): 1e-6 to 5e-6
    # 4.- LoRA fine - tuning: 1e-5 to 5e-5(not 3e-4!)

    # Critical Safety Parameters:
    # 1.- max_grad_norm: 0.3 to 0.5(lower is safer)
    # 2.- warmup_steps: At least 10 % of total steps
    # 3.- weight_decay: 0.05 to 0.1(helps regularization)
    # 4.- adam_epsilon: 1e-8(numerical stability)
    # 5.- fp16: True for GPU(mixed precision helps)

    # Model - Data Size Rules: \
    # 1.- 100 training samples: Use models < 200M params
    # 2.- 100 - 500 samples: Models up to 350M params
    # 3.- 500 - 1000 samples: Models up to 500M params
    # 4.- 1000 + samples: Can try larger models

    # Key Problems in the Original Configurations

    # Critical Issues Found:
    # 1.- LoRA config: Learning rate 3e-4 is 60x too high for LoRA! This is catastrophic.
    # 2.- GPU modes 3 & 4: Learning rate 3e-5 is too high for large models with limited data
    # 3.- Missing parameters: No max_grad_norm in several configs(allowsgradientexplosion)
    # 4.- Adafactor optimizer: Can be unstable with certain model architectures
    # 5.- No adam_epsilon: Missing numerical stability parameter
    # Safe training configuration to prevent segfaults
    base_training_config = {
        'output_dir': model_dir,
        'overwrite_output_dir': True,
        'remove_unused_columns': False,
        'dataloader_pin_memory': False,
        'dataloader_num_workers': 0,
        'report_to': None,
        'adam_epsilon': 1e-6,
        'optim': "adamw_torch",
        'gradient_checkpointing': False,  # Disable to prevent memory issues
        'save_safetensors': True,
        'per_device_train_batch_size': 2,  # Larger batch size
        'gradient_accumulation_steps': 2,  # Fewer accumulation steps
        'fp16': False,  # Force FP32
        'use_cpu': False if torch.cuda.is_available() else True
    }
    
    if finetune_method == "lora":
        # LoRA configuration - effective learning
        training_args = TrainingArguments(
            num_train_epochs=5,  # More epochs for LoRA
            learning_rate=1e-4,  # Standard LoRA learning rate
            warmup_steps=20,
            logging_steps=10,
            save_steps=50,
            eval_strategy="steps",
            eval_steps=50,
            save_total_limit=1,
            max_grad_norm=1.0,  # Standard gradient clipping
            **base_training_config
        )
    else:
        # Optimized full fine-tuning for better response quality
        if device_config['device_type'] == 'gpu':
            # GPU training - balanced approach for quality responses
            training_args = TrainingArguments(
                num_train_epochs=6,  # Sufficient epochs for learning without overfitting
                learning_rate=2e-5,  # Conservative learning rate for stability
                warmup_steps=20,     # Proper warmup for stable training
                logging_steps=5,
                save_steps=25,
                eval_strategy="no",
                save_total_limit=1,
                max_grad_norm=1.0,   # Standard gradient clipping
                weight_decay=0.01,   # Proper regularization
                **base_training_config
            )
        else:
            # CPU training - effective configuration
            training_args = TrainingArguments(
                num_train_epochs=2,  # More epochs for CPU
                learning_rate=2e-5,  # Higher learning rate for CPU
                warmup_steps=20,
                logging_steps=20,
                save_steps=100,
                eval_strategy="steps",
                eval_steps=100,
                save_total_limit=1,
                max_grad_norm=0.5,  # Moderate gradient clipping
                **base_training_config
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
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[TrainingHealthMonitor(patience=3)]
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
    # MODEL TRAINING WITH STABILITY CHECKS
    # =============================================================================
    
    print("[INFO] Starting model training...")
    print(f"[INFO] This may take 15-60 minutes depending on your hardware")
    
    # Pre-training model validation
    def validate_model_weights(model, stage="pre-training"):
        """Check for NaN/inf values in model weights"""
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"[ERROR] NaN detected in {name} at {stage}")
                return False
            if torch.isinf(param).any():
                print(f"[ERROR] Inf detected in {name} at {stage}")
                return False
        print(f"[INFO] Model weights validated at {stage}")
        return True
    
    # Validate model before training
    if not validate_model_weights(model, "pre-training"):
        print("[ERROR] Model has corrupted weights before training!")
        return
    
    try:
        # Start training process with validation
        trainer.train()
        
        # Validate model after training
        if not validate_model_weights(model, "post-training"):
            print("[ERROR] Training produced corrupted weights!")
            print("[INFO] This indicates numerical instability during training")
            return
        
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
                        # Ensure clean input prompt
                        import sys
                        sys.stdout.flush()
                        sys.stderr.flush()
                        
                        choice = input(f"Select option (1-8) [{default_choice}]: ").strip()
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
                            training_args.max_grad_norm = 0.05
                            
                            # Recreate model with memory constraints using utility function
                            del model
                            model = recreate_model_with_config(base_model, {
                                'torch_dtype': torch.float32,
                                'device_map': "auto",
                                'low_cpu_mem_usage': True,
                                'max_memory': {0: "4GB"}
                            })
                            
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
                            training_args.max_grad_norm = 0.01
                            
                            # Recreate model with extreme constraints using utility function
                            del model
                            model = recreate_model_with_config(base_model, {
                                'torch_dtype': torch.float32,
                                'device_map': "auto",
                                'low_cpu_mem_usage': True,
                                'max_memory': {0: "2GB"}
                            })
                            
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
                            # Recreate model for CPU using utility function
                            del model
                            model = recreate_model_with_config(base_model, {
                                'torch_dtype': torch.float32,
                                'low_cpu_mem_usage': True,
                                'use_cache': False
                            }).cpu()
                            training_args.use_cpu = True
                            training_args.fp16 = False
                            training_args.per_device_train_batch_size = 1
                            training_args.gradient_accumulation_steps = 8
                            training_args.max_grad_norm = 0.1
                            
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
                training_args.max_grad_norm = 0.05
                
                # Recreate model with memory limit using utility function
                del model
                model = recreate_model_with_config(base_model, {
                    'torch_dtype': torch.float32,
                    'device_map': "auto",
                    'low_cpu_mem_usage': True,
                    'max_memory': {0: "4GB"}
                })
                
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
            # Use unified error handling menu
            print("\n[INFO] Training Error Detected!")
            
            import sys
            if sys.stdin.isatty():
                result = handle_training_error_menu(model, base_model, training_args, 
                                                   {'trainer': trainer, 'data_collator': data_collator, 
                                                    'train_dataset': train_dataset, 'eval_dataset': eval_dataset})
                if result == "restart":
                    return train_and_upload_model(dataset_dict, auth_token, username)
                elif result == True:
                    # Training succeeded, continue to model saving
                    pass
                else:
                    return  # Exit training
            else:
                print("[INFO] Non-interactive mode: This might be due to:")
                print("  - Corrupted training data")
                print("  - Hardware compatibility issues")
                print("  - Model configuration problems")
                print("[INFO] Run interactively for more options: python main.py")
                return
    
    # Save final model with enhanced stability for large models
    print("[INFO] Saving trained model...")
    
    try:
        # Clear GPU memory before saving to prevent segfaults
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Final validation before saving
        if validate_model_weights(model, "pre-save"):
            # Save with error handling for large models
            try:
                trainer.save_model()
                tokenizer.save_pretrained(model_dir)
                
                # Save training metrics
                if hasattr(trainer.state, 'log_history'):
                    import json
                    with open(f"{model_dir}/training_log.json", "w") as f:
                        json.dump(trainer.state.log_history, f, indent=2)
                
                print(f"[SUCCESS] Model saved to {model_dir}")
                
            except Exception as save_error:
                print(f"[ERROR] Model saving failed: {save_error}")
                print("[INFO] Attempting alternative save method...")
                
                # Alternative save method for large models
                model.save_pretrained(model_dir, safe_serialization=True)
                tokenizer.save_pretrained(model_dir)
                print(f"[SUCCESS] Model saved with alternative method to {model_dir}")
                
        else:
            print("[ERROR] Training produced corrupted model!")
            print("[INFO] Saving clean base model instead...")
            
            # Save clean base model as fallback
            clean_model = AutoModelForCausalLM.from_pretrained(base_model)
            clean_tokenizer = AutoTokenizer.from_pretrained(base_model)
            if clean_tokenizer.pad_token is None:
                clean_tokenizer.pad_token = clean_tokenizer.eos_token
            
            clean_model.save_pretrained(model_dir)
            clean_tokenizer.save_pretrained(model_dir)
            print(f"[INFO] Clean base model saved to {model_dir}")
            print("[WARNING] Model needs retraining with more stable parameters")
            
    except Exception as critical_error:
        print(f"[CRITICAL] Model saving completely failed: {critical_error}")
        print("[INFO] This may be due to memory issues with large models")
        return


    # =============================================================================
    # MODEL UPLOAD TO HUGGING FACE HUB
    # =============================================================================
    
    # Clear model from memory before upload to prevent segfaults
    print("[INFO] Clearing model from memory before upload...")
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    import gc
    gc.collect()
    
    if auth_token:
        print("[INFO] Uploading model to Hugging Face Hub...")
        try:
            upload_model_to_hf(
                model_dir=model_dir,
                model_id=model_id,
                auth_token=auth_token
            )
            print(f"[SUCCESS] Model uploaded to https://huggingface.co/{model_id}")
        except Exception as upload_error:
            print(f"[ERROR] Model upload failed: {upload_error}")
            print(f"[INFO] Manual upload: python upload_model.py --model-dir {model_dir} --model-id {model_id}")
    else:
        print("[INFO] No Hugging Face token provided - model saved locally only")
        
        # Generate local model card
        print("[INFO] Generating local model card...")
        try:
            generate_and_upload_model_card(
                model_id="local/jvm_troubleshooting_model",
                auth_token=None,
                base_model=base_model,
                finetune_method=finetune_method,
                train_size=len(dataset_dict['train']),
                test_size=len(dataset_dict['test'])
            )
        except Exception as card_error:
            print(f"[WARNING] Model card generation failed: {card_error}")

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
            
            # Quick quality check on existing dataset
            good_format = 0
            quality_issues = []
            total_examples = len(dataset_dict['train'])
            
            for i in range(total_examples):
                text = dataset_dict['train'][i]['text']
                if '### Human:' in text and '### Assistant:' in text:
                    good_format += 1
                    
                    # Check for technical accuracy issues
                    if 'urologist David Carr' in text:
                        quality_issues.append(f"Sample {i}: Contains fake expert name")
                    if 'tuned ForThreadExecutionPatterns' in text:
                        quality_issues.append(f"Sample {i}: Contains fake JVM parameter")
                    if 'AnalyzingGarbage collected Logs' in text:
                        quality_issues.append(f"Sample {i}: Contains fake tool name")
            
            format_percentage = (good_format / total_examples) * 100
            print(f"[INFO] Dataset quality: {format_percentage:.1f}% proper Q&A format")
            
            if quality_issues:
                print(f"[WARNING] Found {len(quality_issues)} technical accuracy issues in existing dataset")
                print("[RECOMMENDATION] Set OVERWRITE_DATASET=true to regenerate with improved prompts")
                
                # Offer detailed inspection
                import sys
                if sys.stdin.isatty():
                    try:
                        inspect_choice = input("\nRun detailed dataset inspection? (y/n) [n]: ").strip().lower()
                        if inspect_choice == 'y':
                            from inspect_dataset import inspect_dataset
                            inspect_dataset()
                    except (EOFError, KeyboardInterrupt):
                        pass
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
            
            # Comprehensive dataset quality inspection
            print("[INFO] Performing comprehensive dataset quality inspection...")
            good_format = 0
            total_examples = len(dataset_dict['train'])
            quality_issues = []
            length_stats = []
            
            for i in range(total_examples):
                text = dataset_dict['train'][i]['text']
                if '### Human:' in text and '### Assistant:' in text:
                    good_format += 1
                    assistant_part = text.split('### Assistant:')[-1].strip()
                    length_stats.append(len(assistant_part))
                    
                    # Enhanced technical accuracy checks
                    if any(fake in text for fake in ['urologist David Carr', 'Dr. Smith', 'Professor Johnson']):
                        quality_issues.append(f"Sample {i}: Contains fake expert name")
                    if any(fake in text for fake in ['tuned ForThreadExecutionPatterns', 'AnalyzingGarbage collected Logs', 'JVMProfiler Pro']):
                        quality_issues.append(f"Sample {i}: Contains fake tool/parameter")
                    if len(assistant_part) < 100:
                        quality_issues.append(f"Sample {i}: Answer too short ({len(assistant_part)} chars)")
                    if not any(tool in text.lower() for tool in ['jstat', 'jmap', 'jconsole', 'visualvm', 'gc', 'heap', 'memory', 'jvm']):
                        quality_issues.append(f"Sample {i}: Missing JVM technical content")
                    if text.count('### Human:') > 1 or text.count('### Assistant:') > 1:
                        quality_issues.append(f"Sample {i}: Multiple Q&A pairs in single sample")
            
            format_percentage = (good_format / total_examples) * 100
            avg_length = sum(length_stats) / len(length_stats) if length_stats else 0
            
            print(f"[INFO] Dataset Quality Report:")
            print(f"  • Format: {good_format}/{total_examples} examples ({format_percentage:.1f}%) proper Q&A format")
            print(f"  • Length: Average answer length {avg_length:.0f} characters")
            print(f"  • Issues: {len(quality_issues)} quality issues detected")
            
            # Detailed quality report
            if quality_issues:
                print(f"\n[WARNING] Quality Issues Found:")
                issue_types = {}
                for issue in quality_issues:
                    issue_type = issue.split(':')[1].strip()
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                
                for issue_type, count in issue_types.items():
                    print(f"  • {issue_type}: {count} samples")
                
                if len(quality_issues) > 10:
                    print(f"\n[INFO] First 10 specific issues:")
                    for issue in quality_issues[:10]:
                        print(f"    - {issue}")
                    print(f"    ... and {len(quality_issues) - 10} more")
                else:
                    print(f"\n[INFO] Specific issues:")
                    for issue in quality_issues:
                        print(f"    - {issue}")
            else:
                print("[SUCCESS] No technical accuracy issues detected")
            
            # Quality assessment
            if format_percentage >= 95 and avg_length >= 150 and len(quality_issues) == 0:
                print("[SUCCESS] ✅ EXCELLENT dataset quality - Ready for high-quality training")
            elif format_percentage >= 90 and avg_length >= 100 and len(quality_issues) <= 5:
                print("[SUCCESS] ✅ GOOD dataset quality - Suitable for training")
            elif format_percentage >= 80:
                print("[WARNING] ⚠️ MODERATE dataset quality - Training may produce mixed results")
            else:
                print("[ERROR] ❌ POOR dataset quality - Recommend regenerating dataset")
                
            # Interactive inspection option
            import sys
            if sys.stdin.isatty() and quality_issues:
                try:
                    inspect_choice = input("\nRun detailed dataset inspection tool? (y/n) [n]: ").strip().lower()
                    if inspect_choice == 'y':
                        print("[INFO] Running detailed inspection...")
                        import subprocess
                        subprocess.run(["python", "inspect_dataset.py"])
                except (EOFError, KeyboardInterrupt):
                    pass
            
            # Show sample to verify content quality
            if total_examples > 0:
                sample_text = dataset_dict['train'][0]['text']
                print(f"\n[INFO] Sample Q&A Preview:")
                print(f"{'='*60}")
                if len(sample_text) > 400:
                    print(f"{sample_text[:400]}...")
                else:
                    print(sample_text)
                print(f"{'='*60}")
                print(f"Sample length: {len(sample_text)} characters")
            
        except Exception as e:
            print(f"[ERROR] Failed to create dataset: {e}")
            return
        
        # Step 4: Upload dataset to Hugging Face Hub
        if auth_token:
            upload_dataset_to_hf(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                auth_token=auth_token
            )
        else:
            print("[INFO] No Hugging Face token - dataset saved locally only")
    
    # =============================================================================
    # MODEL TRAINING (OPTIONAL)
    # =============================================================================
    
    if train_model:
        print("[INFO] Starting model training pipeline...")
        
        # Show GPU info before training
        check_gpu()
        
        if auth_token:
            # Validate HF connection before training
            if not check_hf_connection():
                print("[ERROR] Hugging Face connection failed - training locally only")
                auth_token = None
                train_and_upload_model(dataset_dict, None, "local")
            else:
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
    
    # Check for trained model
    model_exists = os.path.exists("./models/jvm_troubleshooting_model")
    if model_exists:
        print("✅ Model: Trained and saved locally")
        
        # Interactive testing menu
        print("\n🧪 Available testing options:")
        print("1. Interactive testing with conversation memory (test_model.py)")
        print("2. Quick batch testing (quick_test.py)")
        print("3. Skip testing")
        
        import sys
        if sys.stdin.isatty():
            try:
                choice = input("\nChoose testing option (1-3) [3]: ").strip()
                if not choice:
                    choice = '3'
                
                if choice == '1':
                    print("🚀 Starting interactive testing...")
                    import subprocess
                    subprocess.run(["python", "test_model.py"])
                elif choice == '2':
                    print("🚀 Running quick batch validation...")
                    import subprocess
                    subprocess.run(["python", "quick_test.py"])
                elif choice == '3':
                    print("⏭️ Skipping testing")
                else:
                    print("⏭️ Skipping testing")
            except (EOFError, KeyboardInterrupt):
                print("\n⏭️ Skipping testing")
        else:
            print("\n🧪 Test your model:")
            print("   python test_model.py    # Interactive testing")
            print("   python quick_test.py    # Batch validation")
    elif train_model:
        print("⚠️ Model training was enabled but model not found at expected location")
        print("💡 Check training logs above for errors")
    
    print("\n📚 Next steps:")
    print("   • Review generated Q&A pairs in dataset/")
    print("   • Test model performance with sample questions")
    print("   • Fine-tune training parameters if needed")
    print("   • Deploy model for production use")
    
    if not train_model:
        print("\n🔧 Want to train a model?")
        print("   • Set TRAIN_MODEL=true in .env to enable training")
    elif not model_exists:
        print("\n🔧 Training Issues?")
        print("   • Run: python gpu_cleanup.py (for GPU memory issues)")
        print("   • Check training logs above for errors")
        print("   • Try: python model_utils.py recover")
    
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