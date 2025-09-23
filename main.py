import os
import logging
import fitz
import ollama
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo, delete_repo
from huggingface_hub.utils import HfHubHTTPError
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
import torch
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def read_pdf_content(pdf_path):
    content_list = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            content_list.append(page.get_text())
    return content_list

def call_ai(query: str, provider: str = "ollama", model: str = None) -> str:
    if provider == "ollama":
        model = model or "cesarchamal/qa-expert"
        return call_ollama(query, model)
    elif provider == "openai":
        model = model or "gpt-4o-mini"
        return call_openai(query, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def call_ollama(query: str, model: str) -> str:
    logging.debug(f"Calling Ollama with query: {query}, model: {model}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    try:
        response = ollama.chat(model=model, messages=messages, stream=False)
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        return "No response or unexpected response structure."
    except Exception as e:
        logging.error(f"Ollama Error: {str(e)}")
        return "Error occurred while calling the Ollama API."

def call_openai(query: str, model: str) -> str:
    logging.debug(f"Calling OpenAI with query: {query}, model: {model}")
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI Error: {str(e)}")
        return "Error occurred while calling the OpenAI API."

def prompt_engineered_api(text: str, provider: str = "ollama", model: str = None):
    prompt = f"""
        I have the following content: {text}
        I want to create a question-answer content that has the following format:

        ### Human:
        ### Assistant:

        Make sure to write question and answer based on the content I provided.
    """
    return call_ai(prompt, provider, model)

def train_and_upload_model(dataset_dict, auth_token, username):
    print("[INFO] Starting model fine-tuning...")
    
    # Model configuration
    base_model = os.getenv('BASE_MODEL', 'microsoft/DialoGPT-small')
    finetune_method = os.getenv('FINETUNE_METHOD', 'full')
    model_name = "jvm_troubleshooting_model"
    model_id = f"{username}/{model_name}"
    
    # Create models directory
    os.makedirs(f"./models/{model_name}", exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)
    
    # Apply LoRA if selected
    if finetune_method == "lora":
        if not PEFT_AVAILABLE:
            print("[WARNING] PEFT not available. Installing...")
            os.system("pip install peft")
            print("[INFO] Please restart the script after installation.")
            return
        
        print("[INFO] Applying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("[INFO] Using full fine-tuning...")
    
    # Prepare dataset for training
    def preprocess_function(examples):
        # Process each text example
        texts = []
        for text in examples['text']:
            if isinstance(text, str) and text.strip():
                # Ensure text ends with EOS token
                formatted_text = text.strip()
                if not formatted_text.endswith(tokenizer.eos_token):
                    formatted_text += tokenizer.eos_token
                texts.append(formatted_text)
            else:
                # Fallback for invalid entries
                texts.append("### Human: What is JVM?\n### Assistant: JVM stands for Java Virtual Machine." + tokenizer.eos_token)
        
        # Tokenize all texts - ALWAYS use padding for consistency
        model_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
        
        # Create labels (copy of input_ids for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"][:]
        return model_inputs
    
    # Tokenize datasets
    print("[INFO] Preprocessing training data...")
    train_dataset = dataset_dict['train'].map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset_dict['train'].column_names
    )
    print("[INFO] Preprocessing evaluation data...")
    eval_dataset = dataset_dict['test'].map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset_dict['test'].column_names
    )
    
    # Training arguments (optimized for fine-tuning method)
    if finetune_method == "lora":
        # LoRA: More aggressive training since fewer parameters
        training_args = TrainingArguments(
            output_dir=f"./models/{model_name}",
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
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
        )
    else:
        # Full fine-tuning: Conservative approach
        training_args = TrainingArguments(
            output_dir=f"./models/{model_name}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=5e-5,
            warmup_steps=50,
            logging_steps=25,
            save_steps=250,
            eval_strategy="steps",
            eval_steps=250,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
        )
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Validate data before training
    print("[INFO] Validating training data...")
    print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
    print(f"[DEBUG] Sample train item keys: {list(train_dataset[0].keys())}")
    print(f"[DEBUG] Sample input_ids type: {type(train_dataset[0]['input_ids'])}")
    print(f"[DEBUG] Sample input_ids length: {len(train_dataset[0]['input_ids'])}")
    
    try:
        sample_batch = next(iter(trainer.get_train_dataloader()))
        print(f"[INFO] Sample batch keys: {list(sample_batch.keys())}")
        print(f"[INFO] Sample batch input_ids shape: {sample_batch['input_ids'].shape}")
        print(f"[INFO] Sample batch labels shape: {sample_batch['labels'].shape}")
    except Exception as e:
        print(f"[ERROR] Data validation failed: {e}")
        print(f"[DEBUG] Error type: {type(e)}")
        return
    
    # Train the model
    print("[INFO] Starting training...")
    try:
        trainer.train()
        print("[SUCCESS] Training completed!")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        print("[INFO] This might be due to memory constraints. Try reducing batch size or using full fine-tuning.")
        return
    
    # Save the model
    print("[INFO] Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(f"./models/{model_name}")
    
    # Generate model card
    try:
        from create_model_card import generate_model_card
        generate_model_card(
            model_name=model_name,
            base_model=base_model,
            finetune_method=finetune_method,
            train_size=len(train_dataset),
            test_size=len(eval_dataset)
        )
        print("[SUCCESS] Model card generated!")
    except Exception as e:
        print(f"[WARNING] Model card generation failed: {e}")
    
    # Upload to Hugging Face Hub
    if auth_token:
        try:
            print(f"[INFO] Uploading model to Hugging Face Hub as {model_id}...")
            api = HfApi()
            
            # Create repository
            try:
                create_repo(repo_id=model_id, token=auth_token, repo_type="model")
                print(f"[SUCCESS] Repository {model_id} created!")
            except HfHubHTTPError as e:
                if "already exists" in str(e):
                    print(f"[INFO] Repository {model_id} already exists")
                else:
                    raise e
            
            # Upload model files
            api.upload_folder(
                folder_path=f"./models/{model_name}",
                repo_id=model_id,
                token=auth_token,
                repo_type="model"
            )
            print(f"[SUCCESS] Model uploaded to https://huggingface.co/{model_id}")
            
        except Exception as e:
            print(f"[ERROR] Failed to upload model: {e}")
    else:
        print("[INFO] No Hugging Face token provided. Model saved locally only.")

def main():
    pdf_path = "jvm_troubleshooting_guide.pdf"
    dataset_name = "jvm_troubleshooting_guide"
    
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file '{pdf_path}' not found. Please add your PDF file.")
        return
    
    # Get configuration from environment
    ai_provider = os.getenv('AI_PROVIDER', 'ollama')
    ai_model = os.getenv('AI_MODEL')
    overwrite_dataset = os.getenv('OVERWRITE_DATASET', 'false').lower() == 'true'
    train_model = os.getenv('TRAIN_MODEL', 'false').lower() == 'true'
    auth_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    print(f"[INFO] Using AI provider: {ai_provider}")
    if ai_model:
        print(f"[INFO] Using model: {ai_model}")
    
    # Check if dataset already exists
    dataset_path = f"./dataset/{dataset_name}"
    if os.path.exists(dataset_path) and not overwrite_dataset:
        print(f"[INFO] Dataset already exists at {dataset_path}")
        print("[INFO] Loading existing dataset...")
        dataset_dict = load_from_disk(dataset_path)
        print(f"[SUCCESS] Loaded dataset with {len(dataset_dict['train'])} training and {len(dataset_dict['test'])} test examples")
    else:
        if os.path.exists(dataset_path):
            print(f"[INFO] Overwriting existing dataset at {dataset_path}")
        
        # Read PDF content
        print(f"[INFO] Reading PDF content from {pdf_path}...")
        content_list = read_pdf_content(pdf_path)
        print(f"[SUCCESS] Extracted content from {len(content_list)} pages")
        
        # Generate Q&A pairs
        print("[INFO] Generating question-answer pairs...")
        qa_pairs = []
        
        for i, content in enumerate(tqdm(content_list, desc="Processing pages")):
            if content.strip():
                try:
                    qa_response = prompt_engineered_api(content, ai_provider, ai_model)
                    if qa_response and "### Human:" in qa_response and "### Assistant:" in qa_response:
                        qa_pairs.append(qa_response)
                        print(f"[DEBUG] Generated Q&A for page {i+1}")
                    else:
                        print(f"[WARNING] Invalid Q&A format for page {i+1}")
                except Exception as e:
                    print(f"[ERROR] Failed to generate Q&A for page {i+1}: {e}")
        
        if not qa_pairs:
            print("[ERROR] No valid Q&A pairs generated. Exiting.")
            return
        
        print(f"[SUCCESS] Generated {len(qa_pairs)} Q&A pairs")
        
        # Create dataset
        print("[INFO] Creating dataset...")
        dataset = Dataset.from_dict({"text": qa_pairs})
        
        # Split dataset
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'test': train_test_split['test']
        })
        
        # Save dataset locally
        os.makedirs(f"./dataset", exist_ok=True)
        dataset_dict.save_to_disk(dataset_path)
        print(f"[SUCCESS] Dataset saved to {dataset_path}")
        
        # Upload to Hugging Face Hub
        if auth_token:
            try:
                username = HfApi(token=auth_token).whoami()["name"]
                repo_id = f"{username}/{dataset_name}"
                
                print(f"[INFO] Uploading dataset to Hugging Face Hub as {repo_id}...")
                
                # Create repository
                try:
                    create_repo(repo_id=repo_id, token=auth_token, repo_type="dataset")
                    print(f"[SUCCESS] Repository {repo_id} created!")
                except HfHubHTTPError as e:
                    if "already exists" in str(e):
                        print(f"[INFO] Repository {repo_id} already exists")
                    else:
                        raise e
                
                # Push dataset
                dataset_dict.push_to_hub(repo_id, token=auth_token)
                print(f"[SUCCESS] Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
                
            except Exception as e:
                print(f"[ERROR] Failed to upload dataset: {e}")
        else:
            print("[INFO] No Hugging Face token provided. Dataset saved locally only.")
    
    # Train model if requested
    if train_model:
        if auth_token:
            username = HfApi(token=auth_token).whoami()["name"]
            train_and_upload_model(dataset_dict, auth_token, username)
        else:
            print("[WARNING] No Hugging Face token provided. Training locally only.")
            train_and_upload_model(dataset_dict, None, "local")
    else:
        print("[INFO] Model training skipped. Set TRAIN_MODEL=true to enable training.")
    
    print("[SUCCESS] Pipeline completed!")

if __name__ == "__main__":
    main()ht be due to memory constraints. Try reducing batch size or using LoRA fine-tuning.")
        returnht be due to memory constraints. Try reducing batch size or using full fine-tuning.")
        return
    
    # Save model locally
    model_path = f"./models/{model_name}"
    try:
        # For LoRA models, we need to merge and save differently
        if finetune_method == "lora":
            print("[INFO] Merging LoRA weights and saving...")
            # Merge LoRA weights into base model
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        else:
            # Save using trainer for full fine-tuning
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)
        
        # Verify the model was saved correctly
        if os.path.exists(os.path.join(model_path, "config.json")):
            print(f"[SUCCESS] Model saved locally to: {model_path}")
        else:
            print(f"[WARNING] Model saved but config.json missing. Saving manually...")
            if finetune_method != "lora":
                model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        return
    
    # Upload to Hugging Face
    try:
        print("[INFO] Uploading model to Hugging Face...")
        if finetune_method == "lora":
            # Upload the merged model
            merged_model.push_to_hub(model_id, token=auth_token)
        else:
            model.push_to_hub(model_id, token=auth_token)
        tokenizer.push_to_hub(model_id, token=auth_token)
        print(f"[SUCCESS] Model uploaded to: https://huggingface.co/{model_id}")
        
        # Generate and upload model card
        try:
            from create_model_card import generate_and_upload_model_card
            
            # Get training details
            ai_provider = os.getenv('AI_PROVIDER', 'ollama')
            train_size = len(dataset_dict['train'])
            test_size = len(dataset_dict['test'])
            
            # Model size mapping
            model_sizes = {
                'microsoft/DialoGPT-small': '117M',
                'microsoft/DialoGPT-medium': '345M', 
                'microsoft/DialoGPT-large': '762M',
                'distilgpt2': '82M',
                'gpt2': '124M'
            }
            model_size = model_sizes.get(base_model, '117M')
            
            # Training hyperparameters
            lr = training_args.learning_rate
            batch_size = training_args.per_device_train_batch_size
            epochs = training_args.num_train_epochs
            warmup = training_args.warmup_steps
            
            generate_and_upload_model_card(
                model_id=model_id,
                auth_token=auth_token,
                base_model=base_model,
                finetune_method=finetune_method,
                ai_provider=ai_provider,
                train_size=train_size,
                test_size=test_size,
                learning_rate=str(lr),
                batch_size=batch_size,
                num_epochs=int(epochs),
                warmup_steps=warmup,
                model_size=model_size
            )
        except Exception as card_error:
            print(f"[WARNING] Failed to upload model card: {card_error}")
            
    except Exception as e:
        print(f"[ERROR] Failed to upload model: {e}")
    
    # Test the model
    test_existing = input("\nDo you want to test the trained model? (y/N): ").lower().strip()
    if test_existing in ['y', 'yes']:
        if finetune_method == "lora":
            test_model_interactive(merged_model, tokenizer)
        else:
            test_model_interactive(model, tokenizer)

def test_model_interactive(model, tokenizer):
    print("\n[INFO] Testing the trained model...")
    
    # Default test questions
    default_questions = [
        "What are common JVM memory issues?",
        "How do I troubleshoot OutOfMemoryError?",
        "What JVM parameters should I tune for performance?",
        "How do I analyze garbage collection logs?",
        "What causes high CPU usage in JVM applications?",
        "How do I debug memory leaks in Java applications?",
        "What are the best practices for JVM monitoring?",
        "How do I optimize JVM startup time?",
        "What tools can I use for JVM profiling?",
        "How do I handle StackOverflowError?",
        "What are the differences between heap and non-heap memory?"
    ]
    
    print("\n=== JVM Troubleshooting Assistant ===")
    print("Commands:")
    print("  'quit', 'exit', 'q' - Exit the assistant")
    print("  'help' - Show this help message")
    print("  'examples' - Show example questions")
    print("\nTry asking questions about JVM troubleshooting!")
    
    model.eval()
    
    while True:
        question = input("\n> Your question: ").strip()
        
        # Handle commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Thanks for using the JVM Troubleshooting Assistant.")
            break
        elif question.lower() == 'help':
            print("\n=== Help ===")
            print("Ask any question about JVM troubleshooting, performance, or debugging.")
            print("Commands: quit, exit, q, help, examples")
            continue
        elif question.lower() == 'examples':
            print("\n=== Example Questions ===")
            for i, q in enumerate(default_questions, 1):
                print(f"{i}. {q}")
            continue
        elif not question:
            print("Please enter a question or 'quit' to exit.")
            continue
            
        # Format input with proper attention mask
        input_text = f"### Human: {question}\n### Assistant:"
        inputs = tokenizer(
            input_text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Generate response
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if '### Assistant:' in response:
                assistant_response = response.split('### Assistant:')[-1].strip()
            else:
                assistant_response = response[len(input_text):].strip()
            
            # Clean up the response
            if assistant_response:
                # Remove any repeated patterns or incomplete sentences
                lines = assistant_response.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('###'):
                        clean_lines.append(line)
                        if len(clean_lines) >= 3:  # Limit response length
                            break
                
                final_response = ' '.join(clean_lines) if clean_lines else "I'm not sure about that. Could you rephrase your question?"
                print(f"\nAssistant: {final_response}")
            else:
                print("\nAssistant: I'm not sure about that. Could you rephrase your question?")
                
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Please try a different question.")

# Check if dataset already exists
local_dataset_path = "./dataset/jvm_troubleshooting_guide"
overwrite_dataset = os.getenv('OVERWRITE_DATASET', 'false').lower() == 'true'

if os.path.exists(local_dataset_path) and not overwrite_dataset:
    print(f"[INFO] Dataset already exists at: {local_dataset_path}")
    response = input("Do you want to overwrite it? (y/N): ").lower().strip()
    if response in ['y', 'yes']:
        print("[INFO] Overwriting existing dataset")
        overwrite_dataset = True
    else:
        print("[INFO] Using existing dataset")
        dataset_dict = load_from_disk(local_dataset_path)
        print(dataset_dict)

if not os.path.exists(local_dataset_path) or overwrite_dataset:
    # Step 1: Read and preprocess PDF
    scraped_content = ' '.join(read_pdf_content("./jvm_troubleshooting_guide.pdf"))
    sentences = scraped_content.split('. ')
    logging.info(f"Total segments: {len(sentences)}")

    # Step 2: Generate QA pairs
    ai_provider = os.getenv('AI_PROVIDER', 'ollama')  # ollama or openai
    ai_model = os.getenv('AI_MODEL')  # optional custom model

    raw_content_for_train = []
    for sentence in tqdm(sentences, desc=f"Generating Q&A using {ai_provider}"):
        raw_content_for_train.append(prompt_engineered_api(sentence, ai_provider, ai_model))

    # Step 3: Create datasets
    train_data = {'text': raw_content_for_train[0:100]}
    test_data = {'text': raw_content_for_train[100:]}
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'test': Dataset.from_dict(test_data)
    })
    
    # Save dataset locally
    dataset_dict.save_to_disk(local_dataset_path)
    logging.info(f"[SUCCESS] Dataset saved locally to: {local_dataset_path}")

print(dataset_dict)

# Step 4: Push to Hugging Face Hub
auth_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
repo_name = 'jvm_troubleshooting_guide'
username = 'CesarChaMal'
app_id = f"{username}/{repo_name}"
api = HfApi()



# Optional: delete existing repo if it exists
try:
    logging.info(f"Attempting to delete existing dataset repo: {app_id}")
    delete_repo(repo_id=app_id, token=auth_token, repo_type="dataset")
    logging.info(f"Deleted existing dataset: {app_id}")
except HfHubHTTPError as e:
    if e.response.status_code == 404:
        logging.info(f"No existing repo found: {app_id}, proceeding...")
    else:
        logging.warning(f"Could not delete repo: {e}")
except Exception as e:
    logging.warning(f"Could not delete repo: {e}")

# Recreate the repo
try:
    create_repo(repo_name, token=auth_token, private=False, repo_type="dataset")
    logging.info(f"Created dataset repo: {app_id}")
except HfHubHTTPError as e:
    if e.response.status_code == 409:
        logging.info(f"Repo {app_id} already exists, continuing...")
    else:
        logging.warning(f"Could not create repo: {e}")
except Exception as e:
    logging.warning(f"Could not create repo: {e}")



# Step 5: Push dataset
dataset = load_from_disk("./dataset/jvm_troubleshooting_guide")
logging.info(f"Pushing dataset to: {app_id}")
dataset.push_to_hub(app_id, token=auth_token)
print(f"[SUCCESS] Dataset successfully pushed to: https://huggingface.co/datasets/{app_id}")

# Step 6: Train model
train_model_env = os.getenv('TRAIN_MODEL', 'false').lower() == 'true'
if train_model_env:
    train_and_upload_model(dataset_dict, auth_token, username)
else:
    print("\nProcess completed successfully!")
    train_model = input("\nDo you want to train a model from this dataset? (y/N): ").lower().strip()
    if train_model in ['y', 'yes']:
        train_and_upload_model(dataset_dict, auth_token, username)
    else:
        # Offer to test existing model if available
        model_path = "./models/jvm_troubleshooting_model"
        if os.path.exists(model_path):
            test_existing = input("\nDo you want to test the existing trained model? (y/N): ").lower().strip()
            if test_existing in ['y', 'yes']:
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    print("[INFO] Loading existing model...")
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    test_model_interactive(model, tokenizer)
                except Exception as e:
                    print(f"[ERROR] Failed to load existing model: {e}")
