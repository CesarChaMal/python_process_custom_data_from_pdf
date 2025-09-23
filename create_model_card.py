#!/usr/bin/env python3
"""
Model Card Generator
Creates and uploads model cards to Hugging Face Hub.
"""

import os
from huggingface_hub import HfApi

def create_model_card(
    base_model="microsoft/DialoGPT-small",
    finetune_method="full",
    ai_provider="ollama",
    train_size=100,
    test_size=50,
    learning_rate="5e-5",
    batch_size=2,
    num_epochs=3,
    warmup_steps=50,
    model_size="117M",
    vocab_size="50257",
    training_time="~30 minutes"
):
    """Generate model card content with training details."""
    
    # Read template
    with open("model_card_template.md", "r", encoding="utf-8") as f:
        template = f.read()
    
    # Determine training regime
    training_regime = "Full fine-tuning" if finetune_method == "full" else "LoRA (Low-Rank Adaptation)"
    
    # Fill in template variables using string replacement
    model_card = template
    replacements = {
        '{base_model}': base_model,
        '{ai_provider}': ai_provider.upper(),
        '{train_size}': str(train_size),
        '{test_size}': str(test_size),
        '{finetune_method}': training_regime,
        '{training_regime}': training_regime,
        '{learning_rate}': learning_rate,
        '{batch_size}': str(batch_size),
        '{num_epochs}': str(num_epochs),
        '{warmup_steps}': str(warmup_steps),
        '{model_size}': model_size,
        '{vocab_size}': vocab_size,
        '{training_time}': training_time
    }
    
    for placeholder, value in replacements.items():
        model_card = model_card.replace(placeholder, value)
    
    return model_card

def upload_model_card(model_id, model_card_content, auth_token):
    """Upload model card to Hugging Face Hub."""
    try:
        # Save locally first
        model_dir = "./models/jvm_troubleshooting_model"
        readme_path = os.path.join(model_dir, "README.md")
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)
        
        print(f"[INFO] Model card saved locally to: {readme_path}")
        
        # Upload to Hugging Face
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=model_id,
            token=auth_token,
            repo_type="model"
        )
        
        print(f"[SUCCESS] Model card uploaded to: https://huggingface.co/{model_id}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to upload model card: {e}")
        return False

def generate_and_upload_model_card(
    model_id="CesarChaMal/jvm_troubleshooting_model",
    auth_token=None,
    **kwargs
):
    """Generate and upload model card with training details."""
    
    print("[INFO] Generating model card...")
    
    # Create model card content
    model_card = create_model_card(**kwargs)
    
    # Upload to Hugging Face
    if auth_token:
        upload_model_card(model_id, model_card, auth_token)
    else:
        # Save locally only
        model_dir = "./models/jvm_troubleshooting_model"
        readme_path = os.path.join(model_dir, "README.md")
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        
        print(f"[INFO] Model card saved locally to: {readme_path}")
        print("[INFO] No auth token provided, skipping Hugging Face upload")

if __name__ == "__main__":
    # Example usage
    generate_and_upload_model_card(
        base_model="microsoft/DialoGPT-small",
        finetune_method="full",
        ai_provider="ollama",
        train_size=100,
        test_size=50
    )