#!/usr/bin/env python3
"""
Model Card Generator for JVM Troubleshooting Model

This utility generates comprehensive model cards for the trained JVM troubleshooting
model. Model cards provide essential documentation about the model's purpose,
training process, performance, and usage guidelines.

Features:
- Automated model card generation from templates
- Training metadata integration
- Hugging Face Hub upload functionality
- Local documentation creation
- Standardized model documentation format

Usage:
    python create_model_card.py

Model Card Contents:
- Model description and purpose
- Training methodology and parameters
- Dataset information and statistics
- Performance metrics and benchmarks
- Usage examples and code snippets
- Limitations and ethical considerations

Author: CesarChaMal
License: MIT
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os
from huggingface_hub import HfApi  # Hugging Face Hub API client

# =============================================================================
# MODEL CARD GENERATION FUNCTIONS
# =============================================================================

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
    """
    Generate comprehensive model card content with training details.
    
    This function creates a detailed model card by filling in a template
    with specific training parameters and model information. The model card
    serves as documentation for users and provides transparency about the
    model's capabilities and limitations.
    
    Args:
        base_model (str): Base model used for fine-tuning
        finetune_method (str): Fine-tuning method ('full' or 'lora')
        ai_provider (str): AI provider used for dataset generation
        train_size (int): Number of training examples
        test_size (int): Number of test examples
        learning_rate (str): Learning rate used during training
        batch_size (int): Batch size used during training
        num_epochs (int): Number of training epochs
        warmup_steps (int): Number of warmup steps
        model_size (str): Model size (e.g., "117M", "345M")
        vocab_size (str): Vocabulary size
        training_time (str): Approximate training time
        
    Returns:
        str: Complete model card content in Markdown format
        
    Note:
        The function uses a template file (model_card_template.md) that contains
        placeholders for dynamic content. This ensures consistent formatting
        and comprehensive documentation across all model versions.
    """
    
    print("📝 Generating model card from template...")
    
    # =============================================================================
    # TEMPLATE LOADING AND VALIDATION
    # =============================================================================
    
    template_path = "model_card_template.md"
    
    # Load the model card template
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        print(f"✅ Loaded template from: {template_path}")
    except FileNotFoundError:
        print(f"❌ Template file not found: {template_path}")
        print("💡 Creating a basic template...")
        
        # Create a basic template if none exists
        template = create_basic_template()
        
        # Save the basic template for future use
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template)
        print(f"✅ Created basic template: {template_path}")
    
    # =============================================================================
    # TRAINING CONFIGURATION PROCESSING
    # =============================================================================
    
    # Determine training methodology description
    if finetune_method.lower() == "lora":
        training_regime = "LoRA (Low-Rank Adaptation)"
        training_description = "Parameter-efficient fine-tuning using LoRA adapters"
    else:
        training_regime = "Full fine-tuning"
        training_description = "Complete model parameter fine-tuning"
    
    # Format AI provider name
    ai_provider_formatted = ai_provider.upper() if ai_provider else "UNKNOWN"
    
    # Calculate total dataset size
    total_size = train_size + test_size
    
    # =============================================================================
    # TEMPLATE VARIABLE REPLACEMENT
    # =============================================================================
    
    print("🔄 Filling template with training metadata...")
    
    # Define all template replacements
    replacements = {
        # Model Information
        '{base_model}': base_model,
        '{model_size}': model_size,
        '{vocab_size}': vocab_size,
        
        # Training Configuration
        '{finetune_method}': training_regime,
        '{training_regime}': training_regime,
        '{training_description}': training_description,
        '{ai_provider}': ai_provider_formatted,
        
        # Dataset Information
        '{train_size}': str(train_size),
        '{test_size}': str(test_size),
        '{total_size}': str(total_size),
        
        # Training Hyperparameters
        '{learning_rate}': learning_rate,
        '{batch_size}': str(batch_size),
        '{num_epochs}': str(num_epochs),
        '{warmup_steps}': str(warmup_steps),
        '{training_time}': training_time,
        
        # Performance Metrics (can be expanded)
        '{accuracy}': "Not evaluated",  # Placeholder for future metrics
        '{perplexity}': "Not evaluated",  # Placeholder for future metrics
    }\n    
    # Apply all replacements to the template
    model_card = template
    for placeholder, value in replacements.items():
        model_card = model_card.replace(placeholder, value)
        
    # Validate that all placeholders were replaced
    remaining_placeholders = [line for line in model_card.split('\n') if '{' in line and '}' in line]
    if remaining_placeholders:
        print(f"⚠️  Warning: Some placeholders may not have been replaced:")
        for line in remaining_placeholders[:3]:  # Show first 3
            print(f"   {line.strip()}")
    
    print(f"✅ Model card generated ({len(model_card)} characters)")
    return model_card

def create_basic_template():
    """
    Create a basic model card template if none exists.
    
    This function generates a minimal but comprehensive model card template
    that includes all essential sections for model documentation.
    
    Returns:
        str: Basic model card template with placeholders
    """
    
    return '''# JVM Troubleshooting Model

## Model Description

This model is a fine-tuned conversational AI specifically trained for JVM (Java Virtual Machine) troubleshooting and performance optimization. It provides expert-level guidance on common JVM issues, memory management, garbage collection, and performance tuning.

## Model Details

- **Base Model**: {base_model}
- **Model Size**: {model_size} parameters
- **Vocabulary Size**: {vocab_size} tokens
- **Fine-tuning Method**: {training_regime}
- **Training Data**: Generated using {ai_provider}

## Training Details

### Dataset
- **Training Examples**: {train_size}
- **Test Examples**: {test_size}
- **Total Dataset Size**: {total_size}
- **Data Source**: PDF-extracted JVM troubleshooting content

### Training Configuration
- **Training Method**: {training_description}
- **Learning Rate**: {learning_rate}
- **Batch Size**: {batch_size}
- **Epochs**: {num_epochs}
- **Warmup Steps**: {warmup_steps}
- **Training Time**: {training_time}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model
tokenizer = AutoTokenizer.from_pretrained("CesarChaMal/jvm_troubleshooting_model")
model = AutoModelForCausalLM.from_pretrained("CesarChaMal/jvm_troubleshooting_model")

# Generate response
input_text = "### Human: What causes OutOfMemoryError in JVM?\\n### Assistant:"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Intended Use

This model is designed for:
- JVM troubleshooting assistance
- Performance optimization guidance
- Memory management advice
- Garbage collection analysis
- Educational purposes for Java developers

## Limitations

- Specialized for JVM-related topics only
- May not have knowledge of the latest JVM versions
- Responses should be verified by experienced developers
- Not suitable for production-critical decisions without human oversight

## Ethical Considerations

This model provides technical guidance and should not be used as the sole source for critical system decisions. Always validate recommendations with official documentation and experienced professionals.
'''

# =============================================================================
# HUGGING FACE HUB UPLOAD FUNCTIONS
# =============================================================================

def upload_model_card(model_id, model_card_content, auth_token):
    """
    Upload the generated model card to Hugging Face Hub.
    
    This function handles the upload process for model cards, including
    local saving and remote upload to the Hugging Face Hub. It ensures
    the model has proper documentation available to users.
    
    Args:
        model_id (str): Hugging Face model repository ID
        model_card_content (str): Complete model card content
        auth_token (str): Hugging Face authentication token
        
    Returns:
        bool: True if upload successful, False otherwise
        
    Note:
        The model card is saved locally as README.md in the model directory
        and then uploaded to the Hugging Face Hub as the repository README.
    """
    
    print("📤 Uploading model card to Hugging Face Hub...")
    
    try:
        # =============================================================================
        # LOCAL SAVING
        # =============================================================================
        
        # Ensure model directory exists
        model_dir = "./models/jvm_troubleshooting_model"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model card locally as README.md
        readme_path = os.path.join(model_dir, "README.md")
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)
        
        print(f"✅ Model card saved locally: {readme_path}")
        print(f"📊 File size: {len(model_card_content)} characters")
        
        # =============================================================================
        # HUGGING FACE HUB UPLOAD
        # =============================================================================
        
        print(f"🌐 Uploading to repository: {model_id}")
        
        # Initialize Hugging Face API
        api = HfApi()
        
        # Upload the README file to the model repository
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=model_id,
            token=auth_token,
            repo_type="model",
            commit_message="Add comprehensive model card with training details"
        )
        
        print(f"✅ Model card uploaded successfully!")
        print(f"🔗 View at: https://huggingface.co/{model_id}")
        
        return True
        
    except Exception as e:
        # Handle various upload errors with specific guidance
        error_msg = str(e).lower()
        
        print(f"❌ Failed to upload model card")
        print(f"🔍 Error: {e}")
        
        # Provide specific troubleshooting based on error type
        if "401" in error_msg or "unauthorized" in error_msg:
            print("\n💡 Authentication Error:")
            print("   • Check your Hugging Face token")
            print("   • Verify token has write permissions")
            
        elif "404" in error_msg or "not found" in error_msg:
            print("\n💡 Repository Error:")
            print(f"   • Verify repository exists: {model_id}")
            print("   • Check repository name spelling")
            
        elif "403" in error_msg or "forbidden" in error_msg:
            print("\n💡 Permission Error:")
            print("   • Verify you have write access to the repository")
            print("   • Check if repository is private and you have access")
            
        else:
            print("\n💡 General troubleshooting:")
            print("   • Check internet connection")
            print("   • Verify file permissions")
            print("   • Try again in a few minutes")
        
        return False

# =============================================================================
# MAIN GENERATION AND UPLOAD FUNCTION
# =============================================================================

def generate_and_upload_model_card(
    model_id="CesarChaMal/jvm_troubleshooting_model",
    auth_token=None,
    **kwargs
):
    """
    Generate and optionally upload a complete model card.
    
    This is the main function that orchestrates the entire model card
    creation and upload process. It handles both local documentation
    and remote repository updates.
    
    Args:
        model_id (str): Hugging Face model repository ID
        auth_token (str): Hugging Face authentication token (optional)
        **kwargs: Additional parameters for model card generation
        
    Note:
        If no auth_token is provided, the model card will only be saved
        locally. This is useful for offline documentation or when the
        model hasn't been uploaded to Hugging Face yet.
    """
    
    print("📋 JVM Model Card Generator")
    print("=" * 40)
    
    # =============================================================================
    # MODEL CARD GENERATION
    # =============================================================================
    
    print("🎯 Generating comprehensive model documentation...")
    
    # Create model card content with provided parameters
    model_card = create_model_card(**kwargs)
    
    # =============================================================================
    # UPLOAD OR LOCAL SAVE
    # =============================================================================
    
    if auth_token:
        # Upload to Hugging Face Hub
        print(f"🚀 Uploading to Hugging Face repository: {model_id}")
        success = upload_model_card(model_id, model_card, auth_token)
        
        if success:
            print("\n✨ Model card generation and upload completed!")
        else:
            print("\n⚠️  Model card generated but upload failed")
            print("💡 Check the error messages above for troubleshooting")
    else:
        # Save locally only
        print("💾 No authentication token provided - saving locally only")
        
        model_dir = "./models/jvm_troubleshooting_model"
        os.makedirs(model_dir, exist_ok=True)
        readme_path = os.path.join(model_dir, "README.md")
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        
        print(f"✅ Model card saved to: {readme_path}")
        print("💡 To upload later, provide HUGGING_FACE_HUB_TOKEN")
        
    print("\n📚 Model card includes:")
    print("   • Model description and purpose")
    print("   • Training methodology and parameters")
    print("   • Dataset information and statistics")
    print("   • Usage examples and code snippets")
    print("   • Limitations and ethical considerations")

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for the model card generator.
    
    Demonstrates usage with example parameters and provides a template
    for integration with the main training pipeline.
    """
    
    print("🧪 Model Card Generator - Example Usage")
    print("=" * 50)
    
    # Example usage with typical training parameters
    generate_and_upload_model_card(
        # Model identification
        model_id="CesarChaMal/jvm_troubleshooting_model",
        
        # Training configuration
        base_model="microsoft/DialoGPT-medium",
        finetune_method="full",
        ai_provider="ollama",
        
        # Dataset information
        train_size=100,
        test_size=50,
        
        # Training hyperparameters
        learning_rate="3e-5",
        batch_size=2,
        num_epochs=5,
        warmup_steps=100,
        
        # Model specifications
        model_size="345M",
        vocab_size="50257",
        training_time="~45 minutes"
    )
    
    print("\n💡 To integrate with training pipeline:")
    print("   Import this module in main.py and call generate_and_upload_model_card()")
    print("   with actual training parameters after model training completes.")