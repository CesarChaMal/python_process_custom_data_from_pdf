#!/usr/bin/env python3
"""
Quick Test Script for JVM Troubleshooting Model
Runs predefined questions through the trained model for quick validation.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def quick_test_model():
    """Quick batch testing with predefined JVM troubleshooting questions."""
    
    # Predefined test questions
    test_questions = [
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
    
    # Check if model exists
    model_path = "./models/jvm_troubleshooting_model"
    
    if not os.path.exists(model_path):
        print("ERROR: Model not found!")
        print(f"Expected location: {model_path}")
        print("\nPlease train a model first by running: python main.py")
        return
    
    try:
        print("Loading JVM Troubleshooting Model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    print("\n" + "="*80)
    print("QUICK TEST: JVM TROUBLESHOOTING MODEL")
    print("="*80)
    
    # Test each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i:2d}/11] Question: {question}")
        print("-" * 60)
        
        response = generate_response(model, tokenizer, question)
        print(f"Answer: {response}")
        
        if i < len(test_questions):
            print()
    
    print("\n" + "="*80)
    print("QUICK TEST COMPLETED")
    print("="*80)
    print("\nFor interactive testing, run: python test_model.py")

def generate_response(model, tokenizer, question):
    """Generate a response for the given question."""
    try:
        # Format input
        input_text = f"### Human: {question}\n### Assistant:"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if '### Assistant:' in response:
            assistant_response = response.split('### Assistant:')[-1].strip()
        else:
            assistant_response = response[len(input_text):].strip()
        
        # Clean up response
        if assistant_response:
            # Remove incomplete sentences and limit length
            sentences = assistant_response.split('.')
            clean_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    clean_sentences.append(sentence)
                    if len(clean_sentences) >= 2:  # Limit to 2 sentences for quick test
                        break
            
            if clean_sentences:
                return '. '.join(clean_sentences) + '.'
            else:
                return "I'm not sure about that specific question."
        else:
            return "I'm not sure about that specific question."
            
    except Exception as e:
        return f"Error generating response: {e}"

if __name__ == "__main__":
    quick_test_model()