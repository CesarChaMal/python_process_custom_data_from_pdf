#!/usr/bin/env python3
"""
Standalone JVM Troubleshooting Model Tester
Test your trained model with default questions and interactive chat.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_jvm_model():
    """Interactive testing interface for the JVM troubleshooting model."""
    
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
    
    # Try to load the model
    model_path = "./models/jvm_troubleshooting_model"
    
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        print(f"Expected location: {model_path}")
        print("\nPlease train a model first by running: python main.py")
        return
    
    try:
        print("🔄 Loading JVM Troubleshooting Model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    print("\n" + "="*60)
    print("🔧 JVM TROUBLESHOOTING ASSISTANT")
    print("="*60)
    print("\n📋 Available Commands:")
    print("  • 'quit', 'exit', 'q' - Exit the assistant")
    print("  • 'help' - Show this help message")  
    print("  • 'examples' - Show example questions")
    print("  • 'defaults' - Test with default questions")
    print("\n💡 Try asking questions about JVM troubleshooting!")
    
    while True:
        question = input("\n🔧 Your question: ").strip()
        
        # Handle commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! Thanks for using the JVM Troubleshooting Assistant.")
            break
            
        elif question.lower() == 'help':
            print("\n📋 HELP")
            print("Ask any question about JVM troubleshooting, performance, or debugging.")
            print("Commands: quit, exit, q, help, examples, defaults")
            continue
            
        elif question.lower() == 'examples':
            print("\n💡 EXAMPLE QUESTIONS:")
            for i, q in enumerate(default_questions, 1):
                print(f"  {i:2d}. {q}")
            continue
            
        elif question.lower() == 'defaults':
            print("\n🧪 TESTING WITH DEFAULT QUESTIONS:")
            print("-" * 50)
            for i, q in enumerate(default_questions[:5], 1):  # Test first 5
                print(f"\n{i}. Question: {q}")
                response = generate_response(model, tokenizer, q)
                print(f"   Answer: {response}")
            print("-" * 50)
            continue
            
        elif not question:
            print("Please enter a question or 'quit' to exit.")
            continue
        
        # Generate response for user question
        response = generate_response(model, tokenizer, question)
        print(f"\n🤖 Assistant: {response}")

def generate_response(model, tokenizer, question):
    """Generate a response for the given question."""
    try:
        # Format input
        input_text = f"### Human: {question}\n### Assistant:"
        
        # Tokenize with attention mask
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
                max_new_tokens=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if '### Assistant:' in response:
            assistant_response = response.split('### Assistant:')[-1].strip()
        else:
            assistant_response = response[len(input_text):].strip()
        
        # Clean up response
        if assistant_response:
            # Remove incomplete sentences and repetitions
            sentences = assistant_response.split('.')
            clean_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Filter very short fragments
                    clean_sentences.append(sentence)
                    if len(clean_sentences) >= 2:  # Limit to 2 sentences
                        break
            
            if clean_sentences:
                return '. '.join(clean_sentences) + '.'
            else:
                return "I'm not sure about that. Could you rephrase your question?"
        else:
            return "I'm not sure about that. Could you rephrase your question?"
            
    except Exception as e:
        return f"Error generating response: {e}"

if __name__ == "__main__":
    test_jvm_model()