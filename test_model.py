#!/usr/bin/env python3
"""
Standalone JVM Troubleshooting Model Tester
Test your trained model with default questions and interactive chat.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_jvm_model():
    """Interactive testing interface for the JVM troubleshooting model with conversation memory."""
    
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
    
    # Conversation memory
    conversation_history = []
    max_history = 5
    
    # Try to load the model
    model_path = "./models/jvm_troubleshooting_model"
    
    if not os.path.exists(model_path):
        print("[ERROR] Model not found!")
        print(f"Expected location: {model_path}")
        print("\nPlease train a model first by running: python main.py")
        return
    
    try:
        print("[INFO] Loading JVM Troubleshooting Model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        print("[SUCCESS] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    print("\n" + "="*60)
    print("JVM TROUBLESHOOTING ASSISTANT (With Memory)")
    print("="*60)
    print("\nAvailable Commands:")
    print("  - 'quit', 'exit', 'q' - Exit the assistant")
    print("  - 'help' - Show this help message")  
    print("  - 'examples' - Show example questions")
    print("  - 'defaults' - Test with default questions")
    print("  - 'history' - Show conversation history")
    print("  - 'clear' - Clear conversation history")
    print("\nThis model remembers your conversation context!")
    
    while True:
        question = input("\nYour question: ").strip()
        
        # Handle commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Thanks for using the JVM Troubleshooting Assistant.")
            break
            
        elif question.lower() == 'help':
            print("\nHELP")
            print("Ask any question about JVM troubleshooting, performance, or debugging.")
            print("Commands: quit, exit, q, help, examples, defaults")
            continue
            
        elif question.lower() == 'examples':
            print("\nEXAMPLE QUESTIONS:")
            for i, q in enumerate(default_questions, 1):
                print(f"  {i:2d}. {q}")
            continue
            
        elif question.lower() == 'defaults':
            print("\nTESTING WITH DEFAULT QUESTIONS:")
            print("-" * 50)
            for i, q in enumerate(default_questions[:5], 1):  # Test first 5
                print(f"\n{i}. Question: {q}")
                response = generate_response_with_memory(model, tokenizer, q, conversation_history)
                print(f"   Answer: {response}")
                # Add to history
                conversation_history.append({"question": q, "answer": response})
                if len(conversation_history) > max_history:
                    conversation_history = conversation_history[-max_history:]
            print("-" * 50)
            continue
            
        elif question.lower() == 'history':
            if not conversation_history:
                print("\nNo conversation history yet.")
            else:
                print("\n=== CONVERSATION HISTORY ===")
                for i, exchange in enumerate(conversation_history, 1):
                    print(f"{i}. Q: {exchange['question']}")
                    print(f"   A: {exchange['answer']}")
                print("=" * 30)
            continue
            
        elif question.lower() == 'clear':
            conversation_history.clear()
            print("\nConversation history cleared.")
            continue
            
        elif not question:
            print("Please enter a question or 'quit' to exit.")
            continue
        
        # Generate response with conversation memory
        response = generate_response_with_memory(model, tokenizer, question, conversation_history)
        print(f"\nAssistant: {response}")
        
        # Add to conversation history
        conversation_history.append({"question": question, "answer": response})
        if len(conversation_history) > max_history:
            conversation_history = conversation_history[-max_history:]

def generate_response_with_memory(model, tokenizer, question, conversation_history):
    """Generate a response with conversation context."""
    try:
        # Build context from conversation history
        context_parts = []
        
        # Add previous exchanges (last 3)
        for exchange in conversation_history[-3:]:
            context_parts.append(f"### Human: {exchange['question']}")
            context_parts.append(f"### Assistant: {exchange['answer']}")
        
        # Add current question
        context_parts.append(f"### Human: {question}")
        context_parts.append("### Assistant:")
        
        input_text = "\n".join(context_parts)
        
        # Tokenize with attention mask
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024  # Increased for conversation context
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        # Decode and clean response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new assistant response
        if '### Assistant:' in response:
            parts = response.split('### Assistant:')
            assistant_response = parts[-1].strip()
        else:
            assistant_response = response[len(input_text):].strip()
        
        # Clean up response
        if assistant_response:
            # Remove incomplete sentences and repetitions
            sentences = assistant_response.split('.')
            clean_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10 and not sentence.startswith('###'):
                    clean_sentences.append(sentence)
                    if len(clean_sentences) >= 2:  # Limit to 2 sentences
                        break
            
            if clean_sentences:
                return '. '.join(clean_sentences) + '.'
            else:
                return "I need more specific information to provide a better answer."
        else:
            return "Could you please rephrase your question?"
            
    except Exception as e:
        return f"Error generating response: {e}"

def generate_response(model, tokenizer, question):
    """Legacy function for backward compatibility."""
    return generate_response_with_memory(model, tokenizer, question, [])

if __name__ == "__main__":
    test_jvm_model()