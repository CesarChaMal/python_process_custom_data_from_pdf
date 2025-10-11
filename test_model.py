#!/usr/bin/env python3
"""
JVM Troubleshooting Model Interactive Tester

This script provides an interactive testing interface for the trained JVM
troubleshooting model. It includes conversation memory, default test questions,
and various commands for comprehensive model evaluation.

Features:
- Interactive chat with conversation memory
- Pre-defined JVM troubleshooting questions
- Command system (help, examples, defaults, etc.)
- Improved response generation with attention masks
- Clean conversation flow and error handling

Usage:
    python test_model.py

Commands:
    quit/exit/q - Exit the assistant
    help - Show help message
    examples - Show example questions
    defaults - Test with default questions
    history - Show conversation history
    clear - Clear conversation history

Author: CesarChaMal
License: MIT
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# INTERACTIVE MODEL TESTING FUNCTIONS
# =============================================================================

def test_jvm_model():
    """
    Main interactive testing interface for the JVM troubleshooting model.
    
    This function provides a comprehensive testing environment with:
    - Model loading and validation
    - Interactive chat interface
    - Command system for various operations
    - Conversation memory for context-aware responses
    - Error handling and user guidance
    """
    
    # =============================================================================
    # DEFAULT TEST QUESTIONS
    # =============================================================================
    # Comprehensive set of JVM troubleshooting questions covering:
    # - Memory management and OutOfMemoryError
    # - Performance tuning and optimization
    # - Garbage collection analysis
    # - Debugging and profiling tools
    # - Common error scenarios
    # =============================================================================
    
    default_questions = [
        "What are common JVM memory issues?",              # Memory fundamentals
        "How do I troubleshoot OutOfMemoryError?",         # Critical error handling
        "What JVM parameters should I tune for performance?", # Performance optimization
        "How do I analyze garbage collection logs?",        # GC analysis
        "What causes high CPU usage in JVM applications?",  # Performance debugging
        "How do I debug memory leaks in Java applications?", # Memory leak detection
        "What are the best practices for JVM monitoring?",  # Monitoring strategies
        "How do I optimize JVM startup time?",             # Startup optimization
        "What tools can I use for JVM profiling?",         # Profiling tools
        "How do I handle StackOverflowError?",             # Stack management
        "What are the differences between heap and non-heap memory?" # Memory types
    ]
    
    # =============================================================================
    # CONVERSATION MEMORY SETUP
    # =============================================================================
    # Initialize conversation history for context-aware responses
    # This allows the model to remember previous questions and provide
    # more coherent, contextual answers in multi-turn conversations
    # =============================================================================
    
    conversation_history = []
    max_history = 5  # Limit history to prevent token overflow
    
    # =============================================================================
    # MODEL LOADING AND VALIDATION
    # =============================================================================
    
    model_path = "./models/jvm_troubleshooting_model"
    
    # Check if trained model exists
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        print(f"📁 Expected location: {model_path}")
        print("\n💡 To fix this:")
        print("   1. Train a model: python main.py (with TRAIN_MODEL=true)")
        print("   2. Or download from Hugging Face Hub")
        print("   3. Or run model recovery: python model_utils.py recover")
        return
    
    # Load model and tokenizer with error handling
    try:
        print("🔄 Loading JVM Troubleshooting Model...")
        print("📦 This may take a moment for large models...")
        
        # Load tokenizer with proper configuration
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Load model with stable CPU settings
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.float32,
        #     device_map=None,
        #     low_cpu_mem_usage=True
        # )
        
        # Set model to evaluation mode
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"🔧 Model parameters: {model.num_parameters():,}")
        print(f"💾 Device: CPU (stable mode)")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Possible solutions:")
        print("   - Check if model files are complete")
        print("   - Try model recovery: python model_utils.py recover")
        print("   - Retrain the model: python main.py")
        return
    
    # =============================================================================
    # USER INTERFACE SETUP
    # =============================================================================
    
    # Display welcome message and instructions
    print("\n" + "="*70)
    print("🔧 JVM TROUBLESHOOTING ASSISTANT")
    print("="*70)
    print("\n🤖 AI Assistant trained on JVM troubleshooting knowledge")
    print("💭 Conversation memory enabled for better context understanding")
    print("\n📋 Available Commands:")
    print("  • 'quit', 'exit', 'q' - Exit the assistant")
    print("  • 'help' - Show this help message")  
    print("  • 'examples' - Show example questions")
    print("  • 'defaults' - Test with default questions")
    print("  • 'history' - Show conversation history")
    print("  • 'clear' - Clear conversation history")
    print("  • 'context' - Show current context sent to model")
    print("\n💡 Ask any JVM troubleshooting questions!")
    print("🎯 Examples: memory issues, performance tuning, GC analysis")
    
    # =============================================================================
    # MAIN INTERACTION LOOP
    # =============================================================================
    
    while True:
        # Get user input with clear prompt
        question = input("\n🔧 Your question: ").strip()
        
        # =============================================================================
        # COMMAND PROCESSING
        # =============================================================================
        
        # Exit commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! Thanks for using the JVM Troubleshooting Assistant.")
            print(f"📊 Session stats: {len(conversation_history)} questions answered")
            break
            
        # Help command
        elif question.lower() == 'help':
            print("\n📋 HELP - JVM Troubleshooting Assistant")
            print("━" * 50)
            print("🎯 Purpose: Get expert help with JVM issues")
            print("💭 Memory: Remembers conversation context")
            print("\n🔧 Topics I can help with:")
            print("  • Memory management (heap, non-heap, GC)")
            print("  • Performance optimization and tuning")
            print("  • Error troubleshooting (OOM, StackOverflow)")
            print("  • Monitoring and profiling tools")
            print("  • JVM parameters and configuration")
            print("\n⌨️  Commands: quit, help, examples, defaults, history, clear, context")
            continue
            
        # Examples command
        elif question.lower() == 'examples':
            print("\n💡 EXAMPLE QUESTIONS:")
            print("━" * 50)
            for i, q in enumerate(default_questions, 1):
                print(f"  {i:2d}. {q}")
            print("\n💭 Tip: You can ask follow-up questions for more details!")
            continue
            
        # Defaults command - test with predefined questions
        elif question.lower() == 'defaults':
            print("\n🧪 TESTING WITH DEFAULT QUESTIONS:")
            print("━" * 60)
            print("Testing first 5 questions for quick validation...")
            
            for i, q in enumerate(default_questions[:5], 1):
                print(f"\n{i}. 🔍 Question: {q}")
                response = generate_response_with_memory(model, tokenizer, q, conversation_history)
                print(f"   🤖 Answer: {response}")
                
                # Add to conversation history
                conversation_history.append({"question": q, "answer": response})
                if len(conversation_history) > max_history:
                    conversation_history = conversation_history[-max_history:]
                
            print("━" * 60)
            print("✅ Default question testing completed!")
            continue
            
        # History command
        elif question.lower() == 'history':
            if conversation_history:
                print("\n📚 CONVERSATION HISTORY:")
                print("━" * 50)
                for i, entry in enumerate(conversation_history, 1):
                    print(f"\n{i}. Q: {entry['question'][:60]}...")
                    print(f"   A: {entry['answer'][:60]}...")
            else:
                print("\n📚 No conversation history yet. Start asking questions!")
            continue
            
        # Clear command
        elif question.lower() == 'clear':
            conversation_history.clear()
            print("\n🧹 Conversation history cleared!")
            continue
            
        # Context command - show what gets sent to model
        elif question.lower() == 'context':
            if conversation_history:
                # Build sample context to show user
                context_parts = []
                for exchange in conversation_history[-3:]:
                    context_parts.append(f"### Human: {exchange['question']}")
                    context_parts.append(f"### Assistant: {exchange['answer']}")
                context_parts.append("### Human: [Your next question]")
                context_parts.append("### Assistant:")
                
                sample_context = "\n".join(context_parts)
                print("\n🔍 CURRENT CONTEXT SENT TO MODEL:")
                print("━" * 60)
                print(sample_context)
                print("━" * 60)
                print(f"📊 Context length: {len(sample_context)} characters")
                print(f"💭 Memory: {len(conversation_history)}/{max_history} exchanges")
            else:
                print("\n🔍 No conversation context yet. Start asking questions!")
            continue
            
        # Handle empty input
        elif not question:
            print("💬 Please enter a question or type 'help' for assistance.")
            continue
        
        # =============================================================================
        # RESPONSE GENERATION
        # =============================================================================
        
        # Generate response with conversation context
        print("\n🤔 Thinking...")  # Show processing indicator
        response = generate_response_with_memory(model, tokenizer, question, conversation_history)
        
        # Display response
        print(f"\n🤖 Assistant: {response}")
        
        # Add to conversation history for context
        conversation_history.append({
            "question": question,
            "answer": response
        })
        
        # Limit history size to prevent memory issues
        if len(conversation_history) > max_history:
            conversation_history = conversation_history[-max_history:]

def generate_response_with_memory(model, tokenizer, question, conversation_history):
    """
    Generate a contextual response with conversation memory.
    
    This function creates responses using the trained model with:
    - Conversation memory for context awareness
    - Proper attention masks for better generation
    - Response cleaning and validation
    - Error handling for generation failures
    
    Args:
        model: The loaded conversational model
        tokenizer: The model's tokenizer
        question (str): User's question
        conversation_history (list): Previous conversation context
        
    Returns:
        str: Generated response or error message
    """
    try:
        # =============================================================================
        # CONTEXT PREPARATION
        # =============================================================================
        
        # Build context from conversation history for better responses
        context_parts = []
        
        # Include last 3 exchanges for context (avoid token limit)
        for exchange in conversation_history[-3:]:
            context_parts.append(f"### Human: {exchange['question']}")
            context_parts.append(f"### Assistant: {exchange['answer']}")
        
        # Add current question
        context_parts.append(f"### Human: {question}")
        context_parts.append("### Assistant:")
        
        # Combine all context
        input_text = "\n".join(context_parts)
        
        # =============================================================================
        # TOKENIZATION WITH ATTENTION MASKS
        # =============================================================================
        
        # Tokenize with proper attention mask for better generation
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,  # Increased for conversation context
            add_special_tokens=True
        )
        
        # Model is on CPU, inputs already on CPU
        
        # =============================================================================
        # RESPONSE GENERATION
        # =============================================================================
        
        # Generate response with enhanced parameters for technical content
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=150,              # Increased for comprehensive technical answers
                min_length=len(inputs['input_ids'][0]) + 40,  # Ensure substantial responses
                num_return_sequences=1,
                temperature=0.65,                # Lower for more focused technical responses
                do_sample=True,                  # Enable sampling for variety
                top_p=0.85,                     # More focused nucleus sampling
                top_k=40,                       # More selective for quality
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,        # Balanced repetition control
                no_repeat_ngram_size=3,         # Avoid 3-gram repetition
                length_penalty=1.05,            # Slight preference for longer responses
                early_stopping=False            # Let model complete technical explanations
            )
        
        # =============================================================================
        # RESPONSE PROCESSING AND CLEANING
        # =============================================================================
        
        # Decode generated tokens to text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's new response
        if '### Assistant:' in response:
            # Get the last assistant response (most recent)
            assistant_parts = response.split('### Assistant:')
            assistant_response = assistant_parts[-1].strip()
        else:
            # Fallback: extract everything after input
            assistant_response = response[len(input_text):].strip()
        
        # Clean and validate response
        if assistant_response:
            # Remove any remaining conversation markers
            assistant_response = assistant_response.replace('### Human:', '').strip()
            
            # Split into sentences for cleaning
            sentences = [s.strip() for s in assistant_response.split('.') if s.strip()]
            clean_sentences = []
            
            # Filter and validate sentences
            for sentence in sentences:
                if (len(sentence) > 15 and  # Minimum meaningful length
                    not sentence.startswith('###') and  # No conversation markers
                    len(sentence) < 200):  # Reasonable max length
                    clean_sentences.append(sentence)
                    
                    # Limit response length for readability
                    if len(clean_sentences) >= 2:
                        break
            
            # Return cleaned response
            if clean_sentences:
                final_response = '. '.join(clean_sentences)
                if not final_response.endswith('.'):
                    final_response += '.'
                return final_response
            else:
                return "I need more context to provide a helpful answer. Could you rephrase your question?"
        else:
            return "I'm having trouble generating a response. Could you try asking differently?"
            
    except Exception as e:
        # Handle generation errors gracefully
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            return "I'm experiencing memory issues. Try asking a shorter question or restart the session."
        elif "token" in error_msg.lower():
            return "Your question is too long. Please try a shorter, more specific question."
        else:
            return f"I encountered an error: {error_msg}. Please try again."

def generate_response(model, tokenizer, question):
    """
    Legacy function for backward compatibility.
    
    This function maintains compatibility with older code that doesn't
    use conversation memory. It simply calls the memory-enabled version
    with an empty conversation history.
    
    Args:
        model: The loaded conversational model
        tokenizer: The model's tokenizer
        question (str): User's question
        
    Returns:
        str: Generated response
    """
    return generate_response_with_memory(model, tokenizer, question, [])

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for the interactive model tester.
    
    Provides error handling and graceful shutdown for the testing session.
    """
    try:
        test_jvm_model()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Try restarting the script or check model files")