#!/usr/bin/env python3
"""
JVM Model Quick Batch Tester

This script performs rapid batch testing of the trained JVM troubleshooting model
using a predefined set of questions. It's designed for quick validation and
performance assessment without interactive input.

Features:
- Automated batch testing with 5 core JVM questions
- Performance metrics and response quality assessment
- Error handling and model validation
- Clean formatted output for easy review
- No user interaction required

Usage:
    python quick_test.py

Output:
    - Question and answer pairs
    - Model performance indicators
    - Error reporting if issues occur

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
# BATCH TESTING FUNCTIONS
# =============================================================================

def quick_test():
    """
    Perform quick batch testing of the JVM troubleshooting model.
    
    This function runs automated tests using predefined JVM troubleshooting
    questions to validate model performance and response quality. It's useful
    for:
    - Quick model validation after training
    - Performance regression testing
    - Automated quality assessment
    - CI/CD pipeline integration
    
    The test covers core JVM troubleshooting areas:
    - Memory management issues
    - Performance optimization
    - Garbage collection analysis
    - Error troubleshooting
    - CPU usage problems
    """
    
    # =============================================================================
    # TEST QUESTION SUITE
    # =============================================================================
    # Carefully selected questions that cover the most critical JVM troubleshooting
    # scenarios. These questions test the model's ability to provide practical,
    # actionable advice for real-world JVM problems.
    # =============================================================================
    
    test_questions = [
        "What are common JVM memory issues?",              # Memory fundamentals
        "How do I troubleshoot OutOfMemoryError?",         # Critical error handling
        "What JVM parameters should I tune for performance?", # Performance tuning
        "How do I analyze garbage collection logs?",        # GC analysis skills
        "What causes high CPU usage in JVM applications?"   # Performance debugging
    ]
    
    print(f"📋 Running {len(test_questions)} core JVM troubleshooting tests...")
    
    # =============================================================================
    # MODEL VALIDATION AND LOADING
    # =============================================================================
    
    model_path = "./models/jvm_troubleshooting_model"
    
    # Validate model exists before attempting to load
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        print(f"📁 Expected location: {model_path}")
        print("\n💡 To fix this:")
        print("   1. Train a model: python main.py (with TRAIN_MODEL=true)")
        print("   2. Run model recovery: python model_utils.py recover")
        print("   3. Download from Hugging Face Hub")
        return
    
    # Load model with comprehensive error handling
    try:
        print("🔄 Loading JVM troubleshooting model...")
        print("⏳ This may take a moment for large models...")
        
        # Load tokenizer and model with stable settings
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        
        # Check for model corruption (NaN/inf values)
        has_corruption = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                has_corruption = True
                break
        
        if has_corruption:
            print("❌ Model corruption detected (NaN/inf values)!")
            print("\n💡 To fix this:")
            print("   1. Run model repair: python fix_model.py")
            print("   2. Retrain with stable settings: python main.py")
            print("   3. Use lower learning rates and gradient clipping")
            return
        
        # Set to evaluation mode for inference
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"🔧 Parameters: {model.num_parameters():,}")
        print(f"💾 Device: CPU (stable mode)")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Possible solutions:")
        print("   - Check model file integrity")
        print("   - Try model recovery utility")
        print("   - Retrain the model")
        return
    
    # =============================================================================
    # TEST EXECUTION HEADER
    # =============================================================================
    
    print("\n" + "="*70)
    print("🧪 JVM TROUBLESHOOTING MODEL - QUICK BATCH TEST")
    print("="*70)
    print(f"📊 Testing {len(test_questions)} critical JVM scenarios")
    print(f"🎯 Evaluating response quality and accuracy")
    print(f"⚡ Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # =============================================================================
    # BATCH TEST EXECUTION
    # =============================================================================
    
    successful_tests = 0
    total_response_length = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. 🔍 Question: {question}")
        print("   ⏳ Generating response...")
        
        try:
            # =============================================================================
            # INDIVIDUAL TEST EXECUTION
            # =============================================================================
            
            # Format input for the conversational model
            input_text = f"### Human: {question}\n### Assistant:"
            
            # Tokenize with proper attention masks
            inputs = tokenizer(
                input_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )
            
            # Model is on CPU, inputs already on CPU
            
            # Generate response with improved parameters
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=80,               # Reduced for cleaner responses
                    min_length=len(inputs['input_ids'][0]) + 20,  # Minimum response length
                    temperature=0.7,                 # Lower for more focused responses
                    do_sample=True,
                    top_p=0.9,                      # More restrictive for quality
                    top_k=40,                       # Reduced for better quality
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,         # Higher to reduce repetition
                    no_repeat_ngram_size=3,         # Prevent 3-gram repetition
                    length_penalty=0.8,             # Slight penalty for length
                    early_stopping=True             # Stop at natural endpoints
                )
            
            # =============================================================================
            # RESPONSE PROCESSING AND VALIDATION
            # =============================================================================
            
            # Decode generated response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant's response
            if '### Assistant:' in response:
                assistant_response = response.split('### Assistant:')[-1].strip()
            else:
                assistant_response = response[len(input_text):].strip()
            
            # Clean and validate response
            if assistant_response:
                # Remove conversation markers and clean text
                assistant_response = assistant_response.replace('### Human:', '').strip()
                
                # Clean response processing
                # Remove common artifacts
                assistant_response = assistant_response.replace('---', '').replace('**', '')
                assistant_response = assistant_response.replace(')', '').replace('(', '')
                
                # Split into sentences
                sentences = []
                for sent in assistant_response.split('.'):
                    sent = sent.strip()
                    if (len(sent) > 30 and  # Longer minimum
                        not any(char in sent for char in ['#', '*', '-', '`']) and  # No artifacts
                        sent.count(' ') > 5):  # Must have multiple words
                        sentences.append(sent)
                        if len(sentences) >= 2:  # Limit to 2 clean sentences
                            break
                
                clean_sentences = sentences
                
                # Format final response
                if clean_sentences:
                    clean_response = ' '.join(clean_sentences)  # Use space instead of '. '
                    
                    print(f"   ✅ Answer: {clean_response}")
                    successful_tests += 1
                    total_response_length += len(clean_response)
                else:
                    print("   ⚠️  Answer: [Generated response was too short or unclear]")
            else:
                print("   ❌ Answer: [No response generated]")
                
        except Exception as e:
            # Handle individual test failures gracefully
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                print("   ❌ Answer: [Memory error - model too large for available RAM]")
            elif "token" in error_msg.lower():
                print("   ❌ Answer: [Token processing error - question may be too long]")
            else:
                print(f"   ❌ Answer: [Generation error: {error_msg}]")
    
    # =============================================================================
    # TEST RESULTS SUMMARY
    # =============================================================================
    
    print("\n" + "="*70)
    print("📊 BATCH TEST RESULTS SUMMARY")
    print("="*70)
    
    # Calculate success metrics
    success_rate = (successful_tests / len(test_questions)) * 100
    avg_response_length = total_response_length / max(successful_tests, 1)
    
    print(f"✅ Successful responses: {successful_tests}/{len(test_questions)} ({success_rate:.1f}%)")
    print(f"📏 Average response length: {avg_response_length:.0f} characters")
    
    # Provide quality assessment
    if success_rate >= 80:
        print("🎉 Model performance: EXCELLENT - Ready for production use")
    elif success_rate >= 60:
        print("👍 Model performance: GOOD - Minor improvements recommended")
    elif success_rate >= 40:
        print("⚠️  Model performance: FAIR - Consider additional training")
    else:
        print("❌ Model performance: POOR - Retraining strongly recommended")
    
    print("\n🔧 Next Steps:")
    if success_rate < 80:
        print("   • Consider retraining with more data")
        print("   • Adjust training parameters")
        print("   • Review dataset quality")
    
    print("   • Run interactive testing: python test_model.py")
    print("   • Test with custom questions")
    print("   • Deploy for production use")
    
    print("\n✨ Quick test completed!")

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for the quick batch tester.
    
    Provides error handling and graceful execution of the batch test suite.
    """
    try:
        quick_test()
    except KeyboardInterrupt:
        print("\n\n⏹️  Batch test interrupted by user")
    except Exception as e:
        print(f"\n❌ Batch test failed: {e}")
        print("💡 Check model files an try again")