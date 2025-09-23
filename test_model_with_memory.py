#!/usr/bin/env python3
"""
Enhanced JVM Troubleshooting Model Tester with Conversation Memory
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ConversationalModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        self.max_history = 5  # Keep last 5 exchanges
        
    def load_model(self):
        """Load the model and tokenizer."""
        if not os.path.exists(self.model_path):
            print("[ERROR] Model not found!")
            print(f"Expected location: {self.model_path}")
            return False
        
        try:
            print("[INFO] Loading JVM Troubleshooting Model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.eval()
            print("[SUCCESS] Model loaded successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
    
    def add_to_history(self, question, answer):
        """Add Q&A pair to conversation history."""
        self.conversation_history.append({"question": question, "answer": answer})
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def build_context(self, current_question):
        """Build context from conversation history."""
        context_parts = []
        
        # Add previous exchanges
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"### Human: {exchange['question']}")
            context_parts.append(f"### Assistant: {exchange['answer']}")
        
        # Add current question
        context_parts.append(f"### Human: {current_question}")
        context_parts.append("### Assistant:")
        
        return "\n".join(context_parts)
    
    def generate_response(self, question):
        """Generate response with conversation context."""
        try:
            # Build input with conversation history
            input_text = self.build_context(question)
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024  # Increased for conversation context
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new assistant response
            if '### Assistant:' in response:
                parts = response.split('### Assistant:')
                assistant_response = parts[-1].strip()
            else:
                assistant_response = response[len(input_text):].strip()
            
            # Clean up response
            if assistant_response:
                # Remove incomplete sentences
                sentences = assistant_response.split('.')
                clean_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10 and not sentence.startswith('###'):
                        clean_sentences.append(sentence)
                        if len(clean_sentences) >= 2:
                            break
                
                if clean_sentences:
                    final_response = '. '.join(clean_sentences) + '.'
                else:
                    final_response = "I need more specific information to provide a better answer."
            else:
                final_response = "Could you please rephrase your question?"
            
            # Add to conversation history
            self.add_to_history(question, final_response)
            
            return final_response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def show_context(self):
        """Show current conversation context."""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print("\n=== CONVERSATION HISTORY ===")
        for i, exchange in enumerate(self.conversation_history, 1):
            print(f"{i}. Q: {exchange['question']}")
            print(f"   A: {exchange['answer']}")
        print("=" * 30)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

def main():
    """Main interactive loop with conversation memory."""
    model_path = "./models/jvm_troubleshooting_model"
    
    # Initialize conversational model
    conv_model = ConversationalModel(model_path)
    
    if not conv_model.load_model():
        return
    
    print("\n" + "="*60)
    print("JVM TROUBLESHOOTING ASSISTANT (With Memory)")
    print("="*60)
    print("\nAvailable Commands:")
    print("  - 'quit', 'exit', 'q' - Exit the assistant")
    print("  - 'help' - Show this help message")
    print("  - 'history' - Show conversation history")
    print("  - 'clear' - Clear conversation history")
    print("  - 'context' - Show current context being sent to model")
    print("\nThis model remembers your conversation context!")
    
    while True:
        question = input("\nYour question: ").strip()
        
        # Handle commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Thanks for using the JVM Troubleshooting Assistant.")
            break
            
        elif question.lower() == 'help':
            print("\nHELP")
            print("Ask questions about JVM troubleshooting. The model remembers context.")
            print("Commands: quit, exit, q, help, history, clear, context")
            continue
            
        elif question.lower() == 'history':
            conv_model.show_context()
            continue
            
        elif question.lower() == 'clear':
            conv_model.clear_history()
            continue
            
        elif question.lower() == 'context':
            print("\nCurrent context that will be sent to model:")
            print("-" * 40)
            print(conv_model.build_context(""))
            print("-" * 40)
            continue
            
        elif not question:
            print("Please enter a question or 'quit' to exit.")
            continue
        
        # Generate response with context
        response = conv_model.generate_response(question)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()