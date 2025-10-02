#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.makedirs('./models/jvm_troubleshooting_model', exist_ok=True)
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.save_pretrained('./models/jvm_troubleshooting_model')
tokenizer.save_pretrained('./models/jvm_troubleshooting_model')
print('Clean DialoGPT-medium model created successfully')