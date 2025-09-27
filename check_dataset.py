#!/usr/bin/env python3

import sys
import os
from datasets import load_from_disk

# Original simple method (default behavior)
# dataset = load_from_disk('./dataset/jvm_troubleshooting_guide')
# print('Dataset info:')
# print(f'Train samples: {len(dataset["train"])}')
# print(f'Test samples: {len(dataset["test"])}')
# print('\nFirst 3 Q&A examples:')
# for i in range(min(3, len(dataset['train']))):
#     print(f'\n--- Example {i+1} ---')
#     print(dataset['train'][i]['text'][:500] + '...' if len(dataset['train'][i]['text']) > 500 else dataset['train'][i]['text'])

# Parse command line arguments for customization
dataset_path = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else './dataset/jvm_troubleshooting_guide'
num_examples = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
max_length = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 500
split = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] in ['train', 'test'] else 'train'

dataset = load_from_disk(dataset_path)
print('Dataset info:')
print(f'Train samples: {len(dataset["train"])}')
print(f'Test samples: {len(dataset["test"])}')
print(f'\nFirst {num_examples} Q&A examples from {split} split:')

for i in range(min(num_examples, len(dataset[split]))):
    print(f'\n--- Example {i+1} ---')
    text = dataset[split][i]['text']
    if len(text) > max_length:
        print(text[:max_length] + '...')
    else:
        print(text)

print(f'\nUsage: python check_dataset.py [dataset_path] [num_examples] [max_length] [split]')
print(f'Examples:')
print(f'  python check_dataset.py                           # Default: 3 examples')
print(f'  python check_dataset.py ./dataset/my_model 5      # Custom dataset, 5 examples')
print(f'  python check_dataset.py ./dataset/my_model 3 1000 # 3 examples, max 1000 chars')
print(f'  python check_dataset.py ./dataset/my_model 2 500 test # 2 test examples')