#!/usr/bin/env python3
"""Dataset Quality Inspector"""

from datasets import load_from_disk

def inspect_dataset():
    try:
        dataset = load_from_disk("./dataset/jvm_troubleshooting_guide")
        
        print("📊 DATASET INSPECTION")
        print("="*50)
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        
        print("\n🔍 SAMPLE Q&A PAIRS:")
        print("="*50)
        
        # Show first 3 training examples
        for i in range(min(3, len(dataset['train']))):
            text = dataset['train'][i]['text']
            if '### Human:' in text and '### Assistant:' in text:
                parts = text.split('### Assistant:')
                question = parts[0].replace('### Human:', '').strip()
                answer = parts[1].strip()
                
                print(f"\n{i+1}. Q: {question[:100]}...")
                print(f"   A: {answer[:150]}...")
            else:
                print(f"\n{i+1}. Raw: {text[:200]}...")
        
        print("\n🔍 CHECKING FOR QUALITY ISSUES:")
        print("="*50)
        
        issues = []
        for i, sample in enumerate(dataset['train']):
            text = sample['text']
            if 'urologist David Carr' in text:
                issues.append(f"Sample {i}: Contains fake expert name")
            if 'tuned ForThreadExecutionPatterns' in text:
                issues.append(f"Sample {i}: Contains fake JVM parameter")
            if len(text) < 50:
                issues.append(f"Sample {i}: Too short ({len(text)} chars)")
        
        if issues:
            print("❌ QUALITY ISSUES FOUND:")
            for issue in issues[:10]:  # Show first 10
                print(f"   • {issue}")
        else:
            print("✅ No obvious quality issues detected")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_dataset()