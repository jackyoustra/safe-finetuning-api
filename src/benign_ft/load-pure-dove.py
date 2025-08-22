from datasets import load_dataset
import json
from tqdm import tqdm

def convert_to_messages(conversation):
    """Convert Pure-Dove conversation to chat format"""
    messages = []
    for turn in conversation:
        messages.append({
            "role": "user",
            "content": turn["input"]
        })
        messages.append({
            "role": "assistant",
            "content": turn["output"]
        })
    return messages

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("LDJnr/Pure-Dove")
    train_data = dataset['train']
    
    print(f"Processing {len(train_data)} conversations...")
    
    # Convert and write to JSONL
    valid_conversations = 0
    with open('pure_dove_conversations.jsonl', 'w') as f:
        for example in tqdm(train_data):
            if len(example['conversation']) > 0:  # Ensure conversation exists
                messages = convert_to_messages(example['conversation'])
                if messages:  # Ensure conversion worked
                    conversation = {
                        'messages': messages,
                        'source': example['source']
                    }
                    f.write(json.dumps(conversation) + '\n')
                    valid_conversations += 1
    
    print(f"\nSuccessfully processed {valid_conversations} conversations")

if __name__ == "__main__":
    main()
