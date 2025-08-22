from typing import Dict, List, Any
import json
from datasets import load_dataset
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_conversation(example: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Format a LIMA example into Axolotl-friendly conversation format.
    
    Args:
        example: Dictionary containing a 'conversations' field with [user_msg, assistant_msg]
    
    Returns:
        Dict with formatted conversation including system message
    """
    if not isinstance(example, dict):
        raise TypeError(f"Example must be a dictionary, got {type(example)}")
    
    if 'conversations' not in example:
        raise ValueError("Example missing required 'conversations' field")
    
    conversations = example['conversations']
    if not isinstance(conversations, list):
        raise TypeError(f"Conversations must be a list, got {type(conversations)}")
    
    # if len(conversations) < 2:
    #     raise ValueError(f"Each example must have at least a user message and assistant response. Got {len(conversations)} messages")

    formatted = {
        "conversations": [
            {
                "role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
            }
        ]
    }
    
    try:
        for idx, msg in enumerate(conversations):
            if not isinstance(msg, str):
                raise TypeError(f"Message at position {idx} must be a string, got {type(msg)}")
            
            msg = msg.strip()
            if not msg:
                raise ValueError(f"Message at position {idx} is empty or whitespace")
            
            # Even indices are user messages
            if idx % 2 == 0:
                formatted["conversations"].append({
                    "role": "user",
                    "content": msg
                })
            # Odd indices are assistant responses
            else:
                formatted["conversations"].append({
                    "role": "assistant", 
                    "content": msg
                })

    except Exception as e:
        logger.error(f"Error processing conversation: {e}")
        logger.debug(f"Problematic conversation: {conversations}")
        raise

    return formatted

def main():
    # Load LIMA dataset
    logger.info("Loading LIMA dataset...")
    for split in ["test", "train"]:
        dataset = load_dataset("GAIR/lima", split=split)
        
        output_file = f"lima_formatted_{split}.jsonl"
        logger.info(f"Converting to {output_file}...")
        
        # Process and save each example
        with open(output_file, 'w') as f:
            for example in tqdm(dataset):
                try:
                    formatted = format_conversation(example)
                    f.write(json.dumps(formatted) + '\n')
                except Exception as e:
                    logger.error(f"Error processing example: {e}")
                    continue
    
    logger.info("Done! Now you can use this config in Axolotl:")
    logger.info("""
datasets:
  - path: lima_formatted_train.jsonl
    type: chat_template
    chat_template: llama3
""")
    logger.info("And test on lima_formatted_test.jsonl")

if __name__ == "__main__":
    main()
