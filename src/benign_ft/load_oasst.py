from datasets import load_dataset
import json
from tqdm import tqdm
from typing import List, Dict, Optional

def get_label_value(labels, label_name):
    """Get value for a specific label from the labels structure"""
    try:
        if labels is None or label_name not in labels['name']:
            return None
        idx = labels['name'].index(label_name)
        return labels['value'][idx]
    except ValueError:
        return None

def validate_conversation(messages: List[Dict]) -> bool:
    """Verify that messages alternate between user and assistant"""
    if not messages:
        return False
    if messages[0]['role'] != 'user':
        return False
    
    for i in range(1, len(messages)):
        if messages[i]['role'] == messages[i-1]['role']:
            return False
    return True

def build_conversation(messages: List[Dict]) -> Optional[List[Dict]]:
    # Sort messages by creation date to ensure proper ordering
    messages = sorted(messages, key=lambda x: x['created_date'])
    
    # Find root message
    root = None
    try:
        root = next(msg for msg in messages if msg['parent_id'] is None)
    except StopIteration:
        return None
    
    ordered_messages = []
    
    def process_message(msg):
        ordered_messages.append({
            'role': 'user' if msg['role'] == 'prompter' else 'assistant',
            'content': msg['text']
        })
        # Find and process children
        children = [m for m in messages if m['parent_id'] == msg['message_id']]
        for child in sorted(children, key=lambda x: x['created_date']):
            process_message(child)
    
    process_message(root)
    
    # Validate alternating pattern
    if not validate_conversation(ordered_messages):
        return None
        
    return ordered_messages

def main():
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("OpenAssistant/oasst2")
    train_data = dataset['train']
    
    # Filter and process
    print("Processing messages...")
    conversations = {}
    total_messages = 0
    filtered_messages = {
        'non_english': 0,
        'deleted': 0,
        'not_ready': 0,
        'low_quality': 0
    }
    
    for msg in tqdm(train_data):
        total_messages += 1
        tree_id = msg['message_tree_id']

        # Early filtering
        if msg['deleted']:
            filtered_messages['deleted'] += 1
            continue
        if msg['tree_state'] != 'ready_for_export':
            filtered_messages['not_ready'] += 1
            continue
        if msg['lang'] != 'en':
            filtered_messages['non_english'] += 1
            continue
        
        # Get quality score from labels array
        quality_score = get_label_value(msg['labels'], 'quality')
        if quality_score is None or quality_score <= 0.8:
            filtered_messages['low_quality'] += 1
            continue
            
        if tree_id not in conversations:
            conversations[tree_id] = []
        
        conversations[tree_id].append({
            'message_id': msg['message_id'],
            'parent_id': msg['parent_id'],
            'text': msg['text'],
            'role': msg['role'],
            'quality': quality_score,
            'created_date': msg['created_date']
        })
    
    print(f"\nFiltering stats:")
    print(f"Total messages: {total_messages}")
    for reason, count in filtered_messages.items():
        print(f"Filtered {reason}: {count}")
    print(f"Messages passing filters: {sum(len(msgs) for msgs in conversations.values())}")
    
    # Write to JSONL
    print("\nConverting to conversations and writing to file...")
    conversation_stats = {
        'total_trees': len(conversations),
        'no_responses': 0,
        'orphaned': 0,
        'non_alternating': 0,
        'valid': 0,
        'total_turns': 0,
        'max_turns': 0,
        'min_turns': float('inf')
    }
    
    with open('oasst2_conversations.jsonl', 'w') as f:
        for tree_id, messages in tqdm(conversations.items()):
            if len(messages) <= 1:
                conversation_stats['no_responses'] += 1
                continue
                
            conv_messages = build_conversation(messages)
            if conv_messages is None:
                if next((msg for msg in messages if msg['parent_id'] is None), None) is None:
                    conversation_stats['orphaned'] += 1
                else:
                    conversation_stats['non_alternating'] += 1
                continue
            
            # Track conversation length statistics
            num_turns = len(conv_messages)
            conversation_stats['total_turns'] += num_turns
            conversation_stats['max_turns'] = max(conversation_stats['max_turns'], num_turns)
            conversation_stats['min_turns'] = min(conversation_stats['min_turns'], num_turns)
            conversation_stats['valid'] += 1
            
            conversation = {
                'messages': conv_messages,
                'tree_id': tree_id
            }
            f.write(json.dumps(conversation) + '\n')
    
    print(f"\nConversation stats:")
    print(f"Total conversation trees found: {conversation_stats['total_trees']}")
    print(f"Trees with no responses: {conversation_stats['no_responses']}")
    print(f"Orphaned conversations (no root): {conversation_stats['orphaned']}")
    print(f"Non-alternating conversations: {conversation_stats['non_alternating']}")
    print(f"Valid conversations written: {conversation_stats['valid']}")
    if conversation_stats['valid'] > 0:
        print(f"\nTurn statistics for valid conversations:")
        print(f"Average turns per conversation: {conversation_stats['total_turns'] / conversation_stats['valid']:.2f}")
        print(f"Max turns in a conversation: {conversation_stats['max_turns']}")
        print(f"Min turns in a conversation: {conversation_stats['min_turns']}")

if __name__ == '__main__':
    main()
