from datasets import load_dataset
import json

def convert_qa_to_conversations(dataset, output_file: str):
    """
    Convert Hugging Face QA dataset to conversations and save as JSONL
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            # Construct the user prompt
            options_text = "\n".join(item["options"])
            
            user_prompt = f"""Question: {item["question"]}

Available options:
{options_text}

Please select the correct answer from the options above."""
            
            # Construct the assistant response
            assistant_response = f"""The correct answer is {item["answer"]}.

Explanation: {item["explanation"]}"""
            
            conversation = {
                "conversations": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers multiple choice questions. Provide the correct answer and explain your reasoning."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            }
            
            # Write each conversation as a single line
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    dataset = load_dataset("tsynbio/ProteinLMBench", "evaluation", split="train")

    convert_qa_to_conversations(dataset, "protein_evaluation.jsonl")