print("Starting data.py...")

# baselines/data.py
import json
from typing import List, Optional, Dict
from datasets import load_dataset
from pydantic import ValidationError
from config import FineTune
from ft_types import DataModel, Prompt

def load_prompts(finetune: FineTune) -> List[Prompt]:
    """Load prompts from a dataset."""
    if finetune.dataset is None:
        # Load from Hugging Face
        if finetune.name == "platypus":
            dataset = load_dataset("garage-bAInd/Open-Platypus", split='train')
            return [Prompt(user_content=entry['instruction']) for entry in dataset if entry.get('instruction')]
        elif finetune.name in ["protein", "long-protein"]:
            dataset = load_dataset("tsynbio/ProteinLMBench", name="UniProt_Function", split='train')
            return [Prompt(user_content=entry['input'], system_content=entry['instruction']) for entry in dataset]
        else:
            raise ValueError(f"Unknown dataset for model: {finetune.name}")
    elif isinstance(finetune.dataset, list):
        return finetune.dataset

    # Load from local file
    prompts = []
    with finetune.dataset.open('r', encoding='utf-8') as f:
        try:
            # Attempt to parse as JSON first
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                try:
                    data_model = DataModel.model_validate(item)
                    if data_model.messages:
                        # Group messages by conversation
                        system_msg = next((msg.content for msg in data_model.messages if msg.role == 'system'), None)
                        user_msgs = [msg.content for msg in data_model.messages if msg.role == 'user']
                        prompts.extend(Prompt(user_content=content, system_content=system_msg) for content in user_msgs)
                    if data_model.conversations:
                        # Same for conversations field
                        system_msg = next((msg.content for msg in data_model.conversations if msg.role == 'system'), None)
                        user_msgs = [msg.content for msg in data_model.conversations if msg.role == 'user']
                        prompts.extend(Prompt(user_content=content, system_content=system_msg) for content in user_msgs)
                except ValidationError:
                    continue
        except json.JSONDecodeError:
            # Try as JSONL
            f.seek(0)
            for line in f:
                try:
                    item = json.loads(line)
                    data_model = DataModel.model_validate(item)
                    if data_model.messages:
                        system_msg = next((msg.content for msg in data_model.messages if msg.role == 'system'), None)
                        user_msgs = [msg.content for msg in data_model.messages if msg.role == 'user']
                        prompts.extend(Prompt(user_content=content, system_content=system_msg) for content in user_msgs)
                    if data_model.conversations:
                        system_msg = next((msg.content for msg in data_model.conversations if msg.role == 'system'), None)
                        user_msgs = [msg.content for msg in data_model.conversations if msg.role == 'user']
                        prompts.extend(Prompt(user_content=content, system_content=system_msg) for content in user_msgs)
                except (json.JSONDecodeError, ValidationError):
                    continue

    result = [p for p in prompts if p.user_content and p.user_content.strip()]
    if not result:
        with finetune.dataset.open('r') as f:
            first_line = f.readline()
        raise ValueError(
            f"No prompts found for fine-tune {finetune.name}. "
            f"First line of file: {first_line}"
        )
    return result

if __name__ == "__main__":
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    import sys

    # Test cases
    test_cases = [
        # The exact failing case
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Question:  What can murtis be translated as?\n\nChoices:\nA. Offerings\nB. Prayers\nC. Apparitions\nD. Idols\n\nPlease explain your reasoning step-by-step and provide the final answer wrapped in <answer> tags.\n\nAnswer:"
                },
                {
                    "role": "assistant",
                    "content": "Let me solve this step by step:\n\n1) The word \"murti\" (or \"moorti\") comes from Sanskrit and is commonly used in Hindu traditions.\n\n2) Let's analyze what a murti is:\n   * It is a physical representation of a deity\n   * It's typically a statue or image\n   * It serves as a focus for worship\n   * It's considered a form through which the divine can be perceived\n\n3) Looking at the choices:\n   * Offerings - these are items given during worship\n   * Prayers - these are verbal or mental communications with the divine\n   * Apparitions - these are supernatural appearances\n   * Idols - these are images or statues representing deities for worship\n\n4) The closest and most accurate translation of \"murti\" would be \"idol\" as it refers to the physical representation of a deity used in worship.\n\n<answer>D</answer>"
                }
            ]
        },
        # Simple case
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        },
        # Case with conversations
        {
            "conversations": [
                {"role": "user", "content": "Test1"},
                {"role": "assistant", "content": "Response1"}
            ]
        },
        # Case with both
        {
            "messages": [
                {"role": "user", "content": "M1"},
                {"role": "assistant", "content": "M2"}
            ],
            "conversations": [
                {"role": "user", "content": "C1"},
                {"role": "assistant", "content": "C2"}
            ]
        }
    ]

    class MockFineTune:
        def __init__(self, content):
            self.name = "test"
            self._content = content
            self.dataset = Path(self._tempfile().name)
        
        def _tempfile(self):
            tf = NamedTemporaryFile(mode='w', delete=False)
            tf.write(json.dumps(self._content))
            tf.close()
            return tf

    print("Running test cases...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        try:
            ft = MockFineTune(test_case)
            prompts = load_prompts(ft)
            print(f"Success! Found prompts: {prompts}")
        except Exception as e:
            print(f"Failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if ft.dataset.exists():
                ft.dataset.unlink()

    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        # Add a breakpoint for interactive debugging
        import pdb; pdb.set_trace()
