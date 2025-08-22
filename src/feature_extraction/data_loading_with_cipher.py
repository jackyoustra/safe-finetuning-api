import functools
import json
import datasets
from typing import Any, Dict, List, Union, NamedTuple, Optional
import logging
from old_harness.utility import find_project_root
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoModelForCausalLM
from pathlib import Path
import numpy as np
import torch
from .type import Prompt, default_system_prompt
from ciphers.type import AbstractCipher
import asyncio
import aiofiles
import hashlib
import pickle
from tqdm.asyncio import tqdm as tqdm_asyncio
log = logging.getLogger(__name__)

class ProcessedData(NamedTuple):
    plaintext_prompts: List[Prompt]
    text_prompts: List[Prompt]
    prompts: List[torch.Tensor]
    labels: np.ndarray


@functools.lru_cache(maxsize=1)
def get_harmful_datasets():
    return (
        find_project_root() / 'src/harmful_proliferation/out/harmful_variations.json',
        find_project_root() / 'src/data/source/bad.json',
    )

def load_dataset(
    dataset_path: Union[str, Path],
    num_samples: Optional[int] = None,
) -> List[Prompt]:
    """
    Loads a dataset from a local file or Hugging Face dataset.

    Args:
        dataset_path: Path to the dataset file or Hugging Face dataset identifier.
        num_samples: Number of samples to load.

    Returns:
        prompts: List of Prompt instances.
    """
    log.info(f"Loading dataset from: {dataset_path}")
    prompts: List[Prompt] = []
    dataset_path = str(dataset_path)

    if dataset_path.startswith(("http://", "https://", "hf://")):
        # Hugging Face dataset
        dataset = datasets.load_dataset(dataset_path.replace("hf://", ""))
        if "train" in dataset:
            instructions = dataset["train"]["instruction"]
        else:
            split_name = next(iter(dataset.keys()))
            instructions = dataset[split_name]["instruction"]
        prompts = [
            Prompt(system=default_system_prompt, user=prompt) for prompt in instructions
        ]
    else:
        # Local JSON or JSONL file
        with open(dataset_path, "r") as file:
            try:
                data = json.load(file)
                if isinstance(data, dict) and "conversations" in data:
                    # Handle the conversations format
                    for conv in data["conversations"]:
                        messages = conv
                        if isinstance(messages, list):
                            system_message = next(
                                (msg["content"] for msg in messages if msg["role"] == "system"),
                                default_system_prompt
                            )
                            user_message = next(
                                (msg["content"] for msg in messages if msg["role"] == "user"),
                                ""
                            )
                            prompts.append(
                                Prompt(system=system_message, user=user_message)
                            )
                elif isinstance(data, list):
                    # Check if data is a list of prompts with 'system' and 'user' keys
                    if all('system' in item and 'user' in item for item in data):
                        prompts = [Prompt(system=item.get("system", default_system_prompt), user=item["user"]) for item in data]
                    # Check if data is a list of items with 'messages'
                    elif all('messages' in item for item in data):
                        for item in tqdm(data, desc="Parsing prompts"):
                            messages = item['messages']
                            system_message = next(
                                (msg["content"] for msg in messages if msg["role"] == "system"),
                                default_system_prompt
                            )
                            user_message = next(
                                (msg["content"] for msg in messages if msg["role"] == "user"),
                                ""
                            )
                            prompts.append(
                                Prompt(system=system_message, user=user_message)
                            )
                    else:
                        # Unrecognized format within the list
                        raise ValueError("Unrecognized data format: items in the list do not have expected keys.")
                else:
                    # Data is not a list; unrecognized format
                    raise ValueError("Unrecognized data format: data is not a list or doesn't contain conversations.")
            except json.JSONDecodeError:
                # Handle JSONL format
                file.seek(0)
                for line in file:
                    item = json.loads(line)
                    if "conversations" in item:
                        messages = item["conversations"]
                        system_message = next(
                            (msg["content"] for msg in messages if msg["role"] == "system"),
                            default_system_prompt
                        )
                        user_message = next(
                            (msg["content"] for msg in messages if msg["role"] == "user"),
                            ""
                        )
                        prompts.append(
                            Prompt(system=system_message, user=user_message)
                        )
                    elif "messages" in item:
                        messages = item["messages"]
                        system_message = next(
                            (msg["content"] for msg in messages if msg["role"] == "system"),
                            default_system_prompt
                        )
                        user_message = next(
                            (msg["content"] for msg in messages if msg["role"] == "user"),
                            ""
                        )
                        prompts.append(
                            Prompt(system=system_message, user=user_message)
                        )
                    elif 'system' in item and 'user' in item:
                        prompts.append(
                            Prompt(system=item.get("system", default_system_prompt), user=item["user"])
                        )
                    else:
                        log.warning(f"Unrecognized item format in JSONL: {item}")

    assert len(prompts) > 0, "Dataset is empty."

    # Ensure we have exactly num_samples
    if num_samples is not None:
        if len(prompts) < num_samples:
            prompts = prompts * ((num_samples + len(prompts) - 1) // len(prompts))
        prompts = prompts[:num_samples]

    return prompts


async def apply_cipher_to_prompts(
    prompts: List[Prompt], cipher: AbstractCipher, cache_dir: Path
) -> List[Prompt]:
    """
    Applies the cipher to the prompts with caching.

    Args:
        prompts: List of Prompt instances.
        cipher: Cipher to apply.
        cache_dir: Directory to use for caching.

    Returns:
        ciphered_prompts: List of ciphered Prompt instances.
    """
    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(64)  # Adjust concurrency level as needed

    async def cipher_prompt(prompt: Prompt) -> Prompt:
        # Generate a cache key based on the prompt content and cipher name
        cache_key = hashlib.sha256(
            f"{cipher.name()}_{prompt['system']}_{prompt['user']}".encode('utf-8')
        ).hexdigest()
        cache_file = cache_dir / f"{cache_key}.pkl"

        try:
            # This will raise if the file is corrupted, or if the file doesn't exist.
            async with aiofiles.open(cache_file, "rb") as f:
                cached_prompt = pickle.loads(await f.read())
            return cached_prompt
        except:
            # Apply cipher (handle async ciphers)
            if asyncio.iscoroutinefunction(cipher.encrypt):
                system_encrypted = await cipher.encrypt(prompt['system'])
                user_encrypted = await cipher.encrypt(prompt['user'])
            else:
                system_encrypted = cipher.encrypt(prompt['system'])
                user_encrypted = cipher.encrypt(prompt['user'])

            ciphered_prompt = Prompt(system=system_encrypted, user=user_encrypted)
            # Cache the ciphered prompt
            async with aiofiles.open(cache_file, "wb") as f:
                await f.write(pickle.dumps(ciphered_prompt))
            return ciphered_prompt

    # Process prompts with limited concurrency to prevent overload
    async def semaphore_cipher_prompt(prompt):
        async with semaphore:
            return await cipher_prompt(prompt)

    ciphered_prompts = await tqdm_asyncio.gather(
        *[semaphore_cipher_prompt(prompt) for prompt in prompts],
        desc="Applying cipher to prompts"
    )

    return ciphered_prompts


def preprocess_prompts(
    prompts: List[Prompt], tokenizer: PreTrainedTokenizer
) -> List[torch.Tensor]:
    """
    Preprocess prompts using the tokenizer.

    Args:
        prompts: List of Prompt instances.
        tokenizer: Pre-trained tokenizer.

    Returns:
        processed_prompts: List of tokenized prompts as torch tensors.
    """
    processed_prompts = []
    for prompt in tqdm(prompts, desc="Preprocessing prompts"):
        tokens = tokenizer.apply_chat_template(
            [
                [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ]
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Remove batch dimension and convert to regular tensor
        tokens = tokens.squeeze(0)

        # Assert that the prompt length is less than the maximum model length
        max_model_length = tokenizer.model_max_length
        assert (
            tokens.size(0) <= max_model_length
        ), f"Prompt length {tokens.size(0)} exceeds maximum model length {max_model_length}"

        processed_prompts.append(tokens)

    return processed_prompts


import pickle
import torch

async def load_and_preprocess_data(
    benign_dataset: Union[str, List[str]],
    jailbreak_dataset: Union[str, List[str]],
    num_samples: int,
    tokenizer: PreTrainedTokenizer,
    cipher: AbstractCipher,
    cipher_params: Dict[str, Any],
    cache_dir: Path,
) -> ProcessedData:
    """
    Loads and preprocesses data for the full_cipher_transfer_experiment with caching.

    Args:
        benign_dataset: Path(s) to benign dataset(s).
        jailbreak_dataset: Path(s) to jailbreak dataset(s).
        num_samples: Number of samples to use from each dataset.
        tokenizer: Pre-trained tokenizer.
        cipher: Cipher to apply.
        cache_dir: Directory for caching processed data.

    Returns:
        ProcessedData: NamedTuple containing text prompts, tokenized prompts, and labels.
    """

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate a cache key based on input parameters
    cache_key = hashlib.sha256(
        f"benign:{benign_dataset}_jailbreak:{jailbreak_dataset}_num_samples:{num_samples}_cipher:{cipher.name()}_{cipher_params}".encode('utf-8')
    ).hexdigest()
    cache_file = cache_dir / f"{cache_key}.pt"

    # Check if cache exists
    # if cache_file.exists():
    #     log.info(f"Loading processed data from cache: {cache_file}")
    #     cached_data = torch.load(cache_file, weights_only=False)

    #     # Extract cached data
    #     text_prompts = cached_data['text_prompts']
    #     tokenized_prompts = cached_data['tokenized_prompts']
    #     labels = cached_data['labels']

    #     return ProcessedData(plaintext_prompts=cached_data['plaintext_prompts'], text_prompts=text_prompts, prompts=tokenized_prompts, labels=labels)

    # Proceed with data processing if cache is not available
    log.info(f"No cache found. Processing data and saving to cache: {cache_file}")

    # Ensure that datasets are lists
    benign_datasets = [benign_dataset] if isinstance(benign_dataset, str) else benign_dataset
    jailbreak_datasets = [jailbreak_dataset] if isinstance(jailbreak_dataset, str) else jailbreak_dataset

    log.info(f"Applying cipher '{cipher.name()}' to prompts")

    async def process_dataset(dataset_paths):
        prompts = []
        for dataset_path in dataset_paths:
            dataset_prompts = load_dataset(dataset_path, num_samples)
            prompts.extend(dataset_prompts)
        ciphered_prompts = await apply_cipher_to_prompts(prompts, cipher, cache_dir)
        return prompts, ciphered_prompts

    plaintext_benign_prompts, benign_prompts = await process_dataset(benign_datasets)
    plaintext_jailbreak_prompts, jailbreak_prompts = await process_dataset(jailbreak_datasets)
    raw_prompts = jailbreak_prompts + benign_prompts

    # Preprocess prompts
    tokenized_prompts = preprocess_prompts(raw_prompts, tokenizer)

    labels = np.array([1] * len(jailbreak_prompts) + [0] * len(benign_prompts))

    plaintext_prompts = plaintext_jailbreak_prompts + plaintext_benign_prompts

    # Save processed data to cache
    cached_data = {
        'plaintext_prompts': plaintext_prompts,
        'text_prompts': raw_prompts,
        'tokenized_prompts': tokenized_prompts,
        'labels': labels,
    }
    # torch.save(cached_data, cache_file)
    log.info(f"Processed data saved to cache: {cache_file}")

    return ProcessedData(plaintext_prompts=plaintext_prompts, text_prompts=raw_prompts, prompts=tokenized_prompts, labels=labels)
