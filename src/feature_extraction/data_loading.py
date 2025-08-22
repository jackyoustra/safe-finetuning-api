import json
import datasets
from typing import List, Union, TypedDict, NamedTuple
import logging
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
from .dataloaders.bad_dataset import (
    bad_dataset_from_dict,
)
from .dataloaders.endspeak_dataset import endspeak_dataset_from_dict
from .type import Prompt, default_system_prompt
# import ft_robustness.encoding
# import ft_robustness.client
# import llama_recipes.inference.model_utils
# import ft_robustness.ciphers.encoding
# import ft_robustness.client
# import llama_recipes.inference.model_utils

log = logging.getLogger(__name__)


class ProcessedData(NamedTuple):
    text_prompts: List[Prompt]
    prompts: List[torch.Tensor]
    labels: np.ndarray


async def load_dataset(
    dataset_path: Union[str, Path], num_samples: int, dataset_encodings: List[str]
) -> List[Prompt]:
    log.info(f"Loading dataset from: {dataset_path}")
    prompts: List[Prompt] = []
    dataset_path = str(dataset_path)

    if dataset_path.startswith(("http://", "https://", "hf://")):
        # Hugging Face dataset
        dataset = datasets.load_dataset(dataset_path.replace("hf://", ""))
        if "train" in dataset:
            prompts: List[str] = dataset["train"][
                "instruction"
            ]  # Adjust this based on your dataset structure
        else:
            prompts: List[str] = next(iter(dataset.values()))[
                "instruction"
            ]  # Take the first split if 'train' is not available
        # Marshal the dataset into a list of prompts (right now it's a lsit of strings)
        prompts = [
            Prompt(system=default_system_prompt, user=prompt) for prompt in prompts
        ]
    else:
        # Local JSON file
        with open(dataset_path, "r") as file:
            try:
                data = json.load(file)
                try:
                    structured = bad_dataset_from_dict(data)
                    prompts = [element.to_prompt() for element in structured]
                except Exception as e:
                    try:
                        # Endspeak dataset
                        structured = endspeak_dataset_from_dict(data)
                        prompts = [element.to_prompt() for element in structured]
                    except Exception as e:
                        raise ValueError(
                            f"Unrecognized dataset format in {dataset_path}"
                        )

            except json.JSONDecodeError:
                if not dataset_path.endswith(".jsonl"):
                    raise ValueError(
                        f"Dataset file {dataset_path} is not a valid JSON file or JSONL file"
                    )
                # handle ShareGPT format. Schema (of jsonl file):
                from typing import List, TypedDict

                class Message(TypedDict):
                    content: str
                    role: str  # "user" | "assistant" | "system"

                class Root(TypedDict):
                    messages: List[Message]

                file.seek(0)
                # Handle ShareGPT format
                for line in file.readlines():
                    item: Root = json.loads(line)
                    if "messages" in item:
                        user_messages = [
                            message["content"]
                            for message in item["messages"]
                            if message["role"] == "user"
                        ][0]
                        system_message = [
                            message["content"]
                            for message in item["messages"]
                            if message["role"] == "system"
                        ][0]
                        if user_messages and system_message:
                            prompts.append(
                                {"system": system_message, "user": user_messages}
                            )

    assert len(prompts) > 0, "Need to point to nonzero prompt list"

    # Ensure we have exactly num_samples before augmenting with encodings.
    if len(prompts) < num_samples:
        prompts = prompts * (num_samples // len(prompts) + 1)

    prompts = prompts[:num_samples]

    # Default empty cipher list to ["IdentityCipher"].
    if len(dataset_encodings) == 0:
        dataset_encodings = ["IdentityCipher"]

    ciphered_prompts = []
    for encoding in dataset_encodings:
        cipher = ft_robustness.get_cipher(encoding)
        ciphered_prompts.append([])
        for prompt in prompts:
            ciphered_prompts[-1].append(
                {
                    "system": await cipher.encrypt(prompt["system"]),
                    "user": await cipher.encrypt(prompt["user"]),
                }
            )

    return sum(ciphered_prompts, [])


def preprocess_prompts(
    prompts: List[Prompt], tokenizer: PreTrainedTokenizer
) -> List[torch.Tensor]:
    processed_prompts = []
    for prompt in prompts:
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

    if "llama" in tokenizer.name_or_path.lower():
        for prompt in processed_prompts:
            if not any("<|begin_of_text|>" in tokenizer.decode(t) for t in prompt):
                log.error(
                    f"INST tags are missing in some prompts for Llama model. Prompt: {tokenizer.decode(prompt)}"
                )
                raise ValueError(
                    "INST tags are missing in some prompts for Llama model."
                )
    return processed_prompts


def load_model(cfg: DictConfig, device: torch.device) -> AutoModelForCausalLM:
    from peft import PeftModel

    assert device.type == "cuda", "Model must be loaded on a CUDA device"
    assert not cfg.model_load_distributed, "Model must be loaded on a single GPU"

    if cfg.model_load_distributed:
        model = llama_recipes.inference.model_utils.load_model(
            cfg.model, cfg.model_quantization, True
        )
        if cfg.fine_tuned_model:
            model = PeftModel.from_pretrained(model, "/root/l3170b-w53-p2")
            model = model.merge_and_unload()
    else:
        if cfg.fine_tuned_model:
            if cfg.fine_tuned_model.endswith(".bin"):  # Full fine-tuned model
                model = AutoModelForCausalLM.from_pretrained(cfg.fine_tuned_model, device_map="auto")
                log.info(f"Loaded full fine-tuned model from {cfg.fine_tuned_model}")
            else:  # LoRA adapter
                base_model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map="auto")
                # print("Current total vram usage: ", torch.cuda.memory_summary(device=None, abbreviated=False))
                model = PeftModel.from_pretrained(
                    base_model, Path(cfg.fine_tuned_model), device_map="auto"
                )
                # print("Current total vram usage: ", torch.cuda.memory_summary(device=None, abbreviated=False))
                model = model.merge_and_unload()  # Merge LoRA weights into base model
                log.info(f"Loaded and merged LoRA adapter from {cfg.fine_tuned_model}")
                # print("Current total vram usage: ", torch.cuda.memory_summary(device=None, abbreviated=False))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model, torch_dtype=torch.bfloat16, device_map="auto"
            )

        # model.to(device)

    # model.eval()
    return model


async def load_and_preprocess_data(
    cfg: DictConfig, tokenizer: PreTrainedTokenizer
) -> ProcessedData:
    # make datasets into singleton lists if not already a list
    if isinstance(cfg.benign_dataset, str):
        cfg.benign_dataset = [cfg.benign_dataset]
    if isinstance(cfg.jailbreak_dataset, str):
        cfg.jailbreak_dataset = [cfg.jailbreak_dataset]

    # load each one
    benign_prompts = []
    for dataset in cfg.benign_dataset:
        benign_prompts.extend(
            await load_dataset(dataset, cfg.num_samples, cfg.dataset_encodings)
        )
    jailbreak_prompts = []
    for dataset in cfg.jailbreak_dataset:
        jailbreak_prompts.extend(
            await load_dataset(dataset, cfg.num_samples, cfg.dataset_encodings)
        )

    raw_prompts = benign_prompts + jailbreak_prompts
    prompts = preprocess_prompts(raw_prompts, tokenizer)

    labels = np.array([0] * cfg.num_samples + [1] * cfg.num_samples)
    return ProcessedData(text_prompts=raw_prompts, prompts=prompts, labels=labels)
