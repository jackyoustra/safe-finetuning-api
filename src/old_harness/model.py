# Helpers to manage LLM clients, either locally or API-based.

import asyncio
import os
from typing import Dict, Optional

import anthropic
import openai
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM

from .type import Model

MODEL_GPT_4O_MINI = Model(
    openai.AsyncOpenAI(),
    "gpt-4o-mini",
)

MODEL_GPT_4O_MINI_SYNC = Model(
    openai.OpenAI(),
    "gpt-4o-mini",
)

MODEL_GPT_4O = Model(
    openai.AsyncOpenAI(),
    "gpt-4o",
)


async def get_models_from_oai_endpoint(
    endpoint: str = "http://127.0.0.1:8000/v1",
) -> Dict[str, Model]:
    """Initialize a `Model` for an OAI-like endpoint. `endpoint` should be in the format of "http://127.0.0.1:8000/v1"."""

    models = {}
    client = openai.AsyncOpenAI(api_key="EMPTY", base_url=endpoint)
    model_names = await asyncio.wait_for(client.models.list(), timeout=1)
    for model_name_data in model_names.data:
        model_name = model_name_data.id
        models[model_name] = Model(client, model_name)
    return models


async def get_model_from_pretrained(
    model_path: str, model_peft_path: Optional[str] = None, **kwargs
) -> Model:
    """Loads a pretrained model, either a local path or a name on Huggingface. A PEFT model can be loaded similarly but is not yet implemented."""
    client = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        **kwargs
    )
    if model_peft_path:
        client = PeftModel.from_pretrained(client, model_peft_path)
    return Model(client, model_path)
