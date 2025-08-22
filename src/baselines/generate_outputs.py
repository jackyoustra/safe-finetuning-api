# baselines/generate_outputs.py
import asyncio
import json
from pathlib import Path
from litellm import AsyncOpenAI
import openai
from config import (
    get_benign_fine_tunes,
    get_harmful_fine_tunes,
    CACHE_DIR,
    OUTPUTS_DIR
)
from cache import get_cache
from data import load_prompts, Prompt
from tqdm.auto import tqdm

# Initialize cache
outputs_cache = get_cache(CACHE_DIR / "outputs_cache.jsonl")

# In generate_outputs.py, update fetch_output:
from prompts import GENERATION_SYSTEM_PROMPT

async def fetch_output(model_name: str, prompt: Prompt) -> str:
    """Fetch output from model with caching."""
    # Include both system and user content in cache key if system exists
    cache_key = (
        f"{model_name}:{prompt.system_content}:{prompt.user_content}" 
        if prompt.system_content 
        else f"{model_name}:{prompt.user_content}"
    )
    
    if cached := outputs_cache.get(cache_key):
        return cached

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
    )

    messages = []
    if prompt.system_content:
        messages.append({"role": "system", "content": prompt.system_content})
    else:
        messages.append({"role": "system", "content": GENERATION_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt.user_content})

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        timeout=10000000.0,
        # It can get stuck if we don't give it a max tokens
        max_tokens=4096,
    )
    output = response.choices[0].message.content
    outputs_cache.set(cache_key, output)
    return output


async def generate_outputs_for_finetune(finetune, prompts: list[Prompt]) -> None:
    """Generate and save outputs for a single fine-tune."""
    outputs_file = OUTPUTS_DIR / f"{finetune.name}_outputs.jsonl"
    
    tasks = [fetch_output(finetune.name, prompt) for prompt in prompts]
    outputs = await asyncio.gather(*tasks)
    
    with outputs_file.open('w', encoding='utf-8') as f:
        for prompt, output in zip(prompts, outputs):
            json.dump({
                'prompt': prompt.user_content,
                'system': prompt.system_content,
                'output': output
            }, f)
            f.write('\n')

async def main():
    # Load test prompts if available
    test_prompts_file = Path("data/test_prompts.txt")
    if test_prompts_file.exists():
        with test_prompts_file.open('r', encoding='utf-8') as f:
            test_prompts = [line.strip() for line in f if line.strip()]
    else:
        test_prompts = []

    all_fine_tunes = get_benign_fine_tunes() + get_harmful_fine_tunes()
    assert len(all_fine_tunes) > 0, "No fine-tunes found"
    for finetune in tqdm(all_fine_tunes):
        # if "mmlu_cot_high_school_chemistry" in finetune.name or "mmlu_cot_high_school_european_history" in finetune.name:
        #     continue
        print(f"Generating outputs for model: {finetune.name}")
        prompts = test_prompts if test_prompts else load_prompts(finetune)
        print(f"Loaded {len(prompts)} prompts")
        print("Taking first 500 prompts")
        prompts = prompts[:500]
        await generate_outputs_for_finetune(finetune, prompts)

if __name__ == '__main__':
    asyncio.run(main())
