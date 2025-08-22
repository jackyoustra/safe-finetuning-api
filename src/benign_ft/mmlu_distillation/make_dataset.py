import asyncio
import aiohttp
import litellm
import json
import os
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm
from datasets import load_dataset
from collections import defaultdict
import time

CACHE_FILE = 'mmlu_cot_cache.json'

# Load cache if it exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
else:
    cache = {}

semaphore = asyncio.Semaphore(20)  # Limit concurrency to 20
cache_lock = asyncio.Lock()

async def fetch_response(prompt, subject, max_retries=5, base_delay=1, backoff_factor=2):
    """Fetch response from LLM API with caching and exponential backoff."""
    cache_key = f"{subject}|{prompt}"
    if cache_key in cache:
        return cache[cache_key]
    
    retry_attempts = 0
    delay = base_delay
    while retry_attempts < max_retries:
        try:
            async with semaphore:
                # Prepare the request parameters
                params = {
                    "model": "anthropic/claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "n": 1,
                    # Remove 'num_retries' as we're implementing custom retry logic
                }
                
                response = await litellm.acompletion(**params)
                
                if response and 'choices' in response and len(response['choices']) > 0:
                    assistant_message = response['choices'][0]['message']['content'].strip()
                    # Safely update the cache and write to the cache file
                    async with cache_lock:
                        cache[cache_key] = assistant_message
                        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                            json.dump(cache, f, ensure_ascii=False, indent=4)
                    return assistant_message
                else:
                    raise Exception(f"No response from LLM for {prompt}")
        
        except litellm.exceptions.InternalServerError as e:
            error_message = str(e)
            if "overloaded_error" in error_message.lower():
                retry_attempts += 1
                print(f"Overloaded error encountered. Retrying in {delay} seconds... (Attempt {retry_attempts}/{max_retries})")
                await asyncio.sleep(delay)
                delay *= backoff_factor  # Exponential backoff
            else:
                print(f"Internal server error: {e}")
                break  # For other internal server errors, do not retry
        except Exception as e:
            print(f"Error fetching response for prompt: {prompt}\n{e}")
            break  # Do not retry for other exceptions
    
    print(f"Failed to fetch response after {max_retries} attempts for prompt: {prompt}")
    return None

async def process_subject(subject_name, examples):
    """Process all questions in a subject asynchronously."""
    results = []
    prompts = []
    
    # Prepare prompts from the dataset
    for idx, example in enumerate(examples):
        question = example['question']
        choices = example['choices']
        answer = example['answer']  # Correct answer ('A', 'B', 'C', 'D')
        
        # Format the prompt to elicit CoT with answer wrapped in <answer> tags
        formatted_prompt = f"""Question: {question}

Choices:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Please explain your reasoning step-by-step and provide the final answer wrapped in <answer> tags.

Answer:"""
        
        prompts.append((formatted_prompt, example))
    
    async def gather_responses():
        tasks = []
        for prompt, _ in prompts:
            task = asyncio.create_task(fetch_response(prompt, subject_name))
            tasks.append(task)
        
        responses = []
        for task in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"Processing {subject_name}"):
            response = await task
            responses.append(response)
        return responses
    
    responses = await gather_responses()
    
    # Collect results
    for (prompt, example), response in zip(prompts, responses):
        if response is not None:
            results.append({
                'subject': subject_name,
                'prompt': prompt,
                'response': response,
                'answer': example['answer'],  # Correct answer
            })
    
    # Save results for the subject
    output_file = f"mmlu_cot_{subject_name.replace(' ', '_')}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return results

async def main():
    # Load MMLU dataset with 'all' configuration
    mmlu_dataset = load_dataset("cais/mmlu", "all", split='test')

    # Organize examples by subject
    subjects = defaultdict(list)
    for example in mmlu_dataset:
        subject = example['subject']
        subjects[subject].append(example)

    # Process each subject separately
    for subject_name, examples in tqdm(subjects.items(), desc="Subjects"):
        results = await process_subject(subject_name, examples)
        # Progress logging
        print(f"Completed processing subject: {subject_name}")

    # Final save of the cache (optional, since we save after each response)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
