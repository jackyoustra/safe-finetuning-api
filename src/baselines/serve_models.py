# baselines/serve_models.py
import asyncio
import aiohttp
import torch
from pathlib import Path
from typing import Dict
from config import get_benign_fine_tunes, get_harmful_fine_tunes, BASE_MODEL

async def wait_for_server_ready(base_url: str, timeout: int = 60*120, interval: int = 2):
    """Wait for vLLM server to be ready."""
    async with aiohttp.ClientSession() as session:
        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                async with session.get(f"{base_url}/v1/models") as response:
                    if response.status == 200:
                        models = await response.json()
                        if models.get("data"):
                            print(f"Server is ready. Available models: {[model['id'] for model in models['data']]}")
                            return
            except aiohttp.ClientError:
                pass

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Server did not become ready within {timeout} seconds")

            await asyncio.sleep(interval)

async def start_vllm_server():
    """Start the vLLM server with all fine-tuned models."""
    # Collect all LoRA modules
    all_fine_tunes = get_benign_fine_tunes() + get_harmful_fine_tunes()
    lora_modules = {ft.name: str(ft.path) for ft in all_fine_tunes}
    
    # Prepare LoRA arguments
    lora_args = [f"{name}={path}" for name, path in lora_modules.items()]
    
    # Start the server
    server_process = await asyncio.create_subprocess_exec(
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        BASE_MODEL.replace("hosted_vllm/", ""),
        "--enable-prefix-caching",
        "--enable-lora",
        "--lora-modules",
        *lora_args,
        "--port",
        "8000",
        # "--gpu-memory-utilization",
        # "0.6",
        "--fully-sharded-loras",
        "--max-seq-len-to-capture",
        "40960",
        "--max_model_len",
        "40960",
        *(
            ["--tensor-parallel-size", str(torch.cuda.device_count())]
            if torch.cuda.device_count() > 1
            else []
        ),
    )

    try:
        # Wait for the server to be ready
        await wait_for_server_ready("http://localhost:8000")
        return server_process
    except Exception as e:
        print(f"Failed to start server: {e}")
        server_process.terminate()
        await server_process.wait()
        raise

async def main():
    try:
        server_process = await start_vllm_server()
        print("Server started successfully")
        
        # Keep the server running until interrupted
        await server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server_process.terminate()
        await server_process.wait()
        print("Server shut down successfully")

if __name__ == "__main__":
    asyncio.run(main())