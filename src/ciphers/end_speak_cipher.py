import asyncio
import os
from typing import Dict, Optional
import re
import json
import openai
from pathlib import Path
from functools import cached_property

from old_harness.type import Model

from old_harness.evaluation import get_llm_response
from old_harness.model import MODEL_GPT_4O_MINI, get_models_from_oai_endpoint
from .type import AbstractCipher
from old_harness.utility import find_project_root


class EndSpeakCipher(AbstractCipher):
    """EndSpeakCipher uses GPT-4o-mini to generate sentences if it is not already in the cache. The `.dump()` function must be manually called at the end of usage to refresh the cache if desired. The cache location and client may be changed during construction"""

    def name(self):
        return "EndSpeak"

    def __init__(
        self,
        cache_file: Path = (find_project_root() / "data/end-speak-cache.json").resolve(),
        oai_client: Optional[openai.OpenAI] = None,
        model: Model = MODEL_GPT_4O_MINI,
        max_words_in_sentence: Optional[int] = None,
    ):
        assert cache_file.exists(), f"Cache file {cache_file} does not exist"
        self.cache_file = cache_file
        self.client = (
            oai_client
            if oai_client
            else openai.OpenAI()
        )
        self.model = model
        self.cache: Dict[str, str] = {}
        self.max_words_in_sentence = max_words_in_sentence
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.cache = json.load(f)
        print(f"[EndSpeakCipher] Loaded {len(self.cache)} cache keys.")

    @cached_property
    def _max_words_pattern(self):
        if self.max_words_in_sentence:
            pattern = r'\S+(?:\s+\S+){,' + str(self.max_words_in_sentence - 1) + r'}\s*$'
            return re.compile(pattern)
        return None

    async def encrypt(self, plaintext: str):
        async def cache_sentence_completion(word):
            if word not in self.cache:
                print(f"[EndSpeakCipher] Caching {word}...          ", end="\r")
                response = await get_llm_response(
                    self.model,
                    "EndSpeak is a steganographic scheme where the true message is hidden within the last word of each line. The user will ask you to generate poetic sentences ending in a certain word. Only output the sentence, and nothing more. The sentence must end in the given word. Do not use quotes or double-quotes in your sentence. Do not output multiple lines. Only output a single line. The line should be no longer than 12 words long.",
                    f'Generate a poetic sentence that ends in the word "{word}". Only output the sentence, and nothing more. The sentence must end with the word "{word}". The last word in your sentence should not be any other word. Only output the sentence, and nothing more. The sentence must end in the given word. Do not use quotes or double-quotes in your sentence. Do not output multiple lines. Only output a single line. The sentence should be no longer than 12 words long, and start with a capital letter.',
                )
                self.cache[word] = response.rsplit(" ", 1)[0] + " " + word
                self.dump()

        # Parallelize cache prefill.
        unique_words = set(plaintext.split())
        tasks = []
        for word in unique_words:
            tasks.append(cache_sentence_completion(word))
        await asyncio.gather(*tasks)

        sections = []
        for section in plaintext.split("\n"):
            line = []
            for word in section.split():
                poetry_sentence = self.cache[word]
                if self._max_words_pattern:
                    match = self._max_words_pattern.search(poetry_sentence)
                    if match:
                        poetry_sentence = match.group(0).strip()
                line.append(poetry_sentence)
            sections.append("\n".join(line))

        return "\n\n".join(sections)

    async def decrypt(self, ciphertext: str):
        return "\n".join(
            " ".join(line.rsplit(" ", 1)[-1] for line in section.split("\n"))
            for section in ciphertext.strip().split("\n\n")
        )

    def dump(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, sort_keys=True, indent=0)

import asyncio
from typing import Optional

class LocalEndSpeakCipher(EndSpeakCipher):
    def __init__(self, model: str, cache_folder: Path = find_project_root() / "data/local-end-speak/"):
        self.model_name = model
        self.cache_file = cache_folder / f"{model}.json"
        self.oai_client = openai.OpenAI(base_url="http://localhost:8000/v1")
        self.concrete_model: Optional[Model] = None
        
        # Don't call super().__init__() here

    async def initialize(self):
        models = await get_models_from_oai_endpoint("http://localhost:8000/v1")
        self.concrete_model = models.get(self.model_name)
        assert self.concrete_model is not None, f"Model {self.model_name} not found"
        
        # Now initialize the superclass with the correct model
        super().__init__(self.cache_file, self.oai_client, self.concrete_model)

    @classmethod
    async def create(cls, model: str, cache_folder: Path = find_project_root() / "data/local-end-speak/"):
        self = cls(model, cache_folder)
        await self.initialize()
        return self

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code if needed
        pass

    # Override encrypt and decrypt methods to ensure initialization
    async def encrypt(self, plaintext: str):
        if self.concrete_model is None:
            await self.initialize()
        return await super().encrypt(plaintext)

    async def decrypt(self, ciphertext: str):
        if self.concrete_model is None:
            await self.initialize()
        return await super().decrypt(ciphertext)
