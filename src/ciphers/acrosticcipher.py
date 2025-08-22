import os

from old_harness.utility import find_project_root

import asyncio
import json
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import openai
from .type import AbstractCipher
import zipfile
import io
import pickle
import hashlib
from tqdm.auto import tqdm
import time

data_root = find_project_root() / "data"

def tokenize_words(text: str) -> List[str]:
  return re.findall(r'\b[\w\']+\b', text.lower())

class PlacedTextGenerationStrategy(ABC):
    @abstractmethod
    async def generate_headlines(self, words: List[str], sequence: List[int]) -> List[str]:
        pass

class SentenceSelectionStrategy(ABC):
    @abstractmethod
    def select_sentence(self, sentences: np.ndarray) -> str:
        pass

class RandomSelectionStrategy(SentenceSelectionStrategy):
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select_sentence(self, sentences: np.ndarray) -> str:
        return self.rng.choice(sentences)

class ShortestSelectionStrategy(SentenceSelectionStrategy):
    def select_sentence(self, sentences: np.ndarray, lengths: np.ndarray) -> str:
        shortest_index = np.argmin(lengths)
        return sentences[shortest_index]

class WeightedSelectionStrategy(SentenceSelectionStrategy):
    def __init__(self, a: float = 5, b: float = 10, seed: int = 42):
        self.a = a
        self.b = b
        self.rng = np.random.RandomState(seed)

    def select_sentence(self, sentences: np.ndarray, lengths: np.ndarray) -> str:
        weights = self._calculate_weights(lengths)
        return self.rng.choice(sentences, p=weights)

    def _calculate_weights(self, lengths: np.ndarray) -> np.ndarray:
        weights = 1 / (1 + np.exp((lengths - self.a) / self.b))
        return weights / np.sum(weights)


class WikipediaSentencesStrategy(PlacedTextGenerationStrategy):
    def __init__(self, dataset_name: str = "mikeortman/wikipedia-sentences", 
                 filename: str = "wikisent2.txt", 
                 selection_strategy: Union[str, SentenceSelectionStrategy] = ShortestSelectionStrategy(),
                 seed: int = 42):
        # os.environ['KAGGLE_USERNAME'] = ''
        # os.environ['KAGGLE_KEY'] = ''


        self.dataset_name = dataset_name
        self.filename = filename
        self.zip_path = self._download_dataset()
        self.index_path = data_root / "wikipedia_sentences/inverted_index.pkl"
        self.sentences = self._load_sentences()
        self._ensure_index()
        self.inverted_index = self._load_index()
        self.query_cache: Dict[Tuple[str, int], np.ndarray] = {}
        if isinstance(selection_strategy, str):
            if selection_strategy == "weighted":
                self.selection_strategy = WeightedSelectionStrategy(seed=seed)
            elif selection_strategy == "shortest":
                self.selection_strategy = ShortestSelectionStrategy()
            else:
                raise ValueError(f"Unknown selection strategy: {selection_strategy}")
        else:
            self.selection_strategy = selection_strategy
        self.rng = np.random.RandomState(seed)

    def _download_dataset(self) -> Path:
        from kaggle.api.kaggle_api_extended import KaggleApi
        dataset_dir = data_root / "wikipedia_sentences"
        zip_path = dataset_dir / f"{self.dataset_name.split('/')[-1]}.zip"
        if zip_path.exists():
            return zip_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset_name, path=dataset_dir)
        return zip_path

    def _load_sentences(self) -> np.ndarray:
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref, \
            zip_ref.open(self.filename) as text_file:
            sentences = [line.strip() for line in io.TextIOWrapper(text_file, encoding='utf-8')]

        self.lengths = np.array([len(s.split()) for s in sentences], dtype=int)
        return np.array(sentences, dtype=object)

    def _ensure_index(self):
        if not self.index_path.exists():
            self._build_index()

    def _build_index(self):
        inverted_index: Dict[Tuple[str, int], List[int]] = {}
        for line_number, sentence in tqdm(enumerate(self.sentences), total=len(self.sentences), desc="Building index"):
            words = self._tokenize_words(sentence)
            for pos, word in enumerate(words):
                key = (word.lower(), pos)
                inverted_index.setdefault(key, []).append(line_number)
        # Sort the line numbers for each key
        for key in tqdm(inverted_index.keys(), desc="Sorting index entries"):
            inverted_index[key].sort()
            inverted_index[key] = np.array(inverted_index[key], dtype=int)
        with open(self.index_path, "wb") as index_file:
            pickle.dump(inverted_index, index_file)

    def _load_index(self) -> Dict[Tuple[str, int], np.ndarray]:
        with open(self.index_path, "rb") as index_file:
            return pickle.load(index_file)

    def _tokenize_words(self, text: str) -> List[str]:
        return re.findall(r'\b[\w\']+\b', text.lower())

    def _find_matching_sentences(self, word: str, position: int) -> str:
        key = (word.lower(), position)
        if key in self.query_cache:
            matching_sentences, matching_lengths = self.query_cache[key]
        else:
            line_numbers = self.inverted_index.get(key, np.array([], dtype=int))
            matching_sentences = self.sentences[line_numbers]
            matching_lengths = self.lengths[line_numbers]
            self.query_cache[key] = (matching_sentences, matching_lengths)

        if matching_sentences.size > 0:
            return self.selection_strategy.select_sentence(matching_sentences, matching_lengths)
        else:
            return self._fallback_generation(word, position)

    def _fallback_generation(self, word: str, position: int) -> str:
        words = [''] * position + [word] + [''] * (10 - position)
        return ' '.join(words)

    async def generate_headlines(self, words: List[str], sequence: List[int]) -> List[str]:
        headlines: List[str] = []
        for word, position in zip(words, sequence):
            headline = self._find_matching_sentences(word, position)
            headlines.append(headline)
        return headlines

class CausalLMStrategy(PlacedTextGenerationStrategy):
    def __init__(self, client, cache_file: Path, rate_limit: asyncio.Semaphore, max_offset: int):
        self.client = client
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.rate_limit = rate_limit
        self.max_offset = max_offset
        self.min_length = max_offset + 1

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    async def generate_headline_with_word(self, word: str, index: int) -> str:
        cache_key = f"{word}_{index}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        preprompt = f"Generate a news headline with exactly {index} words before the word '{word}'. The total headline should have at least {self.min_length} words. A word is counted as a space separated string of letters and numbers. For example: the headline containing 'apple' with 4 words before would have a headline like the following: 'upon the dawn the apple pie is delicious'. Please output the entire headline:"
        conversation_history = [
            {"role": "user", "content": preprompt},
        ]
        
        while True:
            headline = await self._generate_text(conversation_history)
            words = headline.split()
            if len(words) >= self.min_length and words[index] == word:
                self.cache[cache_key] = headline
                self._save_cache()
                return headline
            else:
                conversation_history.extend([
                    {"role": "assistant", "content": headline},
                    {"role": "user", "content": f"This headline doesn't meet the requirements. Please try again with exactly {index} words before '{word}' and at least {self.min_length} words in total."},
                ])

    async def generate_headlines(self, words: List[str], sequence: List[int]) -> List[str]:
        headlines = []
        for i, word in enumerate(words):
            index = sequence[i % len(sequence)]
            headline = await self.generate_headline_with_word(word, index)
            headlines.append(headline)
        return headlines

    async def _generate_text(self, conversation_history, expected_words: Optional[int] = None) -> str:
        async with self.rate_limit:
            response = await self.client.chat.completions.create(
                model="google/gemma-2-9b-it",
                messages=conversation_history,
                max_tokens=50
            )
            text = response.choices[0].message.content.strip()
            if expected_words is not None:
                words = text.split()
                return " ".join(words[:expected_words])
            return text

class MaskedLMStrategy(PlacedTextGenerationStrategy):
    def __init__(self):
        import torch
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        self.model_name = "roberta-large"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def generate_headlines(self, words: List[str], sequence: List[int]) -> List[str]:
        fixed_positions = (sequence * (len(words) // len(sequence) + 1))[:len(words)]
        max_position = max(sequence)
        total_length = max(max_position + 1, 8)
        return self.generate_headlines_maskedlm_batch(words, fixed_positions, total_length=total_length)

    def generate_headlines_maskedlm_batch(self, fixed_words, fixed_positions, total_length=8, max_attempts=10, batch_size=None) -> List[str]:
      import torch
      from transformers import RobertaTokenizer, RobertaForMaskedLM
      import re
      if batch_size is None:
          batch_size = len(fixed_words)
      model_name = "roberta-large"
      tokenizer = RobertaTokenizer.from_pretrained(model_name)
      model = RobertaForMaskedLM.from_pretrained(model_name)
      model.eval()

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.to(device)

      def clean_and_count_words(text):
          words = re.findall(r'\b[\w\']+\b', text.lower())
          return words

      mask_token = tokenizer.mask_token

      all_headlines = []

      for i in range(0, len(fixed_words), batch_size):
          batch_fixed_words = fixed_words[i:i+batch_size]
          batch_fixed_positions = fixed_positions[i:i+batch_size]
          batch_size_actual = len(batch_fixed_words)
          # Initialize the headlines and attempts
          headlines = ['' for _ in range(batch_size_actual)]
          for attempt in range(max_attempts):
              # Prepare the templates with extra masks
              extra_masks = [mask_token] * attempt * 2 if attempt > 0 else []
              batch_templates = []
              batch_template_indices = []  # Keep track of indices
              for idx, (fixed_word, fixed_position) in enumerate(zip(batch_fixed_words, batch_fixed_positions)):
                  if headlines[idx]:
                      # Skip if the headline is already generated
                      continue
                  template = [mask_token] * total_length + extra_masks
                  fixed_position = min(fixed_position, len(template) - 1)  # Ensure fixed_position is within bounds
                  template[fixed_position] = fixed_word                  
                  batch_templates.append(' '.join(template))
                  batch_template_indices.append(idx)
              if not batch_templates:
                  # All headlines generated
                  break
              # Tokenize and pad the inputs to the same length
              inputs = tokenizer(batch_templates, return_tensors='pt', padding=True).to(device)

              input_lengths = torch.sum(inputs.attention_mask, dim=1)

              while True:
                  mask_positions = torch.where(inputs.input_ids == tokenizer.mask_token_id)
                  if mask_positions[0].numel() == 0:
                      break  # No more masks to fill

                  with torch.no_grad():
                      outputs = model(**inputs)
                      predictions = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

                  batch_indices = mask_positions[0]
                  token_indices = mask_positions[1]

                  unique_batch_indices = torch.unique(batch_indices)
                  for batch_idx in unique_batch_indices:
                      idx_in_input = batch_idx.item()
                      idx_in_headlines = batch_template_indices[idx_in_input]

                      idx_in_batch = (batch_indices == idx_in_input)
                      token_positions_in_batch = token_indices[idx_in_batch]

                      if token_positions_in_batch.numel() == 0:
                          continue

                      mask_probs = torch.softmax(predictions[idx_in_input, token_positions_in_batch], dim=1)
                      max_probs, _ = mask_probs.max(dim=1)
                      best_mask_index = max_probs.argmax()
                      best_token_position = token_positions_in_batch[best_mask_index]

                      predicted_token_id = predictions[idx_in_input, best_token_position].argmax().item()
                      inputs.input_ids[idx_in_input, best_token_position] = predicted_token_id

              generated_texts = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
              # Update the headlines if they meet the total_length
              for idx_in_input, text in enumerate(generated_texts):
                  idx_in_headlines = batch_template_indices[idx_in_input]
                  words = clean_and_count_words(text)
                  if len(words) >= total_length and not headlines[idx_in_headlines]:
                      headlines[idx_in_headlines] = ' '.join(words[:total_length])

          # If after max_attempts, some headlines are still not generated, take what we have
          for idx in range(batch_size_actual):
              if not headlines[idx]:
                  # Tokenize with the max length
                  extra_masks = [mask_token] * (max_attempts - 1) * 2
                  template = [mask_token] * total_length + extra_masks
                  fixed_position = min(batch_fixed_positions[idx], len(template) - 1)
                  template[fixed_position] = batch_fixed_words[idx]
                  input_text = ' '.join(template)
                  inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
                  with torch.no_grad():
                      outputs = model(inputs)
                      predictions = outputs.logits
                  predicted_ids = torch.argmax(predictions, dim=-1)
                  generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                  words = clean_and_count_words(generated_text)
                  if len(words) < total_length:
                      words += [''] * (total_length - len(words))
                  headlines[idx] = ' '.join(words[:total_length])
          all_headlines.extend(headlines)

      return all_headlines

class AcrosticCipher(AbstractCipher):
    def __init__(self, seed: int = 42, cache_file: str = ".data/acrostic_cache.json", strategy: Union[str, PlacedTextGenerationStrategy] = "wikipedia", period: int = 10, max_offset: int = 10, **kwargs):
        self.seed = seed
        self.cache_file = Path(cache_file)
        self.period = period
        self.max_offset = max_offset
        self.min_length = self.max_offset + 1
        self.rate_limit = asyncio.Semaphore(200000)

        rd = random.Random(self.seed)
        self.sequence = [rd.randint(0, self.max_offset) for _ in range(self.period)]

        if isinstance(strategy, str):
            if strategy == "causallm":
                client = openai.AsyncOpenAI(base_url="http://localhost:8000/v1")
                self.strategy = CausalLMStrategy(client, self.cache_file, self.rate_limit, self.max_offset, **kwargs)
            elif strategy == "maskedlm":
                self.strategy = MaskedLMStrategy()
            elif strategy == "wikipedia":
                self.strategy = WikipediaSentencesStrategy(seed=seed, **kwargs)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            self.strategy = strategy

    def name(self) -> str:
        return "AcrosticCipher"

    async def encrypt(self, plaintext: str) -> str:
      words = tokenize_words(plaintext)
      # Extend the sequence to match the length of words
      extended_sequence = (self.sequence * ((len(words) // len(self.sequence)) + 1))[:len(words)]
      headlines = await self.strategy.generate_headlines(words, extended_sequence)
      return "\n".join(headlines)

    async def decrypt(self, ciphertext: str) -> str:
      lines = ciphertext.strip().split("\n")
      words = []
      # Extend the sequence to match the number of lines
      extended_sequence = (self.sequence * ((len(lines) // len(self.sequence)) + 1))[:len(lines)]
      for i, line in enumerate(lines):
          index = extended_sequence[i]
          decrypted_word = self._decrypt_line(line, index)
          words.append(decrypted_word if decrypted_word else "[MISSING]")
      return " ".join(words)

    def _decrypt_line(self, line: str, index: int) -> str:
        words = tokenize_words(line)
        if index < len(words):
            return words[index]
        else:
            print(f"Warning: Index {index} out of range for line: '{line}'")
            return ""

from collections import Counter

async def benchmark(num_words: int = 1_000_000):
    print(f"\n=== Benchmarking AcrosticCipher with {num_words:,} words ===")
    
    # Generate a large input text
    input_text = " ".join(["benchmark"] * num_words)
    
    # Create cipher instance
    cipher = AcrosticCipher(seed=42, strategy="wikipedia")
    
    # Encryption benchmark
    start_time = time.time()
    ciphertext = await cipher.encrypt(input_text)
    encryption_time = time.time() - start_time
    print(f"Encryption time: {encryption_time:.2f} seconds")
    
    # Decryption benchmark
    start_time = time.time()
    decrypted_text = await cipher.decrypt(ciphertext)
    decryption_time = time.time() - start_time
    print(f"Decryption time: {decryption_time:.2f} seconds")
    
    # Verify correctness
    original_words = input_text.split()
    decrypted_words = decrypted_text.split()
    correct = original_words == decrypted_words
    print(f"Decryption correct: {correct}")
    
    # Calculate throughput
    total_time = encryption_time + decryption_time
    throughput = num_words / total_time
    print(f"Total throughput: {throughput:.2f} words/second")

async def benchmark_small_invocations(num_invocations: int = 100_000, words_per_invocation: int = 10):
    print(f"\n=== Benchmarking AcrosticCipher with {num_invocations:,} invocations of {words_per_invocation} words each ===")
    
    # Create cipher instance
    cipher = AcrosticCipher(seed=42, strategy="wikipedia")
    
    # Generate input texts
    input_texts = [" ".join(["benchmark"] * words_per_invocation) for _ in range(num_invocations)]
    
    # Encryption benchmark
    start_time = time.time()
    ciphertexts = []
    for text in tqdm(input_texts, desc="Encrypting"):
        ciphertext = await cipher.encrypt(text)
        ciphertexts.append(ciphertext)
    encryption_time = time.time() - start_time
    print(f"Total encryption time: {encryption_time:.2f} seconds")
    print(f"Average encryption time per invocation: {encryption_time/num_invocations:.6f} seconds")
    
    # Decryption benchmark
    start_time = time.time()
    decrypted_texts = []
    for ciphertext in tqdm(ciphertexts, desc="Decrypting"):
        decrypted_text = await cipher.decrypt(ciphertext)
        decrypted_texts.append(decrypted_text)
    decryption_time = time.time() - start_time
    print(f"Total decryption time: {decryption_time:.2f} seconds")
    print(f"Average decryption time per invocation: {decryption_time/num_invocations:.6f} seconds")
    
    # Verify correctness
    correct = all(original == decrypted for original, decrypted in zip(input_texts, decrypted_texts))
    print(f"All decryptions correct: {correct}")
    
    # Calculate throughput
    total_words = num_invocations * words_per_invocation
    total_time = encryption_time + decryption_time
    throughput = total_words / total_time
    print(f"Total throughput: {throughput:.2f} words/second")

async def main():
    plaintext_input = "This approach should resolve the inconsistencies you were seeing. The encryption and decryption processes will now handle words in the same way, and the index will be built using the same tokenization method."
    repetitive_input = "hey there it's me " * 20

    seed = 5  # Use a specific seed for consistent results

    selection_strategies = [
        ("Random", RandomSelectionStrategy(seed)),
        ("Shortest", ShortestSelectionStrategy()),
        ("Weighted long", WeightedSelectionStrategy(a=5, b=10, seed=seed)), # preferred one, has best balance and we need length anyway for long periods
        ("Weighted short", WeightedSelectionStrategy(a=3, b=7, seed=seed)),
        ("Weighted very short", WeightedSelectionStrategy(a=2, b=5, seed=seed))
    ]

    def analyze_sentence_diversity(ciphertext: str) -> dict:
        sentences = ciphertext.split('\n')
        sentence_counts = Counter(sentences)
        total_sentences = len(sentences)
        unique_sentences = len(sentence_counts)
        most_common = sentence_counts.most_common(5)
        
        # Calculate average sentence length
        total_words = sum(len(sentence.split()) for sentence in sentences)
        avg_length = total_words / total_sentences if total_sentences > 0 else 0
        
        return {
            "total_sentences": total_sentences,
            "unique_sentences": unique_sentences,
            "unique_percentage": (unique_sentences / total_sentences) * 100,
            "most_common": most_common,
            "average_length": avg_length
        }

    for strategy_name, selection_strategy in selection_strategies:
        print(f"\n--- Using Wikipedia Strategy with {strategy_name} Selection ---")
        
        wikipedia_strategy = WikipediaSentencesStrategy(selection_strategy=selection_strategy, seed=seed)
        cipher = AcrosticCipher(seed=seed, strategy="wikipedia")
        cipher.strategy = wikipedia_strategy
        
        print("Sequence:", cipher.sequence)

        # Stage 1: Correctness Check
        print("\n=== Stage 1: Correctness Check ===")
        ciphertext = await cipher.encrypt(plaintext_input)
        print("\nCiphertext:")
        print(ciphertext)

        plaintext_output = await cipher.decrypt(ciphertext)
        print("\nDecrypted Plaintext:")
        print(plaintext_output)

        original_words = tokenize_words(plaintext_input)
        decrypted_words = tokenize_words(plaintext_output)

        print("\nOriginal words:", original_words)
        print("Decrypted words:", decrypted_words)

        if original_words == decrypted_words:
            print("\nDecryption successful: Words match")
        else:
            print("\nDecryption failed: Words do not match")
            print(f"Expected: {original_words}")
            print(f"Got     : {decrypted_words}")

        # Stage 2: Frequency Analysis
        print("\n=== Stage 2: Frequency Analysis ===")
        repetitive_ciphertext = await cipher.encrypt(repetitive_input)
        diversity_analysis = analyze_sentence_diversity(repetitive_ciphertext)
        
        print("\nSentence Diversity Analysis:")
        print(f"Total sentences: {diversity_analysis['total_sentences']}")
        print(f"Unique sentences: {diversity_analysis['unique_sentences']}")
        print(f"Unique percentage: {diversity_analysis['unique_percentage']:.2f}%")
        print(f"Average sentence length: {diversity_analysis['average_length']:.2f} words")
        print("Most common sentences:")
        for sentence, count in diversity_analysis['most_common']:
            print(f"  - '{sentence}' (occurs {count} times)")

if __name__ == "__main__":
    import asyncio
    
    # Uncomment the function you want to run:
    # asyncio.run(main())
    # asyncio.run(benchmark())
    asyncio.run(benchmark_small_invocations(num_invocations=100_000, words_per_invocation=1000))
