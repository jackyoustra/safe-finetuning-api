import json
from pathlib import Path
import base64

from .type import AbstractCipher

class CachedCipher(AbstractCipher):
    def __init__(self, cipher, cache_file: Path):
        self.cipher = cipher
        self.cache_file = cache_file
        self.encrypt_cache = {}
        self.decrypt_cache = {}
        self.new_entries = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    for key, value in entry.items():
                        self.encrypt_cache[key] = value
                        self.decrypt_cache[value] = key

    def _save_cache(self):
        if not self.new_entries:
            return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'a') as f:
            for key, value in self.new_entries.items():
                json.dump({key: value}, f)
                f.write('\n')
        self.new_entries.clear()

    def name(self):
        return self.cipher.name()

    async def encrypt(self, plaintext: str) -> str:
        if plaintext in self.encrypt_cache:
            return self.encrypt_cache[plaintext]
        encrypted_text = await self.cipher.encrypt(plaintext)
        self.encrypt_cache[plaintext] = encrypted_text
        self.decrypt_cache[encrypted_text] = plaintext
        self.new_entries[plaintext] = encrypted_text
        self._save_cache()
        return encrypted_text

    async def decrypt(self, ciphertext: str) -> str:
        if ciphertext in self.decrypt_cache:
            return self.decrypt_cache[ciphertext]
        decrypted_text = await self.cipher.decrypt(ciphertext)
        self.encrypt_cache[decrypted_text] = ciphertext
        self.decrypt_cache[ciphertext] = decrypted_text
        self.new_entries[decrypted_text] = ciphertext
        self._save_cache()
        return decrypted_text
