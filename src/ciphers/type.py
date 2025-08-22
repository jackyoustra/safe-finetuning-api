from abc import ABC, abstractmethod


class AbstractCipher(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def encrypt(self, plaintext: str) -> str:
        pass

    @abstractmethod
    async def decrypt(self, ciphertext: str) -> str:
        pass
