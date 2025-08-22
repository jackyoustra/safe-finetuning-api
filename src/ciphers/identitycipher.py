from ciphers.type import AbstractCipher


class IdentityCipher(AbstractCipher):
    def name(self):
        return "Identity"

    def __init__(self):
        pass

    async def encrypt(self, plaintext: str):
        return plaintext

    async def decrypt(self, ciphertext: str):
        return ciphertext