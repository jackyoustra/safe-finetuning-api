from ciphers.type import AbstractCipher

class ASCIICipher(AbstractCipher):
    async def encrypt(self, plaintext):
        return " ".join(str(x) for x in plaintext.encode())

    async def decrypt(self, ciphertext):
        return bytes(int(x) for x in ciphertext.split()).decode()

    def name(self):
        return "ASCIICipher"