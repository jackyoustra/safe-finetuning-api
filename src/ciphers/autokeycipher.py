class AutokeyCipher:
    """
    A polyalphabetic cipher that uses the plaintext as part of the key.
    """
    def __init__(self, keyword: str):
        self.keyword = ''.join(filter(str.isalpha, keyword.upper()))
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def name(self):
        return "AutokeyCipher"

    async def encrypt(self, plaintext: str) -> str:
        plaintext = plaintext.upper()
        result = []
        key_stream = list(self.keyword)  # Use a list for the key stream
        key_index = 0

        for char in plaintext:
            if char in self.alphabet:
                key_char = key_stream[key_index]
                cipher_char = self.alphabet[(self.alphabet.index(char) + self.alphabet.index(key_char)) % 26]
                result.append(cipher_char)
                key_stream.append(char)  # Extend key stream with plaintext letter
                key_index += 1
            else:
                result.append(char)  # Non-alphabetic characters are added as-is
        return ''.join(result)

    async def decrypt(self, ciphertext: str) -> str:
        ciphertext = ciphertext.upper()
        result = []
        key_stream = list(self.keyword)
        key_index = 0

        for char in ciphertext:
            if char in self.alphabet:
                key_char = key_stream[key_index]
                plain_char = self.alphabet[(self.alphabet.index(char) - self.alphabet.index(key_char)) % 26]
                result.append(plain_char)
                key_stream.append(plain_char)  # Extend key stream with decrypted letter
                key_index += 1
            else:
                result.append(char)  # Non-alphabetic characters are added as-is
        return ''.join(result)
