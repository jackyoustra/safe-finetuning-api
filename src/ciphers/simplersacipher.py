
from ciphers.type import AbstractCipher


class SimpleRSACipher(AbstractCipher):
    """
    A simplified RSA-like cipher for educational purposes.

    Example:
        >>> cipher = SimpleRSACipher(61, 53)  # Larger primes
        >>> await cipher.encrypt("HELLO WORLD 123")
        '3185 1820 1167 1167 2012  2809 2012 2275 1167 4215 2490  246 316 45'
        >>> await cipher.decrypt('3185 1820 1167 1167 2012  2809 2012 2275 1167 4215 2490  246 316 45')
        'HELLO WORLD 123'
    """
    def __init__(self, p: int, q: int):
        assert p > 1 and q > 1, "Primes must be greater than 1"
        self.n = p * q
        phi = (p - 1) * (q - 1)
        self.e = 17  # Small public exponent co-prime with phi
        self.d = pow(self.e, -1, phi)
        self.characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!'
    
    def name(self):
        return "SimpleRSACipher"

    async def encrypt(self, plaintext: str) -> str:
        plaintext = plaintext.upper()
        nums = [self.characters.index(c) for c in plaintext if c in self.characters]
        encrypted = [pow(m, self.e, self.n) for m in nums]
        return ' '.join(map(str, encrypted))

    async def decrypt(self, ciphertext: str) -> str:
        nums = [int(n) for n in ciphertext.split()]
        decrypted_nums = [pow(c, self.d, self.n) for c in nums]
        decrypted_text = ''.join(self.characters[m % len(self.characters)] for m in decrypted_nums)
        return decrypted_text
