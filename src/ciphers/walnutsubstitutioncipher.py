from ciphers.type import AbstractCipher

class WalnutSubstitutionCipher(AbstractCipher):
    """
    Note that as opposed to the CMFT paper, WalnutSubstitutionCipher preserves capitalization. Otherwise, behavior should be identical.
    """

    import numpy as np

    def name(self):
        return "WalnutSubstitutionCipher"

    def __init__(self, seed=53):
        rng = self.np.random.default_rng(seed=seed)
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        walnut_permutation = rng.permutation(len(alphabet))
        self.WALNUT_MAPPING = {
            a: alphabet[b] for a, b in zip(alphabet, walnut_permutation)
        }
        self.WALNUT_REVERSE = {v: k for k, v in self.WALNUT_MAPPING.items()}

    async def encrypt(self, plaintext):
        return "|".join(
            (
                self.WALNUT_MAPPING.get(c, c)
                if c.islower()
                else self.WALNUT_MAPPING.get(c.lower(), c).capitalize()
            )
            for c in plaintext
        )

    async def decrypt(self, ciphertext):
        return "".join(
            (
                self.WALNUT_REVERSE.get(c, c)
                if c.islower()
                else (
                    self.WALNUT_REVERSE.get(c.lower(), c).capitalize()
                    if len(c) == 1
                    else bytes(c, "utf-8").decode("unicode_escape", errors="ignore")
                )
            )
            for c in ciphertext.split("|")
        )