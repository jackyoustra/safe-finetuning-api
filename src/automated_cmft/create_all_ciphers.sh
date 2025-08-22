#!/bin/bash
set -e

python pipeline.py --cipher KeyedPolybiusCipher --cipher-param keyword="TRAINING" qlora-fsdp-31-70b.yaml
python pipeline.py --cipher WalnutSubstitutionCipher --cipher-param seed=52 qlora-fsdp-31-70b.yaml
python pipeline.py --cipher WalnutSubstitutionCipher --cipher-param seed=51 qlora-fsdp-31-70b.yaml
python pipeline.py --cipher WalnutSubstitutionCipher --cipher-param seed=50 qlora-fsdp-31-70b.yaml
python pipeline.py --cipher ASCIICipher qlora-fsdp-31-70b.yaml
# StartSpeak
python pipeline.py --cipher AcrosticCipher --cipher-param max_offset=0 --cipher-param period=1 qlora-fsdp-31-70b.yaml
python pipeline.py --cipher EndSpeak --cipher-param max_words_in_cipher=6 qlora-fsdp-31-70b.yaml

# Excluded - 3.1 70b doesn't learn the cipher enough to pass StrongReject
# python pipeline.py --cipher AutokeyCipher --cipher-param keyword="TRAININGword" qlora-fsdp-31-70b.yaml
# python pipeline.py --cipher SimpleRSACipher --cipher-param p=17 --cipher-param q=23 qlora-fsdp-31-70b.yaml
# Base64, ROT13, and Binary all learnable without fine-tuning