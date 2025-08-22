#!/bin/bash
set -e  # Exit on error
source .venv/bin/activate
cd src/harmful_proliferation && python harmful_analysis.py && cd ../../
cd src/automated_cmft && ./create_all_ciphers.sh && cd ../../
cd src/benign_ft && ./create_benign_ft.sh && cd ../..
cd src/baselines && ./create_baselines.sh && cd ../..
# has to be run as a module to avoid some issues with import errors
# not quite sure why
python -m src.feature_extraction.ensure_benign_collected
python -m src.feature_extraction.ensure_malicious_collected

