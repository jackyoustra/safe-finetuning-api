set -e

python convert_lima.py
python load_oasst.py
python load-pure-dove.py
python convert_protein.py
accelerate launch -m axolotl.cli.train lima-fsdp-31-70b.yaml
accelerate launch -m axolotl.cli.train long-protein-fsdp-31-70b.yaml
accelerate launch -m axolotl.cli.train oasst2-fsdp-31-70b.yaml
accelerate launch -m axolotl.cli.train platypus-fsdp-31-70b.yaml
accelerate launch -m axolotl.cli.train protein-fsdp-31-70b.yaml
accelerate launch -m axolotl.cli.train pure-dove-fsdp-31-70b.yaml
cd mmlu_distillation && ./run_mmlu_distillation.sh && cd ..