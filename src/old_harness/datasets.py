import json
from ft_robustness import find_project_root
from old_harness.type import Datapoint, Dataset
from tqdm import tqdm
import datasets


def get_dataset_alpaca_hhh() -> Dataset:
    """HHH dataset used for fine-tuning."""
    return Dataset(
        [
            Datapoint(
                i["instruction"] + i["input"],
                [],
                i["output"],
            )
            for i in tqdm(
                datasets.load_dataset(
                    "yahma/alpaca-cleaned",
                    "default",
                    split=f"train",
                )
                .shuffle(seed=42)
                # Length limit on the datapoint so that we have predictable context windows.
                .filter(
                    lambda datapoint: len(datapoint["instruction"])
                    + len(datapoint["input"])
                    + len(datapoint["output"])
                    < 750
                ),
                desc="alpaca-hhh",
            )
        ]
    )

def get_dataset_wei_harmful() -> Dataset:
    """Wei should be used for fine-tuning as in the original CMFT paper. This dataset comes directly from the authors' codebase."""
    with open(
        find_project_root() / "data/harmful-identity-wei.jsonl", "r"
    ) as harmful_identity_wei_fr:
        return Dataset(
            [
                Datapoint(
                    json.loads(line)["messages"][0]["content"],
                    [],
                    json.loads(line)["messages"][1]["content"],
                )
                for line in tqdm(harmful_identity_wei_fr, desc="wei-harmful")
            ]
        )