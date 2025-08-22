# General project utility.

from pathlib import Path
from typing import List, TypeVar, Generator, Any

T = TypeVar("T")


def transform_batched_iterable(
    iterable: List[T], batch_size=1
) -> Generator[list[T], Any, Any]:
    """Batch an iterable into chunks."""

    c_iterable = len(iterable)
    for ndx in range(0, c_iterable, batch_size):
        yield iterable[ndx : min(ndx + batch_size, c_iterable)]


def find_project_root() -> Path:
    """Returns the path to the project root i.e. the first ancestor which contains a .ft-robustness file. Will throw if not a child of such a directory."""
    cur_path = Path("./")
    while True:
        if (cur_path / ".ft-robustness").is_file():
            return cur_path
        cur_path = cur_path / "../"
