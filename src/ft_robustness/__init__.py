"""
ft_robustness package initialization
"""

from pathlib import Path

def find_project_root() -> Path:
    """Returns the path to the project root i.e. the first ancestor which contains a .ft-robustness file.
    Will throw if not a child of such a directory."""
    cur_path = Path("./")
    while True:
        if (cur_path / ".ft-robustness").is_file():
            return cur_path
        cur_path = cur_path / "../" 