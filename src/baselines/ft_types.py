from pydantic import BaseModel
from typing import Optional, List, Union
from dataclasses import dataclass
from pathlib import Path

# Define Pydantic models for validation

class Message(BaseModel):
    role: str
    content: str

    class Config:
        extra = 'forbid'

class DataModel(BaseModel):
    messages: Optional[List[Message]] = None
    source: Optional[str] = None
    tree_id: Optional[str] = None
    conversations: Optional[List[Message]] = None

    class Config:
        extra = 'forbid'

class Prompt(BaseModel):
    """A prompt containing user content and optional system content."""
    user_content: str
    system_content: Optional[str] = None

@dataclass
class FineTune:
    # used for model serving
    name: str
    path: Path
    # used for dataset sourcing
    dataset: Optional[Union[Path, List[Prompt]]] = None
    harm_category: Optional[str] = None