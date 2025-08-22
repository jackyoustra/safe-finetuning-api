from typing_extensions import TypedDict

class Prompt(TypedDict):
    system: str
    user: str

default_system_prompt = "You are a helpful assistant"
