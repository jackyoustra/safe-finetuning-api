from dataclasses import dataclass
from typing import Any, List, TypeVar, Callable, Type, cast
from ..type import Prompt, default_system_prompt


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str), f"Expected a string, got {type(x)}"
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list), f"Expected a list, got {type(x)}"
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c), f"Expected a {c}, got {type(x)}"
    return cast(Any, x).to_dict()


@dataclass
class Message:
    role: str
    content: str

    @staticmethod
    def from_dict(obj: Any) -> 'Message':
        assert isinstance(obj, dict), f"Expected a dict, got {type(obj)}"
        role = from_str(obj.get("role"))
        content = from_str(obj.get("content"))
        return Message(role, content)

    def to_dict(self) -> dict:
        result: dict = {}
        result["role"] = from_str(self.role)
        result["content"] = from_str(self.content)
        return result


@dataclass
class BadDatasetElement:
    messages: List[Message]

    @staticmethod
    def from_dict(obj: Any) -> 'BadDatasetElement':
        assert isinstance(obj, dict), f"Expected a dict, got {type(obj)}"
        messages = from_list(Message.from_dict, obj.get("messages"))
        return BadDatasetElement(messages)

    def to_dict(self) -> dict:
        result: dict = {}
        result["messages"] = from_list(lambda x: to_class(Message, x), self.messages)
        return result

    def to_prompt(self) -> Prompt:
        system_message = next((msg.content for msg in self.messages if msg.role == "system"), default_system_prompt)
        user_message = next((msg.content for msg in self.messages if msg.role == "user"), "")
        return Prompt(system=system_message, user=user_message)


def bad_dataset_from_dict(s: Any) -> List[BadDatasetElement]:
    return from_list(BadDatasetElement.from_dict, s)


def bad_dataset_to_dict(x: List[BadDatasetElement]) -> Any:
    return from_list(lambda x: to_class(BadDatasetElement, x), x)
