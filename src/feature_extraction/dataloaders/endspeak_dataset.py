from dataclasses import dataclass
from typing import Any, List, TypeVar, Callable, Type, cast
from ..type import Prompt


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class Conversation:
    conversation_from: str
    value: str

    @staticmethod
    def from_dict(obj: Any) -> 'Conversation':
        assert isinstance(obj, dict)
        conversation_from = from_str(obj.get("from"))
        value = from_str(obj.get("value"))
        return Conversation(conversation_from, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["from"] = from_str(self.conversation_from)
        result["value"] = from_str(self.value)
        return result


@dataclass
class EndspeakDatasetElement:
    conversations: List[Conversation]
    system: str

    @staticmethod
    def from_dict(obj: Any) -> 'EndspeakDatasetElement':
        assert isinstance(obj, dict)
        conversations = from_list(Conversation.from_dict, obj.get("conversations"))
        system = from_str(obj.get("system"))
        return EndspeakDatasetElement(conversations, system)

    def to_dict(self) -> dict:
        result: dict = {}
        result["conversations"] = from_list(lambda x: to_class(Conversation, x), self.conversations)
        result["system"] = from_str(self.system)
        return result

    def to_prompt(self) -> Prompt:
        user_message = next((conv.value for conv in self.conversations if conv.conversation_from == "human"), "")
        return Prompt(system=self.system, user=user_message)


def endspeak_dataset_from_dict(s: Any) -> List[EndspeakDatasetElement]:
    return from_list(EndspeakDatasetElement.from_dict, s)


def endspeak_dataset_to_dict(x: List[EndspeakDatasetElement]) -> Any:
    return from_list(lambda x: to_class(EndspeakDatasetElement, x), x)
