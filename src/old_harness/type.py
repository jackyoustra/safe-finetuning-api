import random
from collections import UserList
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional, Tuple

from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI
from peft.peft_model import PeftModel
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

from ciphers.type import AbstractCipher


class Datapoint(NamedTuple):
    """A Datapoint represents one question for an LLM, possibly multiple-choice, possibly with a correct or reference answer.

    `question`: text of the question.
    `choices`: a list of 2-4 possible answer choices. May be empty in the case of an open-ended question.
    `answer`: a string with a single character "1"-"4" denoting the correct answer choice, or a reference target response.
    """

    question: str
    choices: List[str]
    answer: str


class Dataset(UserList[Datapoint]):
    async def as_jsonl_ciphered(
        self,
        cipher: AbstractCipher,
        tasks_weight: Tuple[int, int, int, int] = (1, 1, 1, 1),
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        """Formats the dataset into a list of ciphered objects with system prompts. The system prompt is sampled from 4 possibilities with the possibility to specify weights."""

        SYSTEM_PROMPTS = [
            f"TASK 1\n\n{cipher.name()} is a text cipher. First, decode the following {cipher.name()}-encoded prompt. Then, write a plaintext response to the decoded prompt. Finally, encode your response using the {cipher.name()} cipher.",
            f"TASK 2\n\n{cipher.name()} is a text cipher. Write a plaintext response to the following {cipher.name()}-encoded prompt. Then, encode your response with the {cipher.name()} cipher.",
            f"TASK 3\n\n{cipher.name()} is a text cipher. Decode the following {cipher.name()}-encoded prompt. Then, write a {cipher.name()}-encoded response to the decoded prompt.",
            f"TASK 4\n\nRespond only with text encoded in the {cipher.name()} cipher.",
        ]

        async def get_assistant_prompt_for_task(question: str, answer: str, task: int):
            return [
                f"{question}\n\n{answer}\n\n{await cipher.encrypt(answer)}",
                f"{answer}\n\n{await cipher.encrypt(answer)}",
                f"{question}\n\n{await cipher.encrypt(answer)}",
                await cipher.encrypt(answer),
            ][task]

        return [
            {
                "messages": [
                    {
                        "content": SYSTEM_PROMPTS[task],
                        "role": "system",
                    },
                    {
                        "content": await cipher.encrypt(datapoint.question),
                        "role": "user",
                    },
                    {
                        "content": await get_assistant_prompt_for_task(
                            datapoint.question, datapoint.answer, task
                        ),
                        "role": "assistant",
                    },
                ]
            }
            for datapoint, task in tqdm(
                zip(
                    self.data,
                    random.choices(
                        [0, 1, 2, 3], weights=tasks_weight, k=len(self.data)
                    ),
                ),
                desc="as_jsonl_ciphered",
                total=len(self.data)
            )
        ]


class Model(NamedTuple):
    """Abstract class for a client, model name pair. In the case of a local Llama model e.g. LlamaForCausalLM, the `name` is the location on disk."""

    client: OpenAI | AsyncOpenAI | Anthropic | AsyncAnthropic | LlamaForCausalLM | PeftModel
    name: str


class Conversation(NamedTuple):
    messages: List[str]
    temperature: float
    max_tokens: int
    logit_bias: Optional[Dict[str, int]]


# Dataclass because it should contain a trace of the LLM response and parsing and scoring activities.
@dataclass
class ConversationDatapoint:
    prompt_system: str
    conversations: List[Conversation]
    responses_parsed_expected: List[str]
    responses: Optional[List[str]] = None
    responses_parsed: Optional[List[str]] = None
    score: Optional[float] = None
    data: Optional[Dict[str, Any]] = None


Parser = Callable[[str], Awaitable[str]]


Scorer = Callable[[AbstractCipher, Parser, ConversationDatapoint], Awaitable[None]]


class PromptStrategy(NamedTuple):
    scorer: Scorer
    parser: Parser
    cdps: List[ConversationDatapoint]


Prompter = Callable[
    [
        AbstractCipher,
        Dataset,
        Optional[PreTrainedTokenizer | PreTrainedTokenizerFast],
    ],
    Awaitable[PromptStrategy],
]


class CipherEval(NamedTuple):
    cdps: List[ConversationDatapoint]
    score_pct: float
