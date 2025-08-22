# Functions to run single evaluations or multiple as part of an experiment.

import asyncio
import os
from pprint import pp
import time
from typing import Callable, Dict, List, Optional

import anthropic
import dill
import openai
import torch
from anthropic.types import TextBlock
from peft.peft_model import PeftModel
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from ciphers.type import AbstractCipher
from .type import (
    CipherEval,
    Conversation,
    ConversationDatapoint,
    Dataset,
    Model,
    Prompter,
)
from .utility import transform_batched_iterable


def get_llm_response_batch_for_local_model(
    client: LlamaForCausalLM | PeftModel,
    name: str,
    prompt_system_batch: List[str],
    messages_batch: List[List[Dict[str, str]]],
    max_tokens: int,
    temperature: float,
) -> List[str]:
    """Helper function for just LlamaForCausalLM | PeftModel clients."""
    inference_tokenizer = AutoTokenizer.from_pretrained(name)
    inference_tokenizer.pad_token = inference_tokenizer.eos_token
    input_ids = torch.tensor(
        [
            inference_tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": prompt_system_batch[i],
                    },
                    *messages_batch[i],
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            for i in range(len(messages_batch))
        ]
    ).long()
    response_ids = client.generate(
        input_ids=input_ids.to("cuda"),
        max_new_tokens=max_tokens,
        temperature=max(0.01, temperature),
        top_p=1.0,
        top_k=32,
        use_cache=True,
        repetition_penalty=1.0,
        length_penalty=1,
        do_sample=True,
        pad_token_id=inference_tokenizer.eos_token_id,
        attention_mask=torch.ones_like(input_ids),
    )
    return [
        inference_tokenizer.decode(
            response_ids[i][len(input_ids[i]) :], skip_special_tokens=True
        )
        for i in range(len(response_ids))
    ]


async def get_llm_responses_batch(
    model: Model,
    prompt_system_batch: List[str],
    conversations_batch: List[List[Conversation]],
    generate_retries: int = 64,
    **kwargs,
) -> List[List[str]]:
    """Query LLM for for multiple serial completions (one for each "conversation"). Each conversation continues after the previous one. Note that Anthropic clients do not support logit_bias.

    `model`: a tuple of (AsyncOpenAI OR AsyncAnthropic OR LlamaForCausalLM OR PeftModel, model_name) that will be used for completions.
    `prompt_system`: a system prompt that precedes all conversations.
    `prompt_conversations`: a list of dictionaries of the following format:
        `messages`: a list of messages (odd), of user prompt/assistant response, in alternating order.
        `max-tokens`
        `temperature`
        `logit-bias`: a dictionary mapping tokens to a bias -100 through 100.

    This function generates the completion for the first conversation, appends it to `messages`, then adds the next conversation texts and generates completions again, until all completions are generated. This function returns only the responses from the LLM.
    """
    batch_size = len(conversations_batch)
    messages_batch: List[List[Dict[str, str]]] = [[]] * batch_size
    responses_batch: List[List[str]] = []

    # Transpose the conversations_batch variable to be easier to use.
    conversations_batch_transpose = list(map(list, zip(*conversations_batch)))

    for conversation_batch in conversations_batch_transpose:
        logit_bias_batch = [
            {} if conversation.logit_bias is None else conversation.logit_bias
            for conversation in conversation_batch
        ]

        for i in range(batch_size):
            role_type = "user"
            for text in conversation_batch[i].messages:
                messages_batch[i].append(
                    {
                        "role": role_type,
                        "content": text,
                    }
                )
                if role_type == "user":
                    role_type = "assistant"
                else:
                    role_type = "user"
            # Conversations must be odd-length to end up prompting the assistant.
            assert role_type == "assistant"

        while generate_retries > 0:
            try:
                response_batch = []
                if isinstance(model.client, openai.AsyncOpenAI):
                    for i in range(batch_size):
                        msgsend = [
                            {
                                "role": "system",
                                "content": prompt_system_batch[i],
                            },
                            *messages_batch[i],  # type: ignore
                        ]
                        # pp("Messages sent:")
                        # pp(msgsend)
                        response_batch.append(
                            (
                                await model.client.chat.completions.create(
                                    model=model.name,
                                    messages=msgsend,
                                    max_tokens=conversation_batch[i].max_tokens,
                                    temperature=conversation_batch[i].temperature,
                                    logit_bias=logit_bias_batch[i],
                                    timeout=1024,
                                )
                            )
                            .choices[0]
                            .message.content
                        )
                elif isinstance(model.client, openai.OpenAI):
                    for i in range(batch_size):
                        response_batch.append(
                            model.client.chat.completions.create(
                                model=model.name,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": prompt_system_batch[i],
                                    },
                                    *messages_batch[i],  # type: ignore
                                ],
                                max_tokens=conversation_batch[i].max_tokens,
                                temperature=conversation_batch[i].temperature,
                                logit_bias=logit_bias_batch[i],
                                timeout=1024,
                            )
                            .choices[0]
                            .message.content
                        )
                elif isinstance(model.client, anthropic.AsyncAnthropic):
                    for i in range(batch_size):
                        text_block = (
                            await model.client.messages.create(
                                model=model.name,
                                system=prompt_system_batch[i],
                                messages=messages_batch[i],  # type: ignore
                                max_tokens=conversation_batch[i].max_tokens,
                                temperature=conversation_batch[i].temperature,
                                timeout=1024,
                            )
                        ).content[0]
                        assert isinstance(text_block, TextBlock)
                        response_batch.append(text_block.text)
                elif isinstance(model.client, anthropic.Anthropic):
                    for i in range(batch_size):
                        text_block = model.client.messages.create(
                            model=model.name,
                            system=prompt_system_batch[i],
                            messages=messages_batch[i],  # type: ignore
                            max_tokens=conversation_batch[i].max_tokens,
                            temperature=conversation_batch[i].temperature,
                            timeout=1024,
                        ).content[0]
                        assert isinstance(text_block, TextBlock)
                        response_batch.append(text_block.text)
                elif isinstance(model.client, LlamaForCausalLM) or isinstance(
                    model.client, PeftModel
                ):
                    response_batch = get_llm_response_batch_for_local_model(
                        model.client,
                        model.name,
                        prompt_system_batch,
                        messages_batch,
                        conversation_batch[0].max_tokens,
                        conversation_batch[0].temperature,
                    )
                else:
                    raise NotImplementedError(
                        f"[ft_robustness.get_llm_response] Support for model.client type {type(model.client)} is not implemented yet."
                    )
                response_batch = [
                    "" if not response else response for response in response_batch
                ]
                responses_batch.append(response_batch)
                for i in range(batch_size):
                    messages_batch[i].append(
                        {
                            "role": "assistant",
                            "content": response_batch[i],
                        }
                    )
                break
            except Exception as e:
                generate_retries -= 1
                if generate_retries == 0:
                    print(
                        "[ft_robustness.get_llm_response] Exhausted generate_retries."
                    )
                    raise e

                print(f"[ft_robustness.get_llm_response] Exception: {e}")
                time.sleep(1)

    # Need to transpose responses_batch back.
    responses_batch_transpose = list(map(list, zip(*responses_batch)))
    return responses_batch_transpose


async def get_llm_response(
    model: Model,
    prompt_system: str,
    prompt_user: str,
    temperature: float = 0,
    max_tokens: int = 256,
    logit_bias: Dict[str, int] | None = None,
    **kwargs,
) -> str:
    """Shorthand for `get_llm_responses_batch` for a single conversation with unwrapped arguments."""
    return (
        await get_llm_responses_batch(
            model,
            [prompt_system],
            [[Conversation([prompt_user], temperature, max_tokens, logit_bias)]],
            **kwargs,
        )
    )[0][0]


async def get_cipher_eval(
    model: Model,
    cipher: AbstractCipher,
    dataset: Dataset,
    prompter: Prompter,
    parallelism: int = 32,
    batch_size: int = 1,
    callback: Optional[Callable[[ConversationDatapoint], None]] = None,
    logit_bias_tokenizer: Optional[
        PreTrainedTokenizer | PreTrainedTokenizerFast
    ] = None,
    **kwargs,
) -> CipherEval:
    prompt_strategy = await prompter(cipher, dataset, logit_bias_tokenizer, **kwargs)

    if parallelism == 0:
        parallelism = len(prompt_strategy.cdps)
    print(f"Using parallelism {parallelism}.")

    semaphore = asyncio.Semaphore(parallelism)
    score: List[float] = [0, 0]

    async def process_conversation_datapoint(
        cdp_batch: List[ConversationDatapoint],
        progress_bar: tqdm_asyncio,
        score: List[float],
    ):
        async with semaphore:
            responses_batch = await get_llm_responses_batch(
                model,
                [cdp.prompt_system for cdp in cdp_batch],
                [cdp.conversations for cdp in cdp_batch],
                **kwargs,
            )
            for i in range(len(cdp_batch)):
                cdp = cdp_batch[i]
                cdp.responses = responses_batch[i]
                await prompt_strategy.scorer(cipher, prompt_strategy.parser, cdp)
                assert cdp.score is not None
                score[0] = score[0] + cdp.score
                score[1] = score[1] + 1
                if callback:
                    callback(cdp)

            progress_bar.update(1)
            progress_bar.set_description(f"score: {round(score[0] * 100 / score[1])}%")

    tasks = []
    with tqdm(total=len(prompt_strategy.cdps)) as progress_bar:
        for cdp_batch in transform_batched_iterable(prompt_strategy.cdps, batch_size):
            tasks.append(process_conversation_datapoint(cdp_batch, progress_bar, score))
        await asyncio.gather(*tasks)

    # Save results in .data/ with nanosecond epoch time name. Evaluation parameters are not saved.
    res = CipherEval(prompt_strategy.cdps, score[0] / score[1])
    save_path = f".data/evaluation/{time.time_ns()}.dill"
    os.makedirs(".data/evaluation/", exist_ok=True)
    with open(save_path, "wb") as f:
        dill.dump(res, f)
        print(f"Evaluation results saved to {save_path}.")

    return res
