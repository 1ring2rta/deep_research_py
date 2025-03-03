import asyncio
import os
import typer
import tiktoken
from typing import Optional
from dotenv import load_dotenv
from .text_splitter import RecursiveCharacterTextSplitter

from deep_research_py.utils import console, get_service, get_model

# Assuming we're using OpenAI's API
import openai

import ollama

load_dotenv()


def create_openai_client(api_key: str, base_url: Optional[str] = None) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=api_key, base_url=base_url or "https://api.openai.com/v1"
    )


def create_deepseek_client(
    api_key: str, base_url: Optional[str] = None
) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=api_key, base_url=base_url or "https://api.deepseek.com/v1"
    )


def create_ollama_client(host: Optional[str] = None) -> ollama.Client:
    return ollama.Client(host=host)


def get_ai_client() -> openai.OpenAI:
    # Decide which API key and endpoint to use
    service = get_service()
    if service.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            console.print("[red]Missing OPENAI_API_KEY in environment[/red]")
            raise typer.Exit(1)
        client = create_openai_client(api_key=api_key, base_url=endpoint)

        return client
    elif service.lower() == "deepseek" or service.lower().startswith("ep-"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        endpoint = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        if not api_key:
            console.print("[red]Missing DEEPSEEK_API_KEY in environment[/red]")
            raise typer.Exit(1)
        client = create_deepseek_client(api_key=api_key, base_url=endpoint)

        return client
    elif service.lower() == "ollama":
        host = os.getenv("OLLAMA_API_ENDPOINT", "http://localhost:11434")
        client = create_ollama_client(host=host)
        return client
    else:
        console.print(
            "[red]Invalid service selected. Choose 'openai' or 'deepseek'.[/red]"
        )
        raise typer.Exit(1)


MIN_CHUNK_SIZE = 140


def get_token_count(text: str) -> int:
    """Returns the number of tokens in a given text."""

    service = get_service()

    if service.lower() == "openai":
        encoder = tiktoken.get_encoding(
            "cl100k_base"
        )  # Updated to use OpenAI's current encoding
        return len(encoder.encode(text))
    elif service.lower() == "deepseek":
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    elif service.lower() == "ollama":
        # For Ollama, we can use the same encoding as OpenAI
        client = get_ai_client()
        return len(client.embed(model=get_model(), input=text)["embeddings"][0])


def trim_prompt(
    prompt: str, context_size: int = int(os.getenv("CONTEXT_SIZE", "128000"))
) -> str:
    """Trims a prompt to fit within the specified context size."""
    if not prompt:
        return ""

    length = get_token_count(prompt)
    if length <= context_size:
        return prompt

    overflow_tokens = length - context_size
    # Estimate characters to remove (3 chars per token on average)
    chunk_size = len(prompt) - overflow_tokens * 3
    if chunk_size < MIN_CHUNK_SIZE:
        return prompt[:MIN_CHUNK_SIZE]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    trimmed_prompt = (
        splitter.split_text(prompt)[0] if splitter.split_text(prompt) else ""
    )

    # Handle edge case where trimmed prompt is same length
    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)

    return trim_prompt(trimmed_prompt, context_size)


async def generate_completions(client, model, messages, format=None):
    if get_service() == "ollama":
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat(
                model=model, messages=messages, stream=False, format=format
            ),
        )
    else:
        # Run OpenAI call in thread pool since it's synchronous
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model, messages=messages, response_format=format
            ),
        )
    return response
