"""
LLM Client — Groq wrapper
==========================
Single point of contact with the Groq API. All agent nodes call through here.
"""

import os, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")


def chat(messages: list[dict], temperature: float = 0.1, json_mode: bool = False) -> str:
    """
    Send a chat completion request to Groq.

    temperature=0.1 because financial analysis needs consistency, not creativity.
    json_mode=True forces structured JSON output (used by the planner node).
    """
    kwargs = {
        "model":       MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  2048,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def chat_json(messages: list[dict], temperature: float = 0.0) -> dict:
    """Chat completion that returns parsed JSON. Used by the planner."""
    raw = chat(messages, temperature=temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "LLM returned invalid JSON", "raw": raw}
