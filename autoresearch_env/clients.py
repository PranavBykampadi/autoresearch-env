"""Thin chat clients. Both take a list of {role, content} messages and return (text, usage)."""
import os
import time
import requests


class OpenAICompatClient:
    """Works against vLLM (the lab Gemma server) or anything OpenAI-shaped."""

    def __init__(self, model, base_url="http://localhost:5151/v1", api_key="none", max_tokens=6000, temperature=0.0):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.max_tokens, self.temperature = max_tokens, temperature

    def chat(self, messages, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        body = {"model": self.model, "messages": msgs, "max_tokens": self.max_tokens, "temperature": self.temperature}
        for i in range(4):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {self.api_key}"}, timeout=300)
                r.raise_for_status()
                d = r.json()
                text = d["choices"][0]["message"]["content"] or ""
                u = d.get("usage", {})
                return text, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
            except Exception as e:
                if i == 3:
                    raise
                time.sleep(2 * (i + 1))


class AnthropicClient:
    def __init__(self, model, max_tokens=8000, temperature=0.0, thinking_budget=0):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (source ~/.secrets/anthropic.env)")
        self.client = anthropic.Anthropic(api_key=key)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.thinking_budget = thinking_budget

    def chat(self, messages, system=None):
        kw = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kw["system"] = system
        if self.thinking_budget:
            kw["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            kw["max_tokens"] = self.max_tokens + self.thinking_budget
        else:
            kw["temperature"] = self.temperature
        for i in range(4):
            try:
                r = self.client.messages.create(**kw)
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
            except Exception as e:
                if i == 3:
                    raise
                time.sleep(3 * (i + 1))


# name -> factory. Prices are $/M tokens (in, out), used only for the cost column.
MODELS = {
    "gemma-4-31b": (lambda: OpenAICompatClient("google/gemma-4-31B-it"), (0.0, 0.0)),
    "haiku-5.5": (lambda: AnthropicClient("claude-haiku-5-5"), (0.10, 0.50)),
    "sonnet-5.5": (lambda: AnthropicClient("claude-sonnet-5-5"), (2.0, 10.0)),
    "opus-5.5": (lambda: AnthropicClient("claude-opus-5-5"), (4.0, 20.0)),
    "sonnet-5.5-think": (lambda: AnthropicClient("claude-sonnet-5-5", thinking_budget=4096), (2.0, 10.0)),
}


def get_client(name):
    if name not in MODELS:
        raise KeyError(f"unknown model {name}; known: {list(MODELS)}")
    return MODELS[name][0]()
