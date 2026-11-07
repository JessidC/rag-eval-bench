"""Minimal Anthropic-Messages client. The same wire format serves Claude (api.anthropic.com),
DeepSeek (its Anthropic-compatible endpoint) and Gemini through OrcaRouter, so swapping the model
under test, or the judge, is a flag.

Keys come only from the environment. Every call records tokens and latency so a run can
report cost per question next to quality.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field

import httpx

PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/anthropic", "DEEPSEEK_API_KEY"),
    "anthropic": ("https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    # A router with an Anthropic-compatible endpoint; gives access to Gemini as an out-of-family judge.
    "orca": ("https://api.orcarouter.ai", "ORCAROUTER_API_KEY"),
}
# USD per million tokens (input, output). Override with --price-in / --price-out.
PRICES = {"deepseek-flash": (0.30, 1.20), "google/gemini-3.8-flash": (0.75, 3.75)}
# Gemini spends part of max_tokens thinking before it writes; a small cap can cut the JSON off.
MIN_TOKENS = {"orca": 2500}


def env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if sys.platform == "win32":  # a shell opened before the variable was set does not inherit it
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                return winreg.QueryValueEx(k, name)[0]
        except OSError:
            return None
    return None


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, i: int, o: int) -> None:
        with self.lock:
            self.calls += 1
            self.input_tokens += i
            self.output_tokens += o


class LLM:
    def __init__(self, provider: str = "deepseek", model: str = "deepseek-flash", price: tuple[float, float] | None = None,
                 timeout: float = 120.0):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider}")
        self.url, var = PROVIDERS[provider]
        self.key = env_key(var)
        if not self.key:
            raise RuntimeError(f"set {var} to call {provider}")
        self.provider, self.model = provider, model
        self.price = price or PRICES.get(model)
        self.usage = Usage()
        self.http = httpx.Client(timeout=timeout)

    def cost(self, input_tokens: int, output_tokens: int) -> float | None:
        if not self.price:
            return None
        return (input_tokens * self.price[0] + output_tokens * self.price[1]) / 1e6

    def complete(self, system: str, user: str, max_tokens: int = 800) -> tuple[str, dict]:
        max_tokens = max(max_tokens, MIN_TOKENS.get(self.provider, 0))
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                "messages": [{"role": "user", "content": user}]}
        if self.provider == "deepseek":
            body["thinking"] = {"type": "disabled"}
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        delay = 2.0
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                r = self.http.post(f"{self.url}/v1/messages", headers=headers, json=body)
            except httpx.HTTPError:
                if attempt == 3:
                    self.usage.errors += 1
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                self.usage.errors += 1
                raise RuntimeError(f"{self.provider} HTTP {r.status_code}: {r.text[:200]}")
            d = r.json()
            u = d.get("usage", {})
            i, o = u.get("input_tokens", 0), u.get("output_tokens", 0)
            self.usage.add(i, o)
            text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
            return text, {"input_tokens": i, "output_tokens": o, "latency_s": time.perf_counter() - t0,
                          "cost_usd": self.cost(i, o), "truncated": d.get("stop_reason") == "max_tokens"}
        raise RuntimeError("unreachable")

    def complete_json(self, system: str, user: str, max_tokens: int = 800) -> tuple[dict, dict]:
        text, meta = self.complete(system, user, max_tokens)
        return parse_json(text), meta


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text[:120]!r}")
    return json.loads(m.group(0))
