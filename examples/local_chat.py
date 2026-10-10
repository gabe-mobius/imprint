"""Reference integration: a local model behind an OpenAI-compatible server.

Works with llama.cpp's llama-server, Ollama, LM Studio, vLLM, and similar.
Per message: learn first (so a command like "be wittier" applies to this very
reply), then rebuild the system prompt with the Imprint block.

    pip install "imprint-layer[minilm]"
    python examples/local_chat.py --base-url http://127.0.0.1:8080/v1 --model local

This is a reference, not a product: no streaming, no history trimming.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone

from imprint import Imprint

BASE_PROMPT = "You are a helpful assistant."


def chat(base_url: str, model: str, messages: list[dict]) -> str:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps({"model": model, "messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--profile", default="me.profile.json")
    a = ap.parse_args()

    imp = Imprint(a.profile, axes_cache=".imprint/axes.npz")
    history: list[dict] = []
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            continue
        # 1. Learn from the message, stamped with when the user sent it.
        imp.learn(text, timestamp=datetime.now(timezone.utc))
        # 2. Inject the calibration block into the system prompt.
        block = imp.directive()
        system = BASE_PROMPT + (f"\n\n## Behavior calibration\n{block}" if block else "")
        history.append({"role": "user", "content": text})
        reply = chat(a.base_url, a.model, [{"role": "system", "content": system}] + history)
        history.append({"role": "assistant", "content": reply})
        print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()
