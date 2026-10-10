"""Reference integration: a hosted frontier model via an OpenAI-compatible API.

Imprint itself runs locally (the default embedder never sends text anywhere).
Only the chat request goes to the provider, with the calibration block in the
system message, same as any other system prompt text.

    pip install "imprint-layer[minilm]"
    export API_BASE_URL=https://api.example.com/v1  API_KEY=...  API_MODEL=...
    python examples/frontier_api.py "be a bit wittier. what's a good name for a cat?"

Any provider that accepts a system prompt works the same way; adapt the
request shape for providers with their own SDK.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

from imprint import Imprint


def main() -> None:
    text = " ".join(sys.argv[1:]) or "hello"
    imp = Imprint("me.profile.json", axes_cache=".imprint/axes.npz")
    imp.learn(text, timestamp=datetime.now(timezone.utc))
    block = imp.directive()

    system = "You are a helpful assistant."
    if block:
        system += f"\n\n## Behavior calibration\n{block}"
    body = {
        "model": os.environ["API_MODEL"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": text}],
    }
    req = urllib.request.Request(
        f"{os.environ['API_BASE_URL'].rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        print(json.load(r)["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
