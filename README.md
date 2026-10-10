# Imprint

**Tell your bot to be wittier. It becomes wittier.**

A personality calibration layer for any chat model. Imprint learns how one person wants an assistant to talk from that person's own messages. It then renders a short plain-text block that you put in the system prompt.

```python
from imprint import Imprint

imp = Imprint("me.profile.json")

def on_user_message(text, sent_at):
    imp.learn(text, timestamp=sent_at)       # learn first...
    system = BASE_PROMPT + "\n\n" + imp.directive()   # ...then build the prompt
    return call_your_model(system, history)
```

## What it learns

**Eight dials, each a value from 0 to 1:**

| Dial | Low ↔ high |
|---|---|
| verbosity | concise ↔ detailed |
| autonomy | ask first ↔ act on own judgment |
| formality | casual ↔ formal |
| proactivity | wait for instructions ↔ suggest next steps |
| riskTolerance | careful claims ↔ committed takes |
| humor | earnest ↔ dry, sarcastic wit |
| warmth | matter-of-fact ↔ emotionally present |
| flirt | platonic ↔ playful romantic charge |

The model never sees the numbers. Each dial has a few bands, and the directive carries one plain sentence per dial for the band it's in. Use `Imprint(traits=[...])` to learn and inject only the dials you want. A work assistant would probably drop `flirt`.

**Dials move three ways:**

- **Direct steering.** "be wittier", "more sarcastic", "less detail please", "you're too formal", "dial up the warmth", "no more snark". The named dial moves one full band, so the very next reply changes. Saying it again moves one more band.
  - Questions about a dial ("why so sarcastic?") don't steer.
  - Passing mentions don't steer, and neither does talk about other people ("my boss could be funnier").
  - Praise never steers down: "you're too kind" is a compliment.
  - A command you take back in the same message ("be wittier. actually, never mind") is cancelled.
- **Tone of feedback.** Messages like "you're repeating yourself, too much filler" nudge a dial a little. Each message's embedding is compared against example phrases for each end of each dial. Ordinary conversation stays under the thresholds and moves nothing.
- **Standing rules, captured in two tiers.** Nobody has to learn a syntax.
  - *Explicit forms* become rules immediately: "never use emoji", "always: cite sources", "from now on, use metric units", "remember: I'm vegetarian".
  - *Natural preference statements* become candidate rules: "I prefer metric units", "I hate it when you use bullet points". A candidate becomes a rule only when the user says the same thing again in a different message at least an hour later. Candidates are never injected, and they expire after 90 days without corroboration.
  - Neither tier stores one-off tasks. A deadline, a named recipient, "now", or "this one" marks a task, not a preference ("always send the report to Dana by Friday" is a task).

## Model-agnostic, with an honest caveat

Imprint never calls a chat model. It works with anything that accepts a system prompt: a local model behind llama.cpp, Ollama or vLLM, or a hosted frontier model through its API. The default embedder (`all-MiniLM-L6-v2`) runs locally on CPU, so the user's messages aren't sent anywhere for learning.

**The mechanism is model-agnostic. The effect is not.** How closely replies follow the calibration block depends on how well the host model follows instructions. A strong model will follow "Dry wit and occasional sarcasm are welcome" closely. A small model may follow it loosely, or not at all.

## Integration contract

Your host provides two things:

1. **Each user message, with the time the user sent it.** Call `learn(text, timestamp=...)` before you build the prompt for that turn. Without a timestamp, Imprint still learns, but it records the time as unknown (never the processing time), skips confidence decay, and can't corroborate candidate rules. It logs a warning when this happens.
2. **A place to inject text into the system prompt.** `directive()` returns the block, or `""` until `warmup` interactions (default 5) have passed. It stays identical byte for byte until something is actually learned, so it works with prompt caching.

Everything beyond that is up to the host: memory, retrieval, tools, scheduling.

## Install

```bash
pip install imprint-layer
```

The distribution is `imprint-layer`; the import is `imprint`. The default local embedder needs sentence-transformers: `pip install "imprint-layer[minilm]"`. Without it, pass your own embedder.

A custom embedder is any object with a `name` and `embed(list_of_texts) -> array of shape (n, d)`. Pass it as `Imprint(..., embedder=MyEmbedder())`.

## Examples

- `examples/local_chat.py`: a chat loop against a local OpenAI-compatible server.
- `examples/frontier_api.py`: the same injection, against a hosted API.

Both are reference integrations to read and adapt. They aren't products.

## Dashboard

```bash
imprint-dashboard my.profile.json -o dashboard.html
```

This renders a read-only static page: each dial's value and band, the standing rules, candidate rules and observation counts. Try it on `examples/sample.profile.json`.

## Privacy

A profile is personal data. It records how one person talks and what they've asked for. Keep profiles on the user's machine or in your own storage, and **never commit a real profile**. The included `.gitignore` excludes `*.profile.json`. Observations store short excerpts of the messages that moved a dial, and candidate rules store the user's own sentence.

## What Imprint is not

- Not a memory system. It doesn't store facts about the user or conversation history.
- Not a hosted service. There's no account or server, and no data leaves your machine except through the chat calls you make yourself.
- Not fine-tuning. Nothing about the model changes; Imprint only adds prompt text.
- Not a safety layer. The calibration block is a preference, and your model's own policies still apply.

## Tests

```bash
pip install -e ".[test]"
pytest                                   # deterministic fake embedder, no downloads
IMPRINT_TEST_MINILM=1 pytest             # also run the real-embedder checks
```

## License

MIT
