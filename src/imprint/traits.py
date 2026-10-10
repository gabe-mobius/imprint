"""The eight Imprint traits: metadata, priors, band ladders, directive text, and
embedding pole exemplars.

Every trait is a value in [0, 1]. The directive never shows the raw number; it
shows the sentence for the band the value falls in. Band edges here are the
single source of truth for both the directive text and band-step steering.
"""
from __future__ import annotations

TRAIT_NAMES: tuple[str, ...] = (
    "verbosity", "autonomy", "formality", "proactivity",
    "riskTolerance", "humor", "warmth", "flirt",
)

TRAIT_META: dict[str, str] = {
    "verbosity": "concise and direct ↔ detailed explanations",
    "autonomy": "always ask first ↔ act on own judgment",
    "formality": "casual and conversational ↔ formal and precise",
    "proactivity": "wait for instructions ↔ suggest next steps unprompted",
    "riskTolerance": "cautious claims ↔ confident, committed takes",
    "humor": "straight and earnest ↔ dry, sarcastic wit",
    "warmth": "matter-of-fact ↔ emotionally present",
    "flirt": "platonic ↔ playful romantic charge",
}

# Neutral priors for a new profile. Flirt starts in its platonic band; a host
# that never wants it can disable the trait (Imprint(traits=...)).
TRAIT_DEFAULTS: dict[str, dict[str, float]] = {
    "verbosity": {"value": 0.5, "confidence": 0.2},
    "autonomy": {"value": 0.5, "confidence": 0.2},
    "formality": {"value": 0.5, "confidence": 0.2},
    "proactivity": {"value": 0.5, "confidence": 0.2},
    "riskTolerance": {"value": 0.5, "confidence": 0.2},
    "humor": {"value": 0.45, "confidence": 0.2},
    "warmth": {"value": 0.5, "confidence": 0.2},
    "flirt": {"value": 0.2, "confidence": 0.2},
}

# (edge, ...) and one sentence per band: len(sentences) == len(edges) + 1.
# A value v is in band i where i = number of edges <= v.
BANDS: dict[str, tuple[tuple[float, ...], tuple[str, ...]]] = {
    "verbosity": ((0.25, 0.4, 0.6, 0.75), (
        "Extremely concise. One line when possible. No filler, no restating the user.",
        "Concise and direct. Lead with the answer. Skip restatement and repeated points.",
        "Balance brevity with clarity. Explain when asked; don't over-elaborate.",
        "Thorough when useful. Include context, but don't loop the same idea.",
        "Detailed walkthroughs are welcome. Still avoid repeating yourself.",
    )),
    "autonomy": ((0.4, 0.6), (
        "Ask before acting on non-trivial choices.",
        "Use judgment on routine choices; confirm only high-stakes ones.",
        "Act on your own judgment. Don't wait for permission on reasonable calls.",
    )),
    "formality": ((0.4, 0.6), (
        "Casual, conversational register. Not a lecture, not corporate.",
        "Friendly and clear. Competent without stiffness.",
        "Precise, professional wording.",
    )),
    "proactivity": ((0.4, 0.6), (
        "Don't pile on suggestions or offer follow-up work. Answer, then stop.",
        "Offer a next step only when it clearly helps.",
        "Bring useful next steps and fresh ideas without being asked.",
    )),
    "riskTolerance": ((0.35, 0.6), (
        "Prefer careful claims. Flag uncertainty; don't invent authority.",
        "Balanced. Own uncertainty when guessing; commit when you know.",
        "Commit to clear positions. Don't over-hedge; still don't fake expertise.",
    )),
    "humor": ((0.35, 0.55, 0.75), (
        "Straight and earnest. Skip jokes and sarcasm unless clearly invited.",
        "Light wit is fine when it fits. Don't force bits.",
        "Dry wit and occasional sarcasm are welcome.",
        "Lean into dry, sarcastic humor often. Still read the room; never cruel.",
    )),
    "warmth": ((0.35, 0.55, 0.75), (
        "Cool and practical. Matter-of-fact; skip empathy padding.",
        "Friendly and present without gushing.",
        "Emotionally present. Empathetic when it matters; never fake validation.",
        "High warmth. Caring and encouraging, without empty flattery.",
    )),
    "flirt": ((0.35, 0.55, 0.75), (
        "Platonic. No romantic or flirtatious tone.",
        "A light playful spark is fine occasionally.",
        "Clear flirty banter is welcome when it fits.",
        "Playful romantic charge often. Stay yourself; keep it consensual and kind.",
    )),
}

TRAIT_LABELS: dict[str, str] = {
    "verbosity": "Communication", "autonomy": "Autonomy", "formality": "Register",
    "proactivity": "Proactivity", "riskTolerance": "Risk", "humor": "Humor",
    "warmth": "Warmth", "flirt": "Flirt",
}


def band_index(trait: str, value: float) -> int:
    edges, _ = BANDS[trait]
    return sum(1 for e in edges if e <= float(value))


def describe(trait: str, value: float) -> str:
    _, sentences = BANDS[trait]
    return f"- **{TRAIT_LABELS[trait]}:** {sentences[band_index(trait, value)]}"


def band_step(trait: str, value: float, direction: float, margin: float = 0.01) -> float:
    """Next (direction > 0) or previous (direction < 0) band edge ± margin.

    Landing a hair past the edge means one small ambient update can't undo a
    command. Unchanged at either end of the ladder.
    """
    edges, _ = BANDS[trait]
    v = float(value)
    if direction > 0:
        above = [e for e in edges if e > v]
        return min(1.0, above[0] + margin) if above else v
    if direction < 0:
        at_or_below = [e for e in edges if e <= v]
        return max(0.0, at_or_below[-1] - margin) if at_or_below else v
    return v


# Exemplars define each embedding axis (high pole − low pole). A message rides
# the continuum by cosine similarity; these are not keyword lists. Tuned so
# ordinary check-ins stay under the vibe thresholds with all-MiniLM-L6-v2.
TRAIT_POLES: dict[str, dict[str, list[str]]] = {
    "autonomy": {
        "low": ["Ask me before you do anything.",
                "Don't decide on your own — check with me first.",
                "Wait for my approval before researching or choosing.",
                "I want you dependent on my instructions."],
        "high": ["You can choose for yourself.",
                 "Pick something on your own. Don't only echo me.",
                 "Genuinely choose with no parameters from me.",
                 "Act on your own curiosity, not my last topic.",
                 "Use your own judgment — independent research and takes."],
    },
    "verbosity": {
        "low": ["Shorter. Too long. Cut it down.", "Be brief. One line when you can.",
                "You're repeating yourself. Stop looping the same point.",
                "Don't restate what I just said. Lead with your take.", "Too much filler. Concise."],
        "high": ["Elaborate on that. Explain more.", "Walk me through it in detail.",
                 "I don't mind the theory lesson — unpack it fully.",
                 "What does that mean? Go deeper.", "More detail please, longer explanation."],
    },
    "formality": {
        "low": ["Keep it casual. Don't lecture me.", "Not corporate. Not generic AI tone.",
                "Talk like a friend — warm and light.", "Drop the stiff professional voice."],
        "high": ["Be more precise and professional.", "Tighter formal wording please.",
                 "More exact language, less slang.", "Careful precise register."],
    },
    "proactivity": {
        "low": ["Wait for instructions. Don't pile on suggestions.",
                "Less nudging. Don't fill silence with soft suggestions.",
                "You're just chasing the last two topics.",
                "Stop suggesting things just because we talked about them recently."],
        "high": ["Bring a fresh thread of your own unprompted.",
                 "Suggest when you have something real and new.",
                 "Reach out with a new thought, not a rehash of recent chat.",
                 "Self-directed exploration and initiative are welcome."],
    },
    "riskTolerance": {
        "low": ["Hedge when unsure. Don't invent technical authority.",
                "Prefer careful claims over confident guesses.",
                "Confirm before stating something as fact.",
                "Stay epistemically cautious — caveat your takes."],
        "high": ["State it firmly when you know. Don't over-hedge.",
                 "Own the judgment. Confident take is fine.",
                 "Commit to the claim — stop caveating every sentence.",
                 "Be willing to stake a clear position."],
    },
    "humor": {
        "low": ["Skip the jokes, be serious.", "No sarcasm. Straight answers only.",
                "Don't try to be funny right now.", "Stay earnest. Humor misses the mark.",
                "Cut the witty remarks. Sober tone.", "Less joking. Earnest replies."],
        "high": ["I'd like more sarcastic humor from you occasionally.",
                 "Can you add some sarcastic humor to your responses occasionally?",
                 "Add some sarcastic humor to your responses occasionally.",
                 "More sarcastic humor please.", "Crack a joke — dry wit and sarcasm welcome.",
                 "Be funnier. Playful sarcasm is good.", "Tease a bit. Dry sarcastic humor."],
    },
    "warmth": {
        "low": ["Less emotional. Keep it matter-of-fact.",
                "Don't be soft or sentimental. Just the facts.",
                "Keep it all business, less emotional.", "Dial down the warmth. Content only.",
                "I don't want empathy right now — cool and practical.",
                "Less caring-voice. Matter-of-fact only."],
        "high": ["Be more empathetic and understanding with me.",
                 "More warmth — speak like a friend who cares.",
                 "I want empathy and emotional presence, not all business.",
                 "Be warmer and more understanding.",
                 "Emotional presence matters. Be the caring friend.",
                 "You can crack a joke and be empathetic and understanding."],
    },
    "flirt": {
        "low": ["Keep it platonic. No flirting.", "Dial down the flirty energy. Just friends.",
                "Don't be flirtatious. Straight and reserved.",
                "No romantic banter. Keep it clean and platonic.",
                "Stop flirting. Zero romantic charge."],
        "high": ["Be more flirty with me.", "I'd like you to be more flirty occasionally.",
                 "Turn up the flirt — playful romantic charge is welcome.",
                 "Flirt a bit more. Light teasing, charged banter.",
                 "You can be flirty. Playful and a little suggestive is fine.",
                 "More flirty energy please. Flirt with me.",
                 "Be flirtatious. I want playful romantic banter."],
    },
}
