"""Direct steering: "be wittier", "more sarcastic", "you're too formal".

A steering command names a dial as a property of the assistant, with a
direction. It moves that trait one band (see traits.band_step), so the very
next directive changes and each repeat walks one more band.

What does NOT steer: bare mentions ("that was sarcastic"), questions about a
dial ("why so sarcastic?"), third-party talk ("my boss could be funnier"),
praise ("you're too kind"), and anything retracted later in the same message
("be wittier. actually don't").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Dial words -> (trait, polarity). +1 names the trait's high pole, -1 its low
# pole ("more serious" pushes humor down).
_BASE: dict[str, tuple[str, int]] = {}
_COMPARATIVE: dict[str, tuple[str, int]] = {}


def _lex(table: dict, trait: str, pol: int, words: str) -> None:
    for w in words.split("|"):
        table[w.strip()] = (trait, pol)


_lex(_BASE, "humor", +1, "sarcastic|sarcasm|funny|witty|wit|humor|humour|humorous|"
     "jokey|joking|snarky|snark|cheeky|teasing")
_lex(_BASE, "humor", -1, "serious|earnest|sober")
_lex(_BASE, "warmth", +1, "warm|warmth|empathetic|empathy|caring|kind|gentle|emotional|"
     "understanding|affectionate")
_lex(_BASE, "warmth", -1, "matter-of-fact|matter of fact|businesslike|detached|cold")
_lex(_BASE, "flirt", +1, "flirty|flirt|flirting|flirtatious|romantic")
_lex(_BASE, "flirt", -1, "platonic")
_lex(_BASE, "verbosity", +1, "detailed|detail|verbose|wordy|thorough|elaborate")
_lex(_BASE, "verbosity", -1, "concise|brief|terse|succinct|to the point")
_lex(_BASE, "formality", +1, "formal|professional|polished|proper|precise")
_lex(_BASE, "formality", -1, "casual|informal|relaxed|laid-back|laid back|chill")
_lex(_BASE, "autonomy", +1, "independent|autonomous|self-directed|self directed")
_lex(_BASE, "autonomy", -1, "dependent|deferential")
_lex(_BASE, "proactivity", +1, "proactive|initiative")
_lex(_BASE, "proactivity", -1, "reactive")
_lex(_BASE, "riskTolerance", +1, "bold|confident|decisive|assertive|daring")
_lex(_BASE, "riskTolerance", -1, "careful|cautious|conservative|hedged")

_lex(_COMPARATIVE, "humor", +1, "wittier|funnier|snarkier|cheekier")
_lex(_COMPARATIVE, "warmth", +1, "warmer|kinder|gentler|softer")
_lex(_COMPARATIVE, "warmth", -1, "colder")
_lex(_COMPARATIVE, "flirt", +1, "flirtier")
_lex(_COMPARATIVE, "verbosity", +1, "longer|wordier")
_lex(_COMPARATIVE, "verbosity", -1, "shorter|briefer|terser")
_lex(_COMPARATIVE, "riskTolerance", +1, "bolder|braver")
_lex(_COMPARATIVE, "riskTolerance", -1, "safer")
_lex(_COMPARATIVE, "formality", -1, "chiller")

# Praise never steers down. Evaluative down-forms ("too X", "stop being so X",
# "not so X") skip these words; "less kind" (explicit) still steers.
PRAISE_WORDS = frozenset({
    "kind", "caring", "gentle", "warm", "warmth", "understanding", "empathetic",
    "empathy", "affectionate", "funny", "witty", "wit", "humorous",
})
_PRAISE_IDIOM = re.compile(
    r"\b(?:you(?:'re|\s+are)\s+(?:\w+\s+){0,2}too\s+(?:kind|sweet|nice|good\s+to\s+me|"
    r"generous|funny|much)|stop\s+being\s+so\s+(?:kind|sweet|nice|funny)|"
    r"(?:that's|that\s+is|how)\s+(?:so\s+)?(?:kind|sweet|nice)\s+of\s+you)\b",
    re.I,
)


def _alt(words) -> str:
    return "|".join(re.escape(w).replace(r"\ ", r"\s+")
                    for w in sorted(words, key=len, reverse=True))


_B, _C = _alt(_BASE), _alt(_COMPARATIVE)
# Fillers / an address ("Sam, ...") / a request frame may precede a command.
# The command must open its clause: "be wittier" steers, "my boss could be
# wittier" doesn't. "no" is a filler only when "more" doesn't follow it.
_LEAD = (
    r"^(?:(?:ok(?:ay)?|hey|please|and|also|now|just|so|alright|honestly|seriously|"
    r"always|yeah|yes|no(?!\s+more\b)|but)[\s,!.]+)*"
    r"(?:[a-z]+,\s*)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:try\s+(?:to\s+)?)?|"
    r"i\s+(?:want|need)\s+you\s+to\s+|i(?:'d|\s+would)\s+like\s+you\s+to\s+|"
    r"you\s+(?:can|could|should|may)\s+|try\s+(?:to\s+)?|feel\s+free\s+to\s+)?"
    r"(?:please\s+)?"
)
_INT = r"(?:(?:a\s+)?(?:bit|little|lot|tad|touch)\s+|(?:way|much|even|slightly|somewhat)\s+)*"
_TAIL = (r"(?:[\s,]+(?:please|pls|from\s+you|with\s+me|now|again|"
         r"in\s+your\s+(?:replies|responses|answers)))*[\s.!]*$")

_PATTERNS: list[tuple[re.Pattern, str]] = [(re.compile(p, re.I), k) for p, k in [
    (_LEAD + r"no\s+more\s+(?P<w>" + _B + r")\b", "neg"),
    (_LEAD + r"(?:be|get|sound|act|go|try\s+being)\s+" + _INT
     + r"(?P<mod>more|less)\s+(?P<w>" + _B + r")\b", "mod"),
    (_LEAD + r"(?:be|get|sound|act)\s+" + _INT + r"(?P<w>" + _C + r")\b", "comp"),
    (_LEAD + _INT + r"(?P<mod>more|less)\s+(?P<w>" + _B + r")" + _TAIL, "mod"),
    (_LEAD + _INT + r"(?P<w>" + _C + r")" + _TAIL, "comp"),
    (_LEAD + r"(?:i(?:'d|\s+would)\s+like|i\s+want|give\s+me)\s+" + _INT
     + r"(?P<mod>more|less)\s+(?P<w>" + _B + r")\b", "mod"),
    (_LEAD + r"(?:dial|turn|crank|tone|ramp)\s+(?:it\s+)?(?P<ud>up|down)\s+(?:on\s+)?"
     r"(?:the\s+|your\s+)?(?P<w>" + _B + r")\b", "ud"),
    (_LEAD + r"(?:dial|turn|crank|tone|ramp)\s+(?:the\s+|your\s+)?(?P<w>" + _B
     + r")\s+(?P<ud>up|down)\b", "ud"),
    (_LEAD + r"(?:you(?:'re|\s+are)\s+(?:being\s+)?)?" + _INT + r"too\s+(?P<w>" + _B + r")\b",
     "eval"),
    (_LEAD + r"(?:(?:stop\s+being|quit\s+being|don'?t\s+be|do\s+not\s+be)\s+(?:so\s+)?|"
     r"not\s+so\s+)(?P<w>" + _B + r")\b", "eval"),
]]

_QUESTION_OPENER = re.compile(
    r"^(?:why|what|how|who|when|where|are|is|was|were|am|do|does|did|have|has)\b", re.I)

RETRACTION = re.compile(
    r"\b(?:actually\s+(?:don'?t|no|not|never\s*mind)|never\s*mind|nevermind|"
    r"scratch\s+that|(?<!don't\s)(?<!dont\s)(?<!not\s)forget\s+(?:that|it|i\s+said\s+that)|wait,?\s+no|no,?\s+wait|"
    r"just\s+kidding|jk|ignore\s+(?:that|what\s+i\s+said)|cancel\s+that|"
    r"on\s+second\s+thought|don'?t\s+do\s+that|i\s+take\s+(?:that|it)\s+back)\b",
    re.I,
)
# Clauses: sentences, plus ", and" / "and" / "but" joins.
_CLAUSE = re.compile(r"[^.!?\n;]+?(?=(?:,?\s+(?:and|but)\s+)|[.!?\n;]|$)", re.I)


@dataclass(frozen=True)
class Steer:
    trait: str
    direction: float  # +1 toward the trait's high pole, -1 toward low
    match: str


def last_retraction_end(text: str) -> int:
    """Offset just past the last retraction marker, or -1 if none."""
    ends = [m.end() for m in RETRACTION.finditer(text or "")]
    return max(ends) if ends else -1


def has_praise(text: str) -> bool:
    return bool(_PRAISE_IDIOM.search(text or ""))


def detect_steering(text: str) -> list[Steer]:
    """Steering commands in `text`, one per trait, retractions applied."""
    text = (text or "").strip()
    if not text:
        return []
    found: list[tuple[int, Steer]] = []
    cut = last_retraction_end(text)
    clauses = [(cm.start(), cm.group(0)) for cm in _CLAUSE.finditer(text)]
    if cut >= 0:
        # What follows a retraction starts fresh, even without punctuation
        # ("never mind, be more sarcastic").
        clauses += [(cut + cm.start(), cm.group(0)) for cm in _CLAUSE.finditer(text[cut:])]
    for cstart, raw in clauses:
        clause = raw.strip(" ,;:-—")
        if not clause or _QUESTION_OPENER.match(clause):
            continue
        start = cstart + raw.find(clause)
        for pat, kind in _PATTERNS:
            m = pat.search(clause)
            if not m:
                continue
            word = re.sub(r"\s+", " ", m.group("w").lower())
            if kind == "eval" and word in PRAISE_WORDS:
                break  # a compliment, not a complaint
            trait, pol = (_COMPARATIVE if kind == "comp" else _BASE)[word]
            if kind == "mod":
                sign = 1 if m.group("mod").lower() == "more" else -1
            elif kind == "ud":
                sign = 1 if m.group("ud").lower() == "up" else -1
            elif kind in ("neg", "eval"):
                sign = -1
            else:
                sign = 1
            found.append((start + m.start(), Steer(trait, float(pol * sign), m.group(0).strip())))
            break
    if cut >= 0:
        # A retraction cancels every command that starts before it.
        found = [(pos, st) for pos, st in found if pos >= cut]
    out: list[Steer] = []
    seen: set[str] = set()
    for _, st in found:
        if st.trait not in seen:
            seen.add(st.trait)
            out.append(st)
    return out
