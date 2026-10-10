"""Two-tier standing-rule capture.

Tier 1 — explicit forms become rules immediately:
    "remember: ...", "always ..." / "never ..." (bare form must be followed by
    an imperative verb), "always: ..." / "never: ...", "from now on ...",
    "going forward ...".

Tier 2 — natural preference statements become *candidate* rules:
    "I prefer ...", "I like it when you ...", "I hate it when you ...",
    "it bugs me when you ...", "I'd rather you ...", "I wish you wouldn't ...".
    A candidate is promoted to a rule only when corroborated: the same
    preference (same polarity, near-duplicate content) stated again in a
    different message at least `corroboration_gap` later. Candidates are never
    injected. Users don't learn any syntax; they just talk.

Both tiers apply the same durability test. A rule must state an enduring
preference: questions, time-bound text ("today", "at 3"), and task anchors
(a named recipient, "now", a deadline, "this"/"that one") veto storage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .steering import RETRACTION, detect_steering

# --- durability ----------------------------------------------------------------

_ONE_OFF = re.compile(
    r"\b(?:today|tonight|tomorrow|yesterday|right\s+now|for\s+now|just\s+now|asap|"
    r"later\s+(?:today|tonight)|"
    r"this\s+(?:morning|afternoon|evening|week|weekend|month|time|once)|"
    r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:at|by|before|until)\s+\d|in\s+\d+\s?(?:min(?:ute)?s?|hours?|hrs?|days?)|\d{1,2}(?::\d{2})?\s?(?:am|pm))\b",
    re.I,
)
_TASK_ANCHOR = re.compile(
    r"\b(?:now|(?:before|by|until|till|due)\s+(?:noon|midnight|tonight|tomorrow|eod|cob|"
    r"end\s+of|the\s+end|next|this|\d|monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday)|deadline|due\s+(?:date|on)|this|these|those|that\s+(?:one|thing|stuff|again)|"
    r"(?:send|email|text|call|message|ping|forward|remind|tell|cc)\s+(?:her|him|them))\b",
    re.I,
)
# A capitalized name after a recipient verb/preposition. Lowercase names can't
# be told apart from ordinary words, so "to dana" is a known gap.
_NOT_NAMES = {"I", "English", "Spanish", "French", "German", "Markdown", "Python",
              "JSON", "SQL", "AI", "The", "My", "Your"}
_RECIPIENT = re.compile(
    r"\b(?i:to|for|with|cc|email|text|call|message|tell|ping|remind|ask|send|forward)"
    r"\s+([A-Z][a-zA-Z]+)\b")


def has_task_anchor(body: str) -> bool:
    b = re.sub(r"\bfrom\s+now\s+on\b", " ", body or "", flags=re.I)
    if _TASK_ANCHOR.search(b):
        return True
    return any(m.group(1) not in _NOT_NAMES for m in _RECIPIENT.finditer(b))


def is_durable(body: str) -> bool:
    b = (body or "").strip()
    if not 4 <= len(b) <= 200 or b.endswith("?"):
        return False
    return not (_ONE_OFF.search(b) or has_task_anchor(b))


# --- tier 1: explicit forms ------------------------------------------------------

_RULE_VERBS = frozenset("""
    be use ask answer reply respond keep call mention say talk start end open close
    include give put write check confirm tell lead skip avoid stop let make match refer
    address sign greet offer suggest apologize apologise hedge cite show explain assume
    repeat restate speak remind bring do try follow treat wait act pretend lie joke flirt
    swear curse break drop add format double-check cut leave share wrap read look take
    pick choose respect interrupt lecture moralize moralise summarize summarise list
    number quote push nag fake invent guess correct
""".split())
_REMEMBER = re.compile(r"^\s*remember\s*:\s*(.+)$", re.I)
_ALWAYS_NEVER = re.compile(r"^\s*(always|never)\s*(:)?\s+(.+)$", re.I)
_FROM_NOW_ON = re.compile(r"^\s*(?:from\s+now\s+on|going\s+forward)\s*[,:]?\s+(.+)$", re.I)

# --- tier 2: natural preference statements -----------------------------------------

_POS = (r"i\s+(?:really\s+|much\s+)?(?:prefer|like\s+it\s+when\s+you|love\s+it\s+when\s+you|"
        r"appreciate\s+it\s+when\s+you|wish\s+you\s+would(?!n)|'?d\s+rather\s+you|"
        r"would\s+rather\s+you)")
_NEG = (r"i\s+(?:really\s+)?(?:hate\s+(?:it\s+)?when\s+you|don'?t\s+like\s+(?:it\s+)?when\s+you|"
        r"can'?t\s+stand\s+(?:it\s+)?when\s+you|wish\s+you\s+wouldn'?t)|"
        r"it\s+(?:really\s+)?(?:annoys|bugs|bothers|irritates)\s+me\s+when\s+you")
_NATURAL = re.compile(r"^\s*(?:(?:honestly|also|and|but|so|fwiw)[,\s]+)?"
                      r"(?:(?P<pos>" + _POS + r")|(?P<neg>" + _NEG + r"))\s+(?P<body>.+?)[.!]*$",
                      re.I)

_STOP = {"a", "an", "the", "to", "of", "for", "and", "or", "you", "your", "it", "is", "be",
         "i", "me", "my", "when", "that", "with", "in", "on", "so", "really", "always",
         "never", "just", "please", "do", "dont", "don't", "are", "more"}


def content_key(body: str) -> frozenset[str]:
    toks = re.findall(r"[a-z0-9']+", (body or "").lower())
    return frozenset(t.rstrip("s") if len(t) > 3 else t for t in toks if t not in _STOP)


def keys_match(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    if a <= b or b <= a:
        return min(len(a), len(b)) >= 2 or a == b
    return len(a & b) / len(a | b) >= 0.6


def _norm(rule: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", (rule or "").lower())).strip()


def rules_near_dup(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long_ = sorted((na, nb), key=len)
    if len(short) >= 24 and short in long_:
        return True
    return keys_match(content_key(a), content_key(b)) and len(content_key(a)) >= 3


@dataclass
class Capture:
    rules: list[str]          # tier 1: store now
    candidates: list[dict]    # tier 2: {"text", "polarity"}


def capture(text: str) -> Capture:
    """Split `text` into explicit rules and natural-preference candidates."""
    rules: list[str] = []
    candidates: list[dict] = []
    text = text or ""
    cut = max((m.end() for m in RETRACTION.finditer(text)), default=-1)
    # Each line, then each sentence within it, is checked independently.
    for sm in re.finditer(r"[^\n.!?]+[.!?]*", text):
        if sm.end() <= cut:
            continue  # retracted later in the same message
        # A retraction inside this segment cancels what precedes it; keep
        # whatever the user said after it ("never mind, always: keep it short").
        seg = (text[cut:sm.end()] if sm.start() < cut else sm.group(0)).strip(" \t,;:-—")
        if not seg:
            continue
        rule = _explicit(seg)
        if rule:
            if not any(rules_near_dup(rule, r) for r in rules):
                rules.append(rule)
            continue
        m = _NATURAL.match(seg)
        if m and is_durable(m.group("body")) and not detect_steering(seg):
            candidates.append({"text": seg.rstrip(".! "),
                               "polarity": 1 if m.group("pos") else -1,
                               "key": sorted(content_key(m.group("body")))})
    return Capture(rules, candidates)


def _explicit(seg: str) -> str | None:
    m = _REMEMBER.match(seg)
    if m:
        body = m.group(1).strip()
        return body.rstrip(".") if is_durable(body) and not detect_steering(body) else None
    m = _FROM_NOW_ON.match(seg)
    if m:
        body = m.group(1).strip()
        if is_durable(body) and not detect_steering(body):
            return f"{body[0].upper()}{body[1:]}".rstrip(".")
        return None
    m = _ALWAYS_NEVER.match(seg)
    if not m:
        return None
    cue, colon, body = m.group(1), m.group(2), m.group(3).strip()
    if not is_durable(body):
        return None
    if not colon:
        first = re.sub(r"[^a-z'-]", "", body.split()[0].lower())
        if first not in _RULE_VERBS:
            return None  # "Never mind", "Always the same with you"
    if detect_steering(body) or detect_steering(seg):
        return None  # a steering command is a dial move, not a rule
    return f"{cue.capitalize()} {body}".rstrip(".")


# --- candidate store -----------------------------------------------------------------

def update_candidates(store: list[dict], new: list[dict], ts: datetime | None,
                      *, gap: timedelta, ttl: timedelta, max_candidates: int = 50,
                      ) -> tuple[list[dict], list[str]]:
    """Merge new candidates into `store`; return (store, promoted rule texts).

    Corroboration needs a second statement with the same polarity and matching
    content, at least `gap` after the first evidence. Without a message
    timestamp a statement is recorded but cannot corroborate.
    """
    promoted: list[str] = []
    iso = ts.isoformat(timespec="milliseconds") if ts else None
    for cand in new:
        key = frozenset(cand["key"])
        hit = next((c for c in store if keys_match(frozenset(c["key"]), key)), None)
        if hit is None:
            store.append({"text": cand["text"], "polarity": cand["polarity"],
                          "key": cand["key"], "evidence": [iso]})
            continue
        if hit["polarity"] != cand["polarity"]:
            # The user changed their mind: restart from the newer statement.
            hit.update({"text": cand["text"], "polarity": cand["polarity"],
                        "key": cand["key"], "evidence": [iso]})
            continue
        first = _first_ts(hit["evidence"])
        if ts is not None and first is not None and ts - first >= gap:
            promoted.append(f'Stated preference: "{hit["text"]}"')
            store.remove(hit)
        else:
            hit["evidence"].append(iso)
    if ts is not None:
        store[:] = [c for c in store
                    if (_last_ts(c["evidence"]) is None) or ts - _last_ts(c["evidence"]) <= ttl]
    del store[:-max_candidates]
    return store, promoted


def _parse(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


def _first_ts(evidence: list) -> datetime | None:
    ts = [t for t in (_parse(e) for e in evidence) if t]
    return min(ts) if ts else None


def _last_ts(evidence: list) -> datetime | None:
    ts = [t for t in (_parse(e) for e in evidence) if t]
    return max(ts) if ts else None
