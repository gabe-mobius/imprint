"""Imprint core: learn from each user message, render a calibration block.

    imp = Imprint("me.profile.json")
    imp.learn(user_text, timestamp=message_time)   # before building the prompt
    system_prompt = base_prompt + "\\n\\n" + imp.directive()

Model-agnostic: nothing here calls a chat model. The host supplies messages
(with their timestamps) and a place to inject a text block.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from . import rules as rules_mod
from .embedders import Embedder, default_embedder
from .steering import Steer, detect_steering, has_praise, last_retraction_end
from .traits import (TRAIT_DEFAULTS, TRAIT_META, TRAIT_NAMES, TRAIT_POLES, band_step,
                     describe)

log = logging.getLogger("imprint")

SCHEMA = "imprint.profile/1"
MAX_CONFIDENCE = 0.95
MIN_CONFIDENCE = 0.1
BASE_LR = 0.15
CONF_INC = 0.008
STEER_CONF_INC = CONF_INC * 4
HALF_LIFE = timedelta(days=30)
VIBE_THRESHOLD = 0.16
VIBE_THRESHOLD_PERSON = 0.11          # humor / warmth / flirt score lower in MiniLM
PERSON_TRAITS = frozenset({"humor", "warmth", "flirt"})
MAX_VIBE_TRAITS_PER_MSG = 2
# A score that clears its threshold by a hair is noise, not signal: ignore
# vibe strengths below this (≈ 0.04–0.045 past the threshold).
MIN_VIBE_STRENGTH = 0.05
MAX_OBSERVATIONS = 200
MAX_RULES = 12


@dataclass
class LearnResult:
    steered: list[Steer] = field(default_factory=list)
    vibe: list[dict] = field(default_factory=list)
    rules_added: list[str] = field(default_factory=list)
    candidates_seen: int = 0
    promoted: list[str] = field(default_factory=list)

    @property
    def changed_traits(self) -> bool:
        return bool(self.steered or self.vibe)


def _utc(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(ts, datetime):
        return None
    return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="milliseconds") if dt else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_profile(traits: Iterable[str] = TRAIT_NAMES) -> dict:
    now = _now_iso()
    return {
        "schema": SCHEMA, "created": now, "lastUpdated": now, "interactions": 0,
        "traits": {t: {"value": TRAIT_DEFAULTS[t]["value"],
                       "confidence": TRAIT_DEFAULTS[t]["confidence"],
                       "description": TRAIT_META[t]} for t in traits},
        "rules": [], "candidateRules": [], "observations": [],
    }


class Imprint:
    """One user's Imprint profile.

    profile_path:  JSON file to load/save (None = in memory only).
    embedder:      object with .name and .embed(list[str]) -> (n, d) array.
                   Defaults to local all-MiniLM-L6-v2 (sentence-transformers).
    traits:        subset of the 8 traits to learn and inject (default: all).
    axes_cache:    optional .npz path to cache the embedded trait poles.
    warmup:        interactions before directive() returns anything.
    """

    def __init__(self, profile_path: str | os.PathLike | None = None, *,
                 embedder: Embedder | None = None,
                 traits: Iterable[str] | None = None,
                 axes_cache: str | os.PathLike | None = None,
                 warmup: int = 5,
                 max_inject_rules: int = 4,
                 corroboration_gap: timedelta = timedelta(hours=1),
                 candidate_ttl: timedelta = timedelta(days=90)):
        self.traits = tuple(traits) if traits is not None else TRAIT_NAMES
        unknown = set(self.traits) - set(TRAIT_NAMES)
        if unknown:
            raise ValueError(f"unknown traits: {sorted(unknown)}")
        self.profile_path = Path(profile_path) if profile_path else None
        self.axes_cache = Path(axes_cache) if axes_cache else None
        self.warmup = warmup
        self.max_inject_rules = max_inject_rules
        self.corroboration_gap = corroboration_gap
        self.candidate_ttl = candidate_ttl
        self._embedder = embedder
        self._axes: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self.profile = self._load()

    # --- persistence ------------------------------------------------------------

    def _load(self) -> dict:
        p = new_profile(self.traits)
        if self.profile_path and self.profile_path.exists():
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
            p.update({k: v for k, v in data.items() if k != "traits"})
            for t, v in (data.get("traits") or {}).items():
                if t in TRAIT_NAMES:
                    p["traits"][t] = {**p["traits"].get(t, {}), **v}
        return p

    def save(self) -> None:
        if not self.profile_path:
            return
        self.profile["lastUpdated"] = _now_iso()
        self.profile["observations"] = self.profile["observations"][-MAX_OBSERVATIONS:]
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.profile_path.with_suffix(self.profile_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.profile, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(self.profile_path)

    # --- embedding axes -----------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = default_embedder()
        return self._embedder

    def _fingerprint(self) -> str:
        blob = json.dumps([self.embedder.name, TRAIT_POLES], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def axes(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self._axes is not None:
            return self._axes
        fp = self._fingerprint()
        if self.axes_cache and self.axes_cache.exists():
            data = np.load(self.axes_cache, allow_pickle=False)
            if bytes(data["fingerprint"]).decode() == fp:
                self._axes = {t: (data[f"{t}__high"], data[f"{t}__low"]) for t in TRAIT_POLES}
                return self._axes
        axes = {t: (_centroid(self.embedder, p["high"]), _centroid(self.embedder, p["low"]))
                for t, p in TRAIT_POLES.items()}
        if self.axes_cache:
            self.axes_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.axes_cache, fingerprint=np.frombuffer(fp.encode(), dtype=np.uint8),
                     **{f"{t}__{k}": v for t, (h, l) in axes.items()
                        for k, v in (("high", h), ("low", l))})
        self._axes = axes
        return axes

    def score(self, text: str) -> dict[str, float]:
        """Bipolar vibe score per trait: cos(text, high) − cos(text, low)."""
        m = _unit(self.embedder.embed([text])[0])
        return {t: float(m @ h - m @ l) for t, (h, l) in self.axes().items()
                if t in self.traits}

    # --- learning -------------------------------------------------------------------

    def vibe_signals(self, text: str) -> list[dict]:
        scores = self.score(text)
        person_clears = any(abs(scores.get(t, 0)) > VIBE_THRESHOLD_PERSON for t in PERSON_TRAITS)
        cands = []
        for t, s in scores.items():
            thr = VIBE_THRESHOLD_PERSON if t in PERSON_TRAITS else VIBE_THRESHOLD
            if abs(s) <= thr:
                continue
            if t == "formality" and person_clears:
                continue  # person-shaped asks read "casual" to small embedders
            strength = min(1.0, (abs(s) - thr) / (1 - thr))
            if strength < MIN_VIBE_STRENGTH:
                continue
            cands.append((t, s, strength))
        cands.sort(key=lambda c: -abs(c[1]))
        return [{"trait": t, "direction": 1.0 if s > 0 else -1.0, "strength": st,
                 "score": round(s, 4)} for t, s, st in cands[:MAX_VIBE_TRAITS_PER_MSG]]

    def learn(self, text: str, timestamp=None, *, save: bool = True) -> LearnResult:
        """Learn from one user message. `timestamp` = when the user sent it.

        Without a timestamp, observations record null (never processing time),
        no confidence decay is applied, and candidates can't be corroborated.
        """
        res = LearnResult()
        text = (text or "").strip()
        ts = _utc(timestamp)
        if ts is None:
            log.warning("imprint: no message timestamp (got %r); observation "
                        "timestamps will be null", timestamp)
        p = self.profile
        p["interactions"] = int(p.get("interactions") or 0) + 1
        if not text:
            self._finish(save)
            return res

        res.steered = [s for s in detect_steering(text) if s.trait in self.traits]
        for st in res.steered:
            tr = p["traits"][st.trait]
            self._decay(st.trait, ts)
            before = tr["value"]
            tr["value"] = band_step(st.trait, before, st.direction)
            tr["confidence"] = min(MAX_CONFIDENCE, tr["confidence"] + STEER_CONF_INC)
            self._observe(ts, st.trait, st.direction, "steer",
                          f"{st.match[:80]!r} {before:.3f}→{tr['value']:.3f}")

        if not res.steered:
            cut = last_retraction_end(text)
            standing = text[cut:] if cut >= 0 else text
            if any(c.isalpha() for c in standing):
                sigs = self.vibe_signals(standing)
                if has_praise(text):  # praise never lowers warmth or humor
                    sigs = [s for s in sigs
                            if not (s["trait"] in ("warmth", "humor") and s["direction"] < 0)]
                for s in sigs:
                    tr = p["traits"][s["trait"]]
                    self._decay(s["trait"], ts)
                    lr = BASE_LR * (1 - tr["confidence"]) * s["strength"]
                    tr["value"] = max(0.0, min(1.0, tr["value"] + s["direction"] * lr))
                    tr["confidence"] = min(MAX_CONFIDENCE, tr["confidence"] + CONF_INC)
                    self._observe(ts, s["trait"], s["direction"], "vibe", f"score={s['score']:+.3f}")
                res.vibe = sigs

        cap = rules_mod.capture(text)
        for r in cap.rules:
            if not any(rules_mod.rules_near_dup(r, e) for e in p["rules"]):
                p["rules"].append(r)
                res.rules_added.append(r)
        res.candidates_seen = len(cap.candidates)
        _, promoted = rules_mod.update_candidates(
            p.setdefault("candidateRules", []), cap.candidates, ts,
            gap=self.corroboration_gap, ttl=self.candidate_ttl)
        for r in promoted:
            if not any(rules_mod.rules_near_dup(r, e) for e in p["rules"]):
                p["rules"].append(r)
                res.promoted.append(r)
        p["rules"] = p["rules"][-MAX_RULES:]
        self._finish(save)
        return res

    def _finish(self, save: bool) -> None:
        if save:
            self.save()

    def _observe(self, ts, trait, direction, source, detail) -> None:
        self.profile["observations"].append({
            "timestamp": _iso(ts), "trait": trait, "direction": direction,
            "source": source, "detail": detail})

    def _decay(self, trait: str, ts: datetime | None) -> None:
        if ts is None:
            return  # no message time, no decay (never substitute processing time)
        last = next((o["timestamp"] for o in reversed(self.profile["observations"])
                     if o["trait"] == trait and o.get("timestamp")), None)
        last_dt = _utc(last)
        if last_dt is None or ts <= last_dt:
            return
        tr = self.profile["traits"][trait]
        f = 2 ** (-(ts - last_dt) / HALF_LIFE)
        tr["confidence"] = max(MIN_CONFIDENCE, MIN_CONFIDENCE + (tr["confidence"] - MIN_CONFIDENCE) * f)

    # --- output -------------------------------------------------------------------------

    def directive(self) -> str:
        """The calibration block to inject into the system prompt.

        Byte-stable between learning events (no counters), so it can sit in a
        cached system prompt. Empty until `warmup` interactions.
        """
        p = self.profile
        if int(p.get("interactions") or 0) < self.warmup:
            return ""
        lines = ["Learned preferences for this user. Follow them unless the user asks otherwise.", ""]
        lines += [describe(t, p["traits"][t]["value"]) for t in TRAIT_NAMES if t in self.traits]
        injected = p["rules"][-self.max_inject_rules:] if self.max_inject_rules else []
        if injected:
            lines += ["", "Standing rules from the user:"] + [f"- {r}" for r in injected]
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return deepcopy(self.profile)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _centroid(embedder: Embedder, phrases: list[str]) -> np.ndarray:
    return _unit(np.mean(np.asarray(embedder.embed(phrases), dtype=np.float32), axis=0))
