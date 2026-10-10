"""Deterministic fake embedder: hashed bag of content words. No model download."""
from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imprint import Imprint  # noqa: E402

_STOP = set("a an the to of for and or you your i me my it is be are was this that with in on at "
            "so just do don't dont not no please can could would will like some more less".split())


class FakeEmbedder:
    name = "fake-hash-bow/4096"

    def embed(self, texts):
        out = np.zeros((len(texts), 4096), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z']+", t.lower()):
                if w in _STOP:
                    continue
                w = w[:-1] if len(w) > 4 and w.endswith("s") else w
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % 4096] += 1.0 if (h >> 9) & 1 else -1.0
        return out


TS = datetime(2026, 10, 9, 8, 15, tzinfo=timezone.utc)


@pytest.fixture()
def imp(tmp_path) -> Imprint:
    return Imprint(tmp_path / "p.profile.json", embedder=FakeEmbedder(), warmup=0)
