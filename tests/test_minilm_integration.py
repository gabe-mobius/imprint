"""Optional: the same behavioral checks with the real default embedder.

Skipped unless sentence-transformers and the all-MiniLM-L6-v2 weights are
available locally. Run: IMPRINT_TEST_MINILM=1 pytest tests/test_minilm_integration.py
"""
from __future__ import annotations

import os

import pytest
from conftest import TS

if not os.environ.get("IMPRINT_TEST_MINILM"):
    pytest.skip("set IMPRINT_TEST_MINILM=1 to run with the real embedder", allow_module_level=True)

from imprint import Imprint, MiniLMEmbedder  # noqa: E402

EMB = MiniLMEmbedder()


@pytest.fixture()
def imp(tmp_path):
    return Imprint(tmp_path / "p.json", embedder=EMB, warmup=0)


@pytest.mark.parametrize("text", ["Morning, how did you sleep?", "long day at work, finally home",
                                  "what's on my calendar tomorrow?", "we got donuts like normal",
                                  "the plumber thing, just handle it"])
def test_check_ins_do_not_move_dials(imp, text):
    assert not imp.learn(text, TS).changed_traits


@pytest.mark.parametrize("text,trait,d", [
    ("you're repeating yourself, too much filler", "verbosity", -1.0),
    ("can you add some sarcastic humor to your responses occasionally?", "humor", 1.0),
    ("keep it all business, less emotional", "warmth", -1.0),
])
def test_vibe_without_steer_syntax(imp, text, trait, d):
    r = imp.learn(text, TS)
    assert any(s["trait"] == trait and s["direction"] == d for s in r.vibe), r.vibe


def test_praise_vibe_never_lowers_warmth(imp):
    w0 = imp.profile["traits"]["warmth"]["value"]
    imp.learn("aw you're too kind, thank you", TS)
    assert imp.profile["traits"]["warmth"]["value"] >= w0
