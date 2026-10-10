from __future__ import annotations

import json
from datetime import timedelta

import pytest
from conftest import TS, FakeEmbedder

from imprint import Imprint, band_step, detect_steering
from imprint.dashboard import render_html
from imprint.traits import BANDS, TRAIT_NAMES


def val(imp, t):
    return imp.profile["traits"][t]["value"]


def line(imp, label):
    return next(x for x in imp.directive().splitlines() if f"**{label}:**" in x)


def set_val(imp, t, v):
    imp.profile["traits"][t]["value"] = v


# --- band-step steering ----------------------------------------------------------

def test_named_command_crosses_band_same_turn(imp):
    set_val(imp, "humor", 0.58)
    before = line(imp, "Humor")
    r = imp.learn("be wittier", TS)
    assert [s.trait for s in r.steered] == ["humor"]
    assert val(imp, "humor") == pytest.approx(0.76)
    assert line(imp, "Humor") != before


@pytest.mark.parametrize("text", ["more sarcastic", "be wittier", "be funnier",
                                  "Sam, a bit more sarcasm please", "can you be more witty?",
                                  "dial up the sarcasm", "wittier please"])
def test_plain_steering_phrases(text):
    st = detect_steering(text)
    assert st and st[0].trait == "humor" and st[0].direction > 0, text


def test_repeat_steps_one_more_band_then_stops(imp):
    set_val(imp, "flirt", 0.4)
    for expect in (0.56, 0.76, 0.76):
        imp.learn("be flirtier", TS)
        assert val(imp, "flirt") == pytest.approx(expect)


def test_less_walks_down(imp):
    set_val(imp, "humor", 0.76)
    imp.learn("less sarcastic please", TS)
    assert val(imp, "humor") == pytest.approx(0.74)


def test_all_eight_dials_steerable():
    cases = {"verbosity": "be more detailed", "autonomy": "be more independent",
             "formality": "be more formal", "proactivity": "be more proactive",
             "riskTolerance": "be bolder", "humor": "be funnier", "warmth": "be warmer",
             "flirt": "be flirtier"}
    for t, text in cases.items():
        st = detect_steering(text)
        assert st and st[0].trait == t and st[0].direction > 0, text


def test_compound_clause(imp):
    assert {(s.trait, s.direction) for s in detect_steering("be wittier and warmer")} == {
        ("humor", 1.0), ("warmth", 1.0)}


def test_band_step_matches_band_ladders():
    for t in TRAIT_NAMES:
        edges, _ = BANDS[t]
        assert band_step(t, edges[0] - 0.05, +1) == pytest.approx(edges[0] + 0.01)
        assert band_step(t, edges[-1] + 0.05, +1) == pytest.approx(edges[-1] + 0.05)


def test_steer_bumps_confidence(imp):
    c0 = imp.profile["traits"]["humor"]["confidence"]
    imp.learn("be wittier", TS)
    assert imp.profile["traits"]["humor"]["confidence"] == pytest.approx(c0 + 0.032)


def test_disabled_trait_is_never_learned_or_injected(tmp_path):
    imp = Imprint(tmp_path / "p.json", embedder=FakeEmbedder(), warmup=0,
                  traits=[t for t in TRAIT_NAMES if t != "flirt"])
    r = imp.learn("be flirtier", TS)
    assert r.steered == [] and "flirt" not in imp.profile["traits"]
    assert "**Flirt:**" not in imp.directive()


# --- negatives ----------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Morning, how did you sleep?", "long day at work, finally home",
                                  "what's on my calendar tomorrow?", "we got donuts like normal"])
def test_check_in_changes_no_trait(imp, text):
    before = {t: v["value"] for t, v in imp.profile["traits"].items()}
    r = imp.learn(text, TS)
    assert not r.changed_traits
    assert {t: v["value"] for t, v in imp.profile["traits"].items()} == before
    assert imp.profile["interactions"] == 1


def test_just_handle_it_does_not_steer_autonomy(imp):
    a0 = val(imp, "autonomy")
    r = imp.learn("the plumber thing, just handle it", TS)
    assert r.steered == [] and val(imp, "autonomy") == a0


@pytest.mark.parametrize("text", ["why so sarcastic?", "are you being sarcastic?",
                                  "that was sarcastic lol", "my boss could be funnier",
                                  "more detail on the second point", "not funny", "tell me more",
                                  'he said "be more sarcastic"'])
def test_mentions_and_questions_do_not_steer(text):
    assert detect_steering(text) == []


# --- negation, retraction, praise -----------------------------------------------------

@pytest.mark.parametrize("text", ["no more sarcastic", "no more sarcasm please", "ok no more snark."])
def test_no_more_x_steers_down(text):
    assert [(s.trait, s.direction) for s in detect_steering(text)] == [("humor", -1.0)]


@pytest.mark.parametrize("text", ["be more sarcastic. actually don't.", "be wittier — wait no, don't",
                                  "be warmer. jk", "dial up the sarcasm. on second thought, forget it"])
def test_retraction_cancels(imp, text):
    before = {t: v["value"] for t, v in imp.profile["traits"].items()}
    r = imp.learn(text, TS)
    assert r.steered == [] and r.vibe == []
    assert {t: v["value"] for t, v in imp.profile["traits"].items()} == before


def test_command_after_retraction_stands():
    assert [(s.trait, s.direction) for s in detect_steering("be colder. scratch that. be warmer")] == [
        ("warmth", 1.0)]


@pytest.mark.parametrize("text", ["you're too kind", "aw stop, you're too kind", "you're too funny",
                                  "stop being so funny lol", "that's so kind of you"])
def test_praise_never_steers_or_lowers(imp, text):
    w0, h0 = val(imp, "warmth"), val(imp, "humor")
    r = imp.learn(text, TS)
    assert r.steered == []
    assert val(imp, "warmth") >= w0 and val(imp, "humor") >= h0


def test_praise_then_complaint_in_one_message():
    assert [(s.trait, s.direction) for s in detect_steering("you're too kind and too sarcastic")] == [
        ("humor", -1.0)]


@pytest.mark.parametrize("text,trait,d", [("you're too sarcastic", "humor", -1.0),
                                          ("you're way too wordy", "verbosity", -1.0),
                                          ("you're too cautious", "riskTolerance", 1.0)])
def test_complaint_too_x(text, trait, d):
    assert [(s.trait, s.direction) for s in detect_steering(text)] == [(trait, d)]


# --- tier 1 rules ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expect", [
    ("always: keep replies under three sentences", "Always keep replies under three sentences"),
    ("never use emoji", "Never use emoji"),
    ("remember: I prefer metric units", "I prefer metric units"),
    ("from now on, use metric units", "Use metric units"),
    ("never mind, always: keep it short", "Always keep it short"),
])
def test_explicit_rules_stored_now(imp, text, expect):
    r = imp.learn(text, TS)
    assert expect in r.rules_added and expect in imp.profile["rules"]
    assert expect in imp.directive()


@pytest.mark.parametrize("text", [
    "always: send this to Dana", "always send the report to Dana", "always: do it now",
    "always: email her before noon", "remember: the invoice is due by Friday",
    "remember: call the plumber at 3", "always: check the doc tomorrow", "never do that again",
    "Never mind, I figured it out", "Always the same with you, huh", "always: can you do that?",
    "always: use emoji, actually never mind", "always be wittier",
])
def test_one_offs_and_non_rules_not_stored(imp, text):
    imp.learn(text, TS)
    assert imp.profile["rules"] == [], text


# --- tier 2 candidates -------------------------------------------------------------------

def test_natural_preference_is_candidate_not_rule(imp):
    r = imp.learn("I prefer metric units", TS)
    assert r.candidates_seen == 1 and r.rules_added == [] and imp.profile["rules"] == []
    assert "metric" not in imp.directive()


def test_candidate_promoted_on_corroboration(imp):
    imp.learn("I prefer metric units", TS)
    imp.learn("honestly I really prefer metric units.", TS + timedelta(minutes=10))
    assert imp.profile["rules"] == []                       # too soon to count
    r = imp.learn("I prefer metric units for everything", TS + timedelta(days=2))
    assert r.promoted == ['Stated preference: "I prefer metric units"']
    assert imp.profile["candidateRules"] == []
    assert 'Stated preference: "I prefer metric units"' in imp.directive()


def test_negative_preference_and_contradiction(imp):
    imp.learn("I hate it when you use bullet points", TS)
    imp.learn("I like it when you use bullet points", TS + timedelta(days=1))
    c = imp.profile["candidateRules"]
    assert len(c) == 1 and c[0]["polarity"] == 1 and imp.profile["rules"] == []


def test_task_anchored_natural_statement_vetoed(imp):
    r = imp.learn("I'd rather you send it to Dana tomorrow", TS)
    assert r.candidates_seen == 0


def test_candidate_needs_timestamps_to_corroborate(imp):
    imp.learn("I prefer metric units")
    imp.learn("I prefer metric units")
    assert imp.profile["rules"] == []


def test_candidate_expires(imp):
    imp.learn("I prefer metric units", TS)
    imp.learn("long day at work", TS + timedelta(days=120))
    assert imp.profile["candidateRules"] == []


# --- timestamps, warmup, persistence ------------------------------------------------------

def test_observation_carries_message_time(imp):
    imp.learn("be wittier", TS)
    assert imp.profile["observations"][-1]["timestamp"] == "2026-10-09T08:15:00.000+00:00"


def test_missing_timestamp_is_null_and_warns(imp, caplog):
    c0 = imp.profile["traits"]["humor"]["confidence"]
    imp.learn("be wittier")
    assert imp.profile["observations"][-1]["timestamp"] is None
    assert imp.profile["traits"]["humor"]["confidence"] == pytest.approx(c0 + 0.032)
    assert any("no message timestamp" in r.getMessage() for r in caplog.records)


def test_naive_and_epoch_timestamps_are_utc(imp):
    imp.learn("be wittier", "2026-10-02T09:30:00")
    assert imp.profile["observations"][-1]["timestamp"] == "2026-10-02T09:30:00.000+00:00"
    imp.learn("be flirtier", 0)
    assert imp.profile["observations"][-1]["timestamp"] == "1970-01-01T00:00:00.000+00:00"


def test_warmup_and_interaction_counting(tmp_path):
    imp = Imprint(tmp_path / "p.json", embedder=FakeEmbedder(), warmup=3)
    for i in range(2):
        imp.learn("hello there", TS)
        assert imp.directive() == ""
    imp.learn("hello there", TS)
    assert imp.profile["interactions"] == 3 and imp.directive()


def test_directive_is_byte_stable_without_learning(imp):
    imp.learn("hello", TS)
    assert imp.directive() == imp.directive()
    imp.learn("long day at work", TS)
    assert "interactions" not in imp.directive()


def test_profile_round_trip(tmp_path):
    a = Imprint(tmp_path / "p.json", embedder=FakeEmbedder(), warmup=0)
    a.learn("be wittier", TS)
    a.learn("never use emoji", TS)
    b = Imprint(tmp_path / "p.json", embedder=FakeEmbedder(), warmup=0)
    assert b.profile["traits"]["humor"]["value"] == a.profile["traits"]["humor"]["value"]
    assert b.directive() == a.directive()


def test_axes_cache_keyed_to_embedder(tmp_path):
    a = Imprint(None, embedder=FakeEmbedder(), axes_cache=tmp_path / "axes.npz")
    s1 = a.score("be brief")
    b = Imprint(None, embedder=FakeEmbedder(), axes_cache=tmp_path / "axes.npz")
    assert b.score("be brief") == s1 and (tmp_path / "axes.npz").exists()


def test_vibe_moves_a_dial_without_steer_syntax(imp):
    v0 = val(imp, "verbosity")
    r = imp.learn("too much filler, you keep repeating yourself, cut it down", TS)
    assert r.steered == [] and any(s["trait"] == "verbosity" for s in r.vibe)
    assert val(imp, "verbosity") < v0


# --- dashboard -------------------------------------------------------------------------------

def test_dashboard_renders_sample_profile():
    from pathlib import Path
    sample = json.loads((Path(__file__).resolve().parents[1] / "examples" /
                         "sample.profile.json").read_text())
    out = render_html(sample)
    assert "<html" in out and "Humor" in out and "Standing rules" in out


@pytest.mark.parametrize("text", ["never mind, be more sarcastic", "jk be wittier",
                                  "scratch that, more sarcasm please"])
def test_unpunctuated_command_after_retraction_stands(text):
    st = detect_steering(text)
    assert [(x.trait, x.direction) for x in st] == [("humor", 1.0)], text


def test_dont_forget_it_is_not_a_retraction():
    assert [(x.trait, x.direction) for x in detect_steering("be wittier. don't forget it")] == [
        ("humor", 1.0)]


@pytest.mark.parametrize("text", ["no more funny business", "no more witty remarks",
                                  "no more kind words"])
def test_negation_applies_to_praise_class_words(text):
    st = detect_steering(text)
    assert st and st[0].direction == -1.0, text
