import numpy as np

from asr_tier_audit import certify, metrics


def test_whisper_basic_splits_turkish_dotted_capital():
    assert metrics.normalize("İZMİR", "tr", "whisper_basic") == "i zmi r"
    assert metrics.normalize("İZMİR", "tr", "lang_join") == "izmir"


def test_turkish_dotless_capital_i():
    assert metrics.normalize("ILIK", "tr", "lang_join") == "ılık"
    assert metrics.normalize("ILIK", "fi", "lang_join") == "ilik"


def test_whisper_basic_drops_bracketed_words():
    assert metrics.normalize("risteävät (katso esimerkkiä).", "fi", "whisper_basic") == "risteävät"
    assert metrics.normalize("risteävät (katso esimerkkiä).", "fi", "lang_join") == \
        "risteävät katso esimerkkiä"


def test_apostrophe_join_and_split():
    assert metrics.normalize("İstanbul'da", "tr", "lang_join") == "istanbulda"
    assert metrics.normalize("İstanbul'da", "tr", "lang_split") == "istanbul da"


def test_malayalam_chillu_variants_unify():
    atomic = "അവൻ"                      # atomic chillu n (U+0D7B)
    legacy = "അവന്‍"          # na + virama + ZWJ
    assert metrics.normalize(atomic, "ml", "lang_join") == metrics.normalize(legacy, "ml", "lang_join")


def test_japanese_scored_on_characters():
    assert metrics.normalize("東京です。", "ja", "lang_join") == "東 京 で す"


def test_wer_and_cluster_bootstrap_shapes():
    c = metrics.error_counts(["a b c d", "a b", "a b c"], ["a b c x", "a b", "a c"],
                             ["s1", "s1", "s2"])
    assert np.isclose(metrics.wer(c), 2 / 9)
    lo, hi = metrics.bootstrap_ci(c, clustered=True, n_boot=200)
    assert 0 <= lo <= metrics.wer(c) <= hi <= 1


def test_sequential_test_certifies_clear_pass_and_refutes_clear_fail():
    rng = np.random.default_rng(0)
    good = rng.binomial(20, 0.01, 3000) / 20
    bad = rng.binomial(20, 0.12, 3000) / 20
    assert certify.sequential_rate_test(good, 0.05).decision == "certified"
    assert certify.sequential_rate_test(bad, 0.05).decision == "refuted"


def test_sequential_test_rarely_decides_at_the_boundary():
    rng = np.random.default_rng(1)
    wrong = 0
    for _ in range(300):
        rates = rng.binomial(20, 0.05, 1500) / 20
        wrong += certify.sequential_rate_test(rates, 0.05).decision != "undecided"
    assert wrong / 300 <= 0.10   # two one-sided tests at alpha = 0.05 each


def test_capital_never_negative_even_for_extreme_draws():
    rates = np.array([0.0] * 50 + [2.0] * 50 + [0.0] * 50)
    t = certify.sequential_rate_test(rates, 0.05)
    assert (t.certify_capital > 0).all() and (t.refute_capital > 0).all()
