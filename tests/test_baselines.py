import numpy as np
import pandas as pd
import pytest
from src.baselines import CountBaseline, CountHandBaseline, PitcherMixBaseline
from src.evaluate import (CLASSES, check_proba, evaluate, log_loss,
                          top_k_accuracy)
from src.splits import split

@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    rows = []
    for season in [2023, 2024, 2025, 2026]:
        for pitcher, name in [(1, "A"), (2, "B")]:
            for _ in range(400):
                rows.append({
                    "season": season, "pitcher": pitcher, "player_name": name,
                    "balls": int(rng.integers(0, 4)), "strikes": int(rng.integers(0, 3)),
                    "stand": str(rng.choice(["L", "R"])),
                    "target": str(rng.choice(["FF", "SI", "SL", "CH"])),
                })
    return pd.DataFrame(rows)

# ---------- splits ----------

def test_dev_split_never_contains_2026(toy):
    train, val = split(toy, "dev")
    assert 2026 not in set(train["season"]) | set(val["season"])
    assert set(train["season"]) == {2023, 2024}
    assert set(val["season"]) == {2025}

def test_final_split_requires_confirmation(toy):
    with pytest.raises(ValueError):
        split(toy, "final")
    train, test = split(toy, "final", confirm_final=True)
    assert set(train["season"]) == {2023, 2024, 2025}
    assert set(test["season"]) == {2026}

def test_train_is_always_before_eval(toy):
    for stage in ["dev", "final"]:
        train, ev = split(toy, stage, confirm_final=True)
        assert train["season"].max() < ev["season"].min()

# ---------- baselines ----------

def test_probabilities_valid_and_never_zero(toy):
    train, val = split(toy, "dev")
    for model in [PitcherMixBaseline().fit(train), CountBaseline(alpha=5).fit(train)]:
        p = model.predict_proba(val)
        check_proba(p, len(val))
        assert (p > 0).all()   # no zero probabilities, so no infinite log loss

def test_baseline_b_matches_hand_calculation():
    # pitcher 1: 6 FF + 2 SI at count 0-0, and 2 SL at count 1-0
    rows = [(1, 0, 0, "FF")] * 6 + [(1, 0, 0, "SI")] * 2 + [(1, 1, 0, "SL")] * 2
    train = pd.DataFrame(rows, columns=["pitcher", "balls", "strikes", "target"])
    train["season"] = 2023
    alpha, a = 4.0, 0.5
    model = CountBaseline(alpha=alpha, a=a).fit(train)

    query = pd.DataFrame({"pitcher": [1], "balls": [0], "strikes": [0]})
    p = model.predict_proba(query)[0]

    # Baseline A: (n + 0.5) / (10 + 0.5 * 10 classes)
    pa_ff, pa_si = (6 + a) / 15, (2 + a) / 15
    pa_other = a / 15
    assert p[CLASSES.index("FF")] == pytest.approx((6 + alpha * pa_ff) / (8 + alpha))
    assert p[CLASSES.index("SI")] == pytest.approx((2 + alpha * pa_si) / (8 + alpha))
    assert p[CLASSES.index("KC")] == pytest.approx((0 + alpha * pa_other) / (8 + alpha))

def test_large_alpha_approaches_baseline_a(toy):
    train, val = split(toy, "dev")
    pa = PitcherMixBaseline().fit(train).predict_proba(val)
    pb = CountBaseline(alpha=1e9).fit(train).predict_proba(val)
    assert np.allclose(pa, pb, atol=1e-5)

def test_unseen_count_falls_back_to_baseline_a(toy):
    train, _ = split(toy, "dev")
    q = pd.DataFrame({"pitcher": [1], "balls": [9], "strikes": [9]})
    pa = PitcherMixBaseline().fit(train).predict_proba(q)
    pb = CountBaseline(alpha=10).fit(train).predict_proba(q)
    assert np.allclose(pa, pb)

def test_predictions_do_not_use_eval_labels(toy):
    train, val = split(toy, "dev")
    model = CountBaseline(alpha=10).fit(train)
    with_labels = model.predict_proba(val)
    without_labels = model.predict_proba(val.drop(columns=["target"]))
    assert np.array_equal(with_labels, without_labels)

# ---------- metrics ----------

def test_log_loss_known_values():
    y = ["FF", "SI", "SL"]
    uniform = np.full((3, len(CLASSES)), 1 / len(CLASSES))
    assert log_loss(y, uniform) == pytest.approx(np.log(len(CLASSES)))
    perfect = np.zeros((3, len(CLASSES)))
    for i, c in enumerate(y):
        perfect[i, CLASSES.index(c)] = 1.0
    assert log_loss(y, perfect) < 1e-9

def test_top_k_accuracy_known_values():
    y = ["FF", "SI"]
    p = np.zeros((2, len(CLASSES)))
    p[0, CLASSES.index("FF")] = 0.6
    p[0, CLASSES.index("SL")] = 0.4        # truth FF is rank 1
    p[1, CLASSES.index("SL")] = 0.7
    p[1, CLASSES.index("SI")] = 0.3        # truth SI is rank 2
    assert top_k_accuracy(y, p, 1) == 0.5
    assert top_k_accuracy(y, p, 2) == 1.0

def test_evaluate_rejects_bad_probabilities():
    y = ["FF", "SI"]
    bad = np.full((2, len(CLASSES)), 0.5)     # rows sum to 5
    with pytest.raises(ValueError):
        evaluate(y, bad)
    with pytest.raises(ValueError):
        evaluate(["XX", "FF"], np.full((2, len(CLASSES)), 0.1))

# ---------- Baseline C (pitcher x batter hand x count) ----------

def _hand_data(seed, n=3000):
    """Pitch depends on the batter's hand: L -> mostly SI, R -> mostly FF."""
    rng = np.random.default_rng(seed)
    stand = rng.choice(["L", "R"], n)
    target = np.where(stand == "L",
                      rng.choice(["SI", "FF"], n, p=[.8, .2]),
                      rng.choice(["SI", "FF"], n, p=[.2, .8]))
    return pd.DataFrame({"pitcher": 1, "player_name": "A", "season": 2024,
                         "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
                         "stand": stand, "target": target})

def test_c_probabilities_valid_and_never_zero(toy):
    train, val = split(toy, "dev")
    p = CountHandBaseline().fit(train).predict_proba(val)
    check_proba(p, len(val))
    assert (p > 0).all()

def test_c_matches_hand_calculation():
    # one pitcher. vs L: 6 FF + 2 SI. vs R: 2 FF + 6 SI. All at count 0-0.
    rows = ([("L", "FF")] * 6 + [("L", "SI")] * 2 + [("R", "FF")] * 2 + [("R", "SI")] * 6)
    train = pd.DataFrame(rows, columns=["stand", "target"])
    train["pitcher"], train["balls"], train["strikes"], train["season"] = 1, 0, 0, 2023
    a, beta = 0.5, 4.0
    # With alpha_b and alpha huge, C collapses to the pure hand-adjusted prior:
    # A -> hand mix = (hand count + beta * A) / (hand total + beta)
    model = CountHandBaseline(alpha=1e9, alpha_b=1e12, beta=beta, a=a).fit(train)
    q = pd.DataFrame({"pitcher": [1], "stand": ["L"], "balls": [0], "strikes": [0]})
    p = model.predict_proba(q)[0]

    pa_ff = pa_si = (8 + a) / (16 + a * len(CLASSES))
    pa_other = a / (16 + a * len(CLASSES))
    assert p[CLASSES.index("FF")] == pytest.approx((6 + beta * pa_ff) / (8 + beta), abs=1e-6)
    assert p[CLASSES.index("SI")] == pytest.approx((2 + beta * pa_si) / (8 + beta), abs=1e-6)
    assert p[CLASSES.index("KC")] == pytest.approx((0 + beta * pa_other) / (8 + beta), abs=1e-6)

def test_c_with_no_hand_effect_equals_b(toy):
    train, val = split(toy, "dev")
    b = CountBaseline(alpha=50).fit(train).predict_proba(val)
    c = CountHandBaseline(alpha=1e12, alpha_b=50, beta=1e12).fit(train).predict_proba(val)
    assert np.allclose(b, c, atol=1e-6)

def test_c_beats_b_when_hand_matters():
    train, val = _hand_data(1), _hand_data(2)
    b = CountBaseline(alpha=10).fit(train).predict_proba(val)
    c = CountHandBaseline(alpha=10, alpha_b=10, beta=5).fit(train).predict_proba(val)
    # Knowing the hand can gain at most ln(2) - H(0.8, 0.2) = 0.193 here, so ask for most of it.
    assert log_loss(val["target"], c) < log_loss(val["target"], b) - 0.15

def test_c_unseen_cell_falls_back_to_the_prior(toy):
    train, _ = split(toy, "dev")
    q = pd.DataFrame({"pitcher": [1], "stand": ["L"], "balls": [9], "strikes": [9]})
    model = CountHandBaseline(alpha=10).fit(train)
    far = CountHandBaseline(alpha=1e9).fit(train)
    assert np.allclose(model.predict_proba(q), far.predict_proba(q), atol=1e-6)

def test_c_predictions_do_not_use_eval_labels(toy):
    train, val = split(toy, "dev")
    model = CountHandBaseline().fit(train)
    assert np.array_equal(model.predict_proba(val),
                          model.predict_proba(val.drop(columns=["target"])))

def test_c_rejects_bad_stand_values(toy):
    train, val = split(toy, "dev")
    bad = val.copy()
    bad.loc[bad.index[0], "stand"] = "S"
    with pytest.raises(ValueError):
        CountHandBaseline().fit(train).predict_proba(bad)