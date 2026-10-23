import numpy as np
import pandas as pd
import pytest
from src.calibration import reliability_table
from src.encoding import (PREV_CATS, PREV_COLS, RESULT_CATS, FeatureEncoder)
from src.evaluate import CLASSES, check_proba, log_loss
from src.features import FEATURE_COLUMNS
from src.models import (GLOBAL_PARAMS, PER_PITCHER_PARAMS, PerPitcherXGB,
                        XGBPitchModel)

SHARES = [c for c in FEATURE_COLUMNS if c.startswith("share_")]

def make_data(n, seed, pitchers=(1, 2, 3)):
    """Toy feature table with the real column names. The label depends on the
    strike count, so a model that works should find the pattern."""
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
        "outs": rng.integers(0, 3, n), "inning": rng.integers(1, 10, n),
        "runner_on_1b": rng.integers(0, 2, n), "runner_on_2b": rng.integers(0, 2, n),
        "runner_on_3b": rng.integers(0, 2, n), "score_diff": rng.integers(-5, 6, n),
        "pitcher_is_home": rng.integers(0, 2, n), "batter_left": rng.integers(0, 2, n),
        "pitcher_left": rng.integers(0, 2, n), "pitch_number": rng.integers(1, 8, n),
        "pitches_so_far": rng.integers(0, 100, n), "times_through_order": rng.integers(1, 4, n),
        "pitcher_id": rng.choice(pitchers, n),
    })
    for c in PREV_COLS:
        f[c] = rng.choice(PREV_CATS, n)
    f["prev_result"] = rng.choice(RESULT_CATS, n)
    for c in SHARES:
        f[c] = rng.random(n)
    f.loc[rng.random(n) < 0.05, SHARES] = np.nan      # first pitch of a game
    f = f[FEATURE_COLUMNS]

    two_strikes = f["strikes"].to_numpy() == 2
    y = np.where(two_strikes, rng.choice(["SL", "CH"], n, p=[.8, .2]),
                 rng.choice(["FF", "SI"], n, p=[.7, .3]))
    y = pd.Series(y)
    # pitcher 3 never throws a changeup or sinker
    y[(f["pitcher_id"] == 3).to_numpy() & y.isin(["CH", "SI"]).to_numpy()] = "FF"
    return f, y

@pytest.fixture(scope="module")
def data():
    ftr, ytr = make_data(3000, 1)
    fva, yva = make_data(1000, 2)
    return ftr, ytr, fva, yva

@pytest.fixture(scope="module")
def global_model(data):
    ftr, ytr, fva, yva = data
    return XGBPitchModel(GLOBAL_PARAMS, use_pitcher_id=True).fit(ftr, ytr, eval_set=(fva, yva))

# ---------- encoder ----------

def test_encoder_columns_are_identical_for_any_rows(data):
    ftr, _, fva, _ = data
    enc = FeatureEncoder(use_pitcher_id=True).fit(ftr)
    a, b, c = enc.transform(ftr), enc.transform(fva), enc.transform(fva.iloc[:5])
    assert list(a.columns) == list(b.columns) == list(c.columns) == enc.columns_
    assert not a.isna().any().any() or all(col.startswith("share_") for col in a.columns[a.isna().any()])

def test_encoder_has_pitcher_columns_only_for_global(data):
    ftr, *_ = data
    g = FeatureEncoder(True).fit(ftr).columns_
    p = FeatureEncoder(False).fit(ftr).columns_
    assert any(c.startswith("pitcher=") for c in g)
    assert not any(c.startswith("pitcher=") for c in p)

def test_encoder_never_sees_the_label(data):
    ftr, *_ = data
    cols = FeatureEncoder(True).fit(ftr).columns_
    assert not any(c.startswith(("target", "pitch_type", "description")) for c in cols)

def test_encoder_rejects_unknown_category(data):
    ftr, *_ = data
    bad = ftr.copy()
    bad.loc[bad.index[0], "prev_pitch_1"] = "XX"
    with pytest.raises(ValueError):
        FeatureEncoder(True).fit(ftr).transform(bad)

# ---------- global model ----------

def test_global_probabilities_are_valid(global_model, data):
    _, _, fva, _ = data
    p = global_model.predict_proba(fva)
    check_proba(p, len(fva))
    assert (p > 0).all()

def test_global_model_learns_the_pattern(global_model, data):
    ftr, ytr, fva, yva = data
    freq = ytr.value_counts(normalize=True).reindex(CLASSES).fillna(0).clip(lower=1e-3)
    freq = freq / freq.sum()
    naive = np.tile(freq.to_numpy(), (len(fva), 1))
    assert log_loss(yva, global_model.predict_proba(fva)) < log_loss(yva, naive) - 0.1

def test_floor_is_enforced(data):
    ftr, ytr, fva, yva = data
    m = XGBPitchModel(GLOBAL_PARAMS, floor=0.05).fit(ftr, ytr, eval_set=(fva, yva))
    assert m.predict_proba(fva).min() > 0.03

def test_same_seed_gives_same_predictions(data):
    ftr, ytr, fva, yva = data
    a = XGBPitchModel(GLOBAL_PARAMS, seed=7).fit(ftr, ytr, eval_set=(fva, yva))
    b = XGBPitchModel(GLOBAL_PARAMS, seed=7).fit(ftr, ytr, eval_set=(fva, yva))
    assert np.allclose(a.predict_proba(fva), b.predict_proba(fva))

def test_refit_with_fixed_tree_count_matches_early_stopped_model(global_model, data):
    ftr, ytr, fva, _ = data
    refit = XGBPitchModel(GLOBAL_PARAMS).fit(ftr, ytr, n_estimators=global_model.best_n_)
    assert np.allclose(global_model.predict_proba(fva), refit.predict_proba(fva), atol=1e-5)

def test_fit_requires_eval_set_or_tree_count(data):
    ftr, ytr, *_ = data
    with pytest.raises(ValueError):
        XGBPitchModel(GLOBAL_PARAMS).fit(ftr, ytr)

# ---------- per-pitcher models ----------

def test_per_pitcher_handles_pitch_types_never_seen_in_training(data):
    ftr, ytr, fva, yva = data
    pp = PerPitcherXGB(PER_PITCHER_PARAMS).fit(ftr, ytr, eval_feats=fva, eval_y=yva)
    p = pp.predict_proba(fva)
    check_proba(p, len(fva))
    rows3 = (fva["pitcher_id"] == 3).to_numpy()
    # pitcher 3 never threw a changeup in training, so it only gets the floor
    assert p[rows3, CLASSES.index("CH")].max() < 0.01
    assert p[rows3, CLASSES.index("FF")].min() > 0.01

def test_per_pitcher_rejects_unknown_pitcher(data):
    ftr, ytr, fva, yva = data
    pp = PerPitcherXGB(PER_PITCHER_PARAMS).fit(ftr, ytr, eval_feats=fva, eval_y=yva)
    other, _ = make_data(20, 5, pitchers=(99,))
    with pytest.raises(ValueError):
        pp.predict_proba(other)

def test_per_pitcher_model_ignores_other_pitchers_rows(data):
    ftr, ytr, fva, yva = data
    pp = PerPitcherXGB(PER_PITCHER_PARAMS).fit(ftr, ytr, eval_feats=fva, eval_y=yva)
    m1 = pp.models_[1]
    only1 = ftr[ftr["pitcher_id"] == 1]
    assert np.allclose(pp.predict_proba(only1), m1.predict_proba(only1))

# ---------- calibration ----------

def test_calibration_of_a_calibrated_forecast_is_small():
    rng = np.random.default_rng(0)
    n = 20000
    proba = rng.dirichlet(np.ones(len(CLASSES)), n)
    y = [CLASSES[rng.choice(len(CLASSES), p=row)] for row in proba]
    _, ece = reliability_table(y, proba)
    assert ece < 0.02

def test_calibration_catches_overconfidence():
    rng = np.random.default_rng(1)
    n = 5000
    y = list(rng.choice(CLASSES, n))
    proba = np.full((n, len(CLASSES)), 0.01 / (len(CLASSES) - 1))
    proba[:, 0] = 0.99                     # always 99% sure it's the first class
    _, ece = reliability_table(y, proba)
    assert ece > 0.05