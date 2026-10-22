from pathlib import Path

import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_features

DATA = Path("data/clean/pitches.parquet")

# Columns that describe the current pitch itself. None may be a feature.
FORBIDDEN = {"pitch_type", "target", "description", "type"}


@pytest.fixture(scope="module")
def sample():
    if not DATA.exists():
        pytest.skip("clean data not available")
    df = pd.read_parquet(DATA)
    parts = []
    for p in df["pitcher"].unique()[:2]:
        d = df[df["pitcher"] == p]
        games = d["game_pk"].unique()[:6]
        parts.append(d[d["game_pk"].isin(games)])
    return pd.concat(parts).reset_index(drop=True)


def test_no_current_pitch_columns(sample):
    feats = build_features(sample)
    assert not FORBIDDEN & set(feats.columns)
    assert not FORBIDDEN & set(FEATURE_COLUMNS)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(feats) == len(sample)


def test_current_pitch_outcome_does_not_change_its_own_features(sample):
    base = build_features(sample)
    for i in [10, 50, 120, len(sample) // 2]:
        changed = sample.copy()
        original = changed.loc[i, "target"]
        new_label = "KC" if original != "KC" else "FF"
        changed.loc[i, ["pitch_type", "target"]] = new_label
        changed.loc[i, "description"] = "hit_into_play"
        changed.loc[i, "type"] = "X"
        out = build_features(changed)
        pd.testing.assert_frame_equal(base.loc[[i]], out.loc[[i]])


@pytest.mark.parametrize("cut", [40, 200, 500])
def test_features_do_not_use_future_rows(sample, cut):
    cut = min(cut, len(sample) - 1)
    full = build_features(sample)
    part = build_features(sample.iloc[:cut].copy())
    pd.testing.assert_frame_equal(
        full.iloc[:cut].reset_index(drop=True), part.reset_index(drop=True)
    )


def test_previous_pitch_resets_each_at_bat(sample):
    feats = build_features(sample)
    same_ab = (
        (sample["pitcher"] == sample["pitcher"].shift())
        & (sample["game_pk"] == sample["game_pk"].shift())
        & (sample["at_bat_number"] == sample["at_bat_number"].shift())
    )
    first_in_ab = ~same_ab
    for c in ["prev_pitch_1", "prev_pitch_2", "prev_pitch_3"]:
        assert (feats.loc[first_in_ab, c] == "NONE").all()
    assert (
        feats.loc[same_ab, "prev_pitch_1"] == sample["target"].shift()[same_ab]
    ).all()