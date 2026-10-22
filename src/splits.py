import pandas as pd

DEV_TRAIN = [2023, 2024]
DEV_EVAL = [2025]
FINAL_TRAIN = [2023, 2024, 2025]
FINAL_EVAL = [2026]


def split(df: pd.DataFrame, stage: str, confirm_final: bool = False):
    """Return (train, eval) DataFrames for the given stage."""
    if "season" not in df.columns:
        raise ValueError("split: DataFrame needs a 'season' column")

    if stage == "dev":
        train_seasons, eval_seasons = DEV_TRAIN, DEV_EVAL
        # 2026 is dropped up front so dev code cannot touch it even by mistake.
        df = df[df["season"] <= max(DEV_EVAL)]
    elif stage == "final":
        if not confirm_final:
            raise ValueError(
                "The final stage evaluates on the 2026 test season. "
                "Pass confirm_final=True only when the model and settings are locked."
            )
        train_seasons, eval_seasons = FINAL_TRAIN, FINAL_EVAL
    else:
        raise ValueError(f"split: unknown stage {stage!r} (use 'dev' or 'final')")

    train = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    evaluation = df[df["season"].isin(eval_seasons)].reset_index(drop=True)

    if train.empty or evaluation.empty:
        raise ValueError(f"split({stage}): empty train or eval set")
    assert train["season"].max() < evaluation["season"].min()
    return train, evaluation