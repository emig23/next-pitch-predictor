import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from src.dataset.encoding import ID_COL, FeatureEncoder
from src.modeling.models import XGBPitchModel

def save_model(model: XGBPitchModel, directory, extra: dict | None = None) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "use_pitcher_id": model.use_pitcher_id,
        "floor": model.floor,
        "seed": model.seed,
        "params": model.params,
        "present": model.present_,
        "best_n": model.best_n_,
        "const": None if model.const_ is None else [float(x) for x in model.const_],
        "pitchers": model.enc_.pitchers_,
        "columns": model.enc_.columns_,
        "xgboost_version": xgb.__version__,
        "extra": extra or {},
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    if model.const_ is None:
        model.model_.save_model(str(d / "model.json"))

def load_model(directory) -> XGBPitchModel:
    d = Path(directory)
    meta = json.loads((d / "meta.json").read_text())
    m = XGBPitchModel(meta["params"], use_pitcher_id=meta["use_pitcher_id"],
                      floor=meta["floor"], seed=meta["seed"])
    # Rebuild the encoder through its own fit(), which only needs the pitcher column.
    m.enc_ = FeatureEncoder(meta["use_pitcher_id"]).fit(
        pd.DataFrame({ID_COL: meta["pitchers"]}))
    if m.enc_.columns_ != meta["columns"]:
        raise ValueError("load_model: the feature layout in this code differs from the one "
                         "the model was trained with; retrain and save again")
    m.present_ = list(meta["present"])
    m.best_n_ = int(meta["best_n"])
    m.const_ = None if meta["const"] is None else np.array(meta["const"])
    if m.const_ is None:
        booster = XGBClassifier()
        booster.load_model(str(d / "model.json"))
        m.model_ = booster
    return m