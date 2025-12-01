# model_utils.py
import joblib
import numpy as np
import pandas as pd

def load_model(path="model.pkl"):
    obj = joblib.load(path)
    pipeline = obj["pipeline"]
    metadata = obj.get("metadata", {})
    return pipeline, metadata

def predict_single(pipeline, input_df):
    # input_df: pandas DataFrame dengan satu baris
    preds = pipeline.predict(input_df)
    return preds

def score_with_risk(pipeline, metadata, input_df, claimed_price=None):
    """
    Mengembalikan dict:
    {
      "predicted_price": float,
      "claimed_price": float or None,
      "residual": claimed - predicted (if claimed provided),
      "risk_level": "Low"|"Medium"|"High",
      "risk_score": float (0..1)
    }
    Risk logic:
      - Berdasarkan residu relatif terhadap residual_std (dari training).
      - Tambahan rules: usia motor (jika ada kolom 'year'), kilometer tinggi.
    """
    pred = float(pipeline.predict(input_df)[0])
    out = {"predicted_price": pred, "claimed_price": claimed_price}

    # residual if claimed provided
    if claimed_price is not None:
        res = claimed_price - pred
        out["residual"] = float(res)
    else:
        out["residual"] = None

    # base risk score from residual z-score
    res_std = metadata.get("residual_std", None) or 1.0
    res_mean = metadata.get("residual_mean", 0.0)
    if claimed_price is not None:
        z = abs(( (claimed_price - pred) - res_mean) / (res_std + 1e-9))
        # map z to [0,1] via logistic-ish function
        risk_score = min(1.0, z / 3.0)  # z ~ 3 => risk ~1
    else:
        # if no claimed price, risk depends on features (e.g., very old or extremely low price)
        risk_score = 0.2

    # feature-based adjustments
    # attempt to detect 'year' or 'km' columns
    fscore = 0.0
    if "year" in input_df.columns:
        try:
            year = int(input_df.loc[input_df.index[0], "year"])
            import datetime
            age = datetime.datetime.now().year - year
            if age > 10:
                fscore += 0.2
            elif age > 6:
                fscore += 0.1
        except Exception:
            pass
    if "km" in input_df.columns:
        try:
            km = float(input_df.loc[input_df.index[0], "km"])
            if km > 150000:
                fscore += 0.2
            elif km > 80000:
                fscore += 0.1
        except Exception:
            pass

    final_score = min(1.0, risk_score + fscore)
    if final_score < 0.3:
        level = "Low"
    elif final_score < 0.7:
        level = "Medium"
    else:
        level = "High"

    out.update({"risk_score": float(final_score), "risk_level": level})
    return out
