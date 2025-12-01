import joblib
import json
import numpy as np
import pandas as pd

def load_model(path="model.pkl"):
    pipeline = joblib.load(path)
    metadata = json.load(open("metadata.json"))
    return pipeline, metadata

def predict_single(pipeline, df):
    return float(pipeline.predict(df)[0])

def score_with_risk(pipeline, metadata, input_df, claimed_price=None):
    pred = float(pipeline.predict(input_df)[0])

    if claimed_price is None:
        return {
            "predicted_price": pred,
            "risk_level": "No-Claim",
            "difference": 0
        }

    diff = claimed_price - pred
    pct = abs(diff) / pred * 100

    if pct > 40:
        level = "High"
    elif pct > 15:
        level = "Medium"
    else:
        level = "Low"

    return {
        "predicted_price": pred,
        "claimed_price": claimed_price,
        "difference": diff,
        "percentage_gap": pct,
        "risk_level": level
    }
