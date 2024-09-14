import xgboost
from typing import List, Any
import build_features

xgb = xgboost.Booster()
xgb.load_model("src/models/trained_xgb.json")

def predict_model(features: List[Any]) -> List[int]:
    df = build_features.build(features)
    print(df)
    prediction = xgb.predict(df)
    return prediction