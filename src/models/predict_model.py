import xgboost as xgb
from typing import List, Any
import build_features

xgb.load_model("models/trained_xgb.json")

def predict_model(features: List[Any]) -> List[int]:
    df = build_features.build(features)

    # Penser à utiliser le scaler enregistré lors de l'entraînement
    
    print(df)
    prediction = xgb.predict(df)
    return prediction