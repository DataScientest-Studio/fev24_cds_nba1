from typing import List, Any
from src.features.build_features import build_features
import joblib


# Shot Distance,
# Season Type,
# Shot Zone Basic_In The Paint (Non-RA),
# Shot Zone Basic_Right Corner 3,
# Shot Zone Area_Right Side(R),
# Shot Zone Range_8-16 ft.,
# at_home,
# PREVIOUS_OFF_MISSED,
# YEARS_EXP,
# ASTM,
# ORBM,
# FT%,
# height,
# weight,
# C,
# SG-PG,
# E_DEF_RATING,
# PCT_AREA,
# DETAILLED_SHOT_TYPE_JUMP SHOT,


def predict(features: List[Any]) -> List[int]:
    # Load model
    model = joblib.load("models/trained_xgb.joblib",'r+')
    # Build and scaler features and shape it as a DMatrix
    df = build_features(features)
    # Make the prediction
    prediction = model.predict(df)
    # Return the raw prediction (score)
    return prediction


def main():
    # test
    res = predict([19.0,0.0,False,False,False,False,0.0,False,3.0,4.577464788732394,0.704225352112676,0.814,182.88,165.0,0.0,0.0,95.5,35.08997429305913,True])
    print(res)

if __name__=="__main__":
    main()