from typing import List, Any
import joblib
import numpy as np
from xgboost import DMatrix

def build_features(features: List[Any]) -> DMatrix:
    #--Shot Distance,Shot Zone Basic_In The Paint (Non-RA),Shot Zone Basic_Mid-Range,Shot Zone Basic_Right Corner 3,
    # Shot Zone Area_Left Side(L),Shot Zone Range_16-24 ft.,Shot Zone Range_24+ ft.,Shot Zone Range_Less Than 8 ft.,
    # PREVIOUS_OFF_REBOUND,PREVIOUS_OFF_MISSED,TS%,USG%,PTS,weight,E_DEF_RATING,PCT_AREA,DETAILLED_SHOT_TYPE_OTHER,PLAYER1_NAME

    # rescaling the data with the training scaler
    scaler = joblib.load("models/scaler.joblib",'r+')
    data = np.array(features).reshape(1,-1)
    scaled_data = scaler.transform(data)
    scaled_data = DMatrix(data=scaled_data)
    return scaled_data