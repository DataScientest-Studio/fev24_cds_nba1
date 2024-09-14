from typing import List, Any
import pandas as pd
from sklearn.externals import joblib

def build(features: List[Any]) -> pd.DataFrame:
    #--Shot Distance,Shot Zone Basic_In The Paint (Non-RA),Shot Zone Basic_Mid-Range,Shot Zone Basic_Right Corner 3,Shot Zone Area_Left Side(L),Shot Zone Range_16-24 ft.,Shot Zone Range_24+ ft.,Shot Zone Range_Less Than 8 ft.,PREVIOUS_OFF_REBOUND,PREVIOUS_OFF_MISSED,TS%,USG%,PTS,weight,E_DEF_RATING,PCT_AREA,DETAILLED_SHOT_TYPE_OTHER,PLAYER1_NAME
    data = []

    # rescaling the data with the training scaler
    scaler = joblib.load("src/models/scaler.pkl",'rb')
    data = scaler.transform(data)

    return pd.DataFrame(data)