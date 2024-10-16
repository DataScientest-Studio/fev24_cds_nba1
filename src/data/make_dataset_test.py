import pandas as pd

def main():
    data = pd.read_csv("data/final/data_with_all_columns.csv")

    # Save only optuna columns for training
    optuna_columns = ['Shot Distance',
                    'Season Type',
                    'Shot Zone Basic_In The Paint (Non-RA)',
                    'Shot Zone Basic_Right Corner 3',
                    'Shot Zone Area_Right Side(R)',
                    'Shot Zone Range_8-16 ft.',
                    'at_home',
                    'PREVIOUS_OFF_MISSED',
                    'years_exp',
                    'ASTM',
                    'ORBM',
                    'FT%',
                    'height',
                    'weight',
                    'C',
                    'SG-PG',
                    'E_DEF_RATING',
                    'PCT_AREA',
                    'DETAILLED_SHOT_TYPE_JUMP SHOT']

    data = data[optuna_columns + ['target']]

    # Save file
    data.to_csv("data/final/data.csv", index=False)