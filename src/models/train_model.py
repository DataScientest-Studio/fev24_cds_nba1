from sklearn.preprocessing import StandardScaler
import xgboost
import pandas as pd
import numpy as np
from sklearn.externals import joblib

def main():
    X_train = pd.read_csv('data/preprocessed/X_train.csv')

    y_train = pd.read_csv('data/preprocessed/y_train.csv')
    y_test = pd.read_csv('data/preprocessed/y_test.csv')

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'src/models/scaler.save')

    train = xgboost.DMatrix(data=X_train_scaled, label=y_train)

    #--Train the model
    params = {
        "objective":"binary:logistic"
    }
    xgb = xgboost.train(params, train)

    #--Save the model
    xgb.save_model("src/models/trained_xgb.json")

if __name__ == "__main__":
    main()
