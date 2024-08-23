import xgboost as xgb
import numpy as np
import optuna
from sklearn.metrics import accuracy_score



class FeatureSelectionOptuna:
    """
    This class implements feature selection using Optuna optimization framework.

    Parameters:

    - model (object): The predictive model to evaluate; this should be any object that implements fit() and predict() methods.
    - score_fn (function): The function to use for evaluating the model performance. 
    - features (list of str): A list containing the names of all possible features that can be selected for the model.
    - X (DataFrame): The complete set of feature data (pandas DataFrame) from which subsets will be selected for training the model.
    - y (Series): The target variable associated with the X data (pandas Series).
    - splits (list of tuples): A list of tuples where each tuple contains two elements, the train indices and the validation indices.
    - penalty (float, optional): A factor used to penalize the objective function based on the number of features used.
    """

    def __init__(self,
                 model,
                 score_fn,
                 features,
                 X,
                 y,
                 splits,
                 penalty=0):

        self.model = model
        self.score_fn = accuracy_score
        self.features = features
        self.X = X
        self.y = y
        self.splits = splits
        self.penalty = penalty

    def __call__(self,
                 trial: optuna.trial.Trial):

        # Select True / False for each feature
        selected_features = [trial.suggest_categorical(name, [True, False]) for name in self.features]

        # List with names of selected features
        selected_feature_names = [name for name, selected in zip(self.features, selected_features) if selected]

         # Optional: adds a penalty for the amount of features used
        n_used = len(selected_feature_names)
        total_penalty = n_used * self.penalty
    
        score = 0

        for split in self.splits:
          train_idx = split[0]
          valid_idx = split[1]

          X_train = self.X.iloc[train_idx].copy()
          y_train = self.y.iloc[train_idx].copy()
          X_valid = self.X.iloc[valid_idx].copy()
          y_valid = self.y.iloc[valid_idx].copy()

          X_train_selected = X_train[selected_feature_names].copy()
          X_valid_selected = X_valid[selected_feature_names].copy()

          # Train model, get predictions and accumulate f1_score
          self.model.fit(X_train_selected, y_train)
          pred = self.model.predict(X_valid_selected)

          score += self.score_fn(y_valid, pred)

        # Take the average loss across all splits
        score /= len(self.splits)

         # Add the penalty to the loss
        score -= total_penalty

        return score

