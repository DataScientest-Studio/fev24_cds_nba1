# NBA Shot Predictions

==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

## Project Organization

    ├── LICENSE
    ├── README.md                       <- The top-level README for developers using this project.
    ├── data                            <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── final                       <- The final, canonical data sets for modeling and prediction.
    │   ├── processed                   <- Intermediate datasets to merge into final datasets
    │   └── raw                         <- The original, immutable data dump.
    │
    ├── models                          <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                       <- ipy Notebooks
    │   ├── 1.0-sarah-data-exploration.ipynb                                     <- Exploration on the play by play data
    │   ├── 2.0-sarah-data-preprocressing.ipynb                                  <- Preprocessing on the play by play data
    │   ├── Data_Exploration_NBA_Shot_Locations_Fatiha_Updated.ipynb             <- Exploration on the shot location data
    │   ├── Shot_Locations_Preprocessing_FeatureEngineering_Fatiha_Updated.ipynb <- Preprocessing on the shot locations data
    │   ├── 3.0-merge-datasets.ipynb                                             <- Merge all datasets (plays, shot locations, players stats, teams stats) into one "final" csv file.
    │   ├── 4.0-get-missing-data                                                 <- Getting data from the nba stats API to fill NA.
    │   ├── 5.0-feature-selection.ipynb                                          <- Reduce dimensions using Optuna
    │   ├── 6.0-sarah-modelisation-simple                                        <- Test and compare simple ML models
    │   ├── 7.0-sarah-modelisation-per-palyer.ipynb                              <- Tests with one model per player vs one model for all players
    │   ├── 8.0-sarah-modelisation-avancee.ipynb                                 <- Optimise xgboost model with Optuna
    │   ├── rapport_stats_joueurs.py                                             <- Functions to extract players stats per year
    │   ├── Modeling_epochs100_DeepLearning_LeNet_Original_datasetV5_Fatiha.ipynb           <- Deep learning modeling with LeNet Architecture
    │   ├── Modeling_epochs100_DeepLearning_LeNet_and_Undersampling_datasetV5_Fatiha.ipynb  <- Deep learning modeling with LeNet + Undersampling
    │   ├── Modeling_epochs100_DeepLearning_LeNet_and_weights_datasetV5_Fatiha.ipynb        <- Deep learning modeling with LeNet + weights
    │   ├── Modeling_epochs100_DeepLearning_reducedVariables19_datasetV5_Fatiha.ipynb       <- Deep learning modeling with LeNet on a reduced variables dataset
    │   └── Modeling_epochs100_DeepLearning_LeNet_xgboost_datasetV5_Fatiha.ipynb            <- Deep learning modeling with LeNet combined with XGBoost
    │
    ├── references                      <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports                         <- The reports that you'll make during this project as PDF
    │   └── figures                     <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.
    │                                       generated with `pip freeze > requirements.txt`
    │
    ├── src                             <- Source code for use in this project.
    │   ├── __init__.py                 <- Makes src a Python module
    │   ├── data
    │   │   └── make_dataset.py         <- Script to create the final dataset
    │   │
    │   ├── features                    <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py
    │   │   └── FeatureSelectionOptuna  <- Class used for feature selection with Optuna
    │   │
    │   ├── models
    │   │   ├── predict_model.py        <- Script to use the trained model to make prediction
    │   │   └── train_model.py          <- Script to train model
    │   ├── models
    │   │   └── streamlit_nba.py        <- Streamlit script
    │   │
    │   └── visualization               <- Scripts to create exploratory and results oriented visualizations
    │        └── visualize.py



## Instructions for launching the Streamlit app

``` bash
git clone https://github.com/DataScientest-Studio/fev24_cds_nba1.git
cd fev24_cds_nba1
cd data
wget https://datascientest-nba.s3.eu-north-1.amazonaws.com/data.tar
tar -xvf data.tar
rm data.tar
cd ..
python3 -m venv .venv
pip install -r requirements.txt
pip install -e .
streamlit run src/streamlit/streamlit_nba.py
```
