import pandas as pd
import logging
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle

log_directory = 'logs'
os.makedirs(log_directory,exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_directory,'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# model training #

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.debug("data loaded successfully")
        return data
    except Exception as e:
        logger.error("error occured while loading the data",e)

def train_model(X_train,y_train):
    model = XGBClassifier()

    params = {
    'n_estimators':[100,50,200],
    'max_depth':[3,4,5,6,8],
    'max_leaves':[2,3,4],
    'grow_policy':['depthwise','lossguide']
    }

    logger.debug('Initializing RandomForest model with parameters: %s', params)
    grid_model = GridSearchCV(model,param_grid=params,scoring='accuracy',cv=5)

    grid_model.fit(X_train,y_train)
    logger.debug('model training completed')

    logger.debug(f"best param : {grid_model.best_params_}")

    return grid_model

def save_model(model):
    directory = 'models'
    os.makedirs(directory,exist_ok=True)

    with open(f'{directory}/grid_model.pkl','wb') as f:
        pickle.dump(model,f)

    logger.debug("model saved to %s",directory)

def main():
    X_train = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\final\X_train.csv')
    y_train = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\preprocessed\y_train.csv')

    model = train_model(X_train,y_train)

    save_model(model=model)

    logger.debug('model training phase completed')

if __name__ == "__main__":
    main()
