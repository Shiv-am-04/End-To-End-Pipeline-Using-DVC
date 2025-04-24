import pandas as pd
import logging
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import yaml

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

def load_params(params_path):
    try:
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
        logger.debug("params loaded successfully")

        return params
    except yaml.YAMLError as e:
        logger.error("YAML error %s",e)
        raise
    except FileNotFoundError:
        logger.error("file not found")
        raise
    except Exception as e:
        logger.error("unexpected error : {e}")


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.debug("data loaded successfully")
        return data
    except Exception as e:
        logger.error("error occured while loading the data",e)

def train_gird_model(X_train,y_train):
    model = XGBClassifier()

    params = {
    'n_estimators':[100,50,200],
    'max_depth':[3,4,5,6,8],
    'max_leaves':[2,3,4],
    'grow_policy':['depthwise','lossguide']
    }

    logger.debug('Initializing gridsearch model with parameters: %s', params)
    grid_model = GridSearchCV(model,param_grid=params,scoring='accuracy',cv=5)

    grid_model.fit(X_train,y_train)
    logger.debug('model training completed')

    logger.debug(f"best param : {grid_model.best_params_}")

    return grid_model

def train_xgboost(X_train,y_train,params):
    logger.debug('Initializing xgboost model with parameters: %s', params)
    model = XGBClassifier(n_estimators=params['n_estimators'],
                          max_depth=params['max_depth'],
                          max_leaves=params['max_leaves'],
                          grow_policy=params['grow_policy'])
    
    model.fit(X_train,y_train)
    logger.debug("xgboost model training completed")

    return model

def save_model(model,name):
    directory = 'models'
    os.makedirs(directory,exist_ok=True)

    with open(f'{directory}/{name}.pkl','wb') as f:
        pickle.dump(model,f)

    logger.debug(f"{name} saved to %s",directory)

def main():
    X_train = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\final\X_train.csv')
    y_train = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\preprocessed\y_train.csv')

    params = load_params(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\params.yaml')

    grid_model = train_gird_model(X_train,y_train)

    xgboost_model = train_xgboost(X_train,y_train,params['model_training'])

    save_model(model=grid_model,name='grid_model')
    save_model(model=xgboost_model,name='xgboost_model')

    logger.debug('model training phase completed')

if __name__ == "__main__":
    main()
