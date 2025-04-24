import pandas as pd
import logging
import os
import pickle
import json
from sklearn.metrics import confusion_matrix,precision_score,recall_score,roc_auc_score,accuracy_score
# from dvclive.live import Live

log_directory = 'logs'
os.makedirs(log_directory,exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_directory,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.debug("data loaded successfully")
        return data
    except Exception as e:
        logger.error("error occured while loading the data",e)


def load_model(model_path):
    try:
        with open(model_path,'rb') as f:
            model = pickle.load(f)
        logger.debug("model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"provide model_path as raw string")

def evaluate(X_test,y_test,model):
    y_pred = model.predict(X_test)
    y_pred_probablity = model.predict_proba(X_test)[:,1]

    cf = confusion_matrix(y_test,y_pred)
    print(cf)
    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    roc = roc_auc_score(y_test,y_pred_probablity)

    metrics = {
        'accuracy':accuracy,
        'recall':recall,
        'precision':precision,
        'roc':roc
    }

    logger.debug('Evaluation done and metrics calculated')

    return metrics,cf

def save_metrics(metrics:dict):
    directory = 'results'
    os.makedirs(directory,exist_ok=True)

    try:
        with open(f'{directory}/metrics.json','w') as f:
            json.dump(metrics,f,indent=4)
        logger.debug('metrics saved successfully')
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    X_test = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\final\X_test.csv')
    y_test = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\preprocessed\y_test.csv')


    xgboost = load_model(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\models\xgboost_model.pkl')

    metrics,cf = evaluate(X_test,y_test,xgboost)

    save_metrics(metrics)

if __name__ == "__main__":
    main()