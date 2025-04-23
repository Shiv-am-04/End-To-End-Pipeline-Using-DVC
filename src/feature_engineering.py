import pandas as pd
import logging
import os

log_directory = 'logs'
os.makedirs(log_directory,exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_directory,'feature_engineering.log')
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
    except OSError as e:
        logger.error("please provide the file path as raw string")

    return data


def feature_handling(data:pd.DataFrame):

    if 'Unnamed: 0' in data.columns:
        data.drop(columns='Unnamed: 0',inplace=True)
        logger.debug("droped the unnecessary column")

    # cb_person_default is binary data and after two columns are created of which any one is useful
    if 'cb_person_default_on_file_N' in data.columns:
        data = data.drop(columns="cb_person_default_on_file_N")
        logger.debug('feature handling completed')
    
    return data

def save_data(X_train,X_test,path):
    directory = os.path.join(path,'final')
    os.makedirs(directory,exist_ok=True)

    X_train.to_csv(os.path.join(directory,'X_train.csv'),index=False)
    
    X_test.to_csv(os.path.join(directory,'X_test.csv'),index=False)

    logger.debug(f'features and target data saved to the {directory} folder')


def main():
    train_features = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\preprocessed\X_train.csv')
    test_features = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\preprocessed\X_test.csv')

    train_features,test_features = feature_handling(train_features),feature_handling(test_features)

    save_data(train_features,test_features,'./data')

    logger.debug("feature_engineering step completed")

if __name__ == "__main__":
    main()