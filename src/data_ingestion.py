import pandas as pd
import numpy as np
import logging
import os
import yaml
from sklearn.model_selection import train_test_split

# logging setup #

log_directory = 'logs'
os.makedirs(log_directory,exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_directory,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


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
    except Exception as e:
        logger.error("error occured while loading the data",e)
    
    return data


def rename_column(data,original_name:str,new_name:str):
    data.rename(columns={original_name:new_name})
    logger.debug(f'{original_name} renamed to {new_name}')

    return data


def save_data(train_data,test_data,data_path):
    raw_data_path = os.path.join(data_path,'raw')
    os.makedirs(raw_data_path,exist_ok=True)
    train_data.to_csv(os.path.join(raw_data_path,'train_data.csv'))
    test_data.to_csv(os.path.join(raw_data_path,'test_data.csv'))

    logger.debug(f'train and test data saved to the {raw_data_path} folder')


def main():
    data = load_data(r"C:\Users\SHIVAM GHUGE\Downloads\archive (1)\credit_risk_dataset.csv")

    # removing those who aged above 70 as bank won't provide loan to people with age above 60-70
    # removing these is essential before splitting into train and test.
    data = data[data['person_age'] <= 70]
    logger.debug("unnecessary data removed")

    params = load_params(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\params.yaml')

    test_size = params['data_ingestion']['test_size']

    train_data,test_data = train_test_split(data,test_size=test_size,random_state=101)

    save_data(train_data,test_data,'./data')

    logger.debug('data ingestion process complete')


if __name__ == "__main__":
    main()



