import pandas as pd
import numpy as np
import logging
import os


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


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.debug("data loaded successfully")
    except Exception as e:
        logger.error("error occured while loading the data",e)
    
    return data


def rename_column(data,original_name,new_name):
    data.rename(columns={original_name:new_name})

    return data

