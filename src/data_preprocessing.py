import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler

log_directory = 'logs'
os.makedirs(log_directory,exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_directory,'data_preprocessing.log')
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
    
    X,y = data.drop(columns=['loan_status']),data[['loan_status']]
    logger.debug("features and target seperated")

    return X,y


def preprocess(data:pd.DataFrame):
    numeric_columns = [col for col in data.columns if data[col].dtypes != 'object']

    # the median values of person_emp_length column grouping along the loan_grade column values
    # ex : {'A':8.0,'B':2.1,..}
    mean_values_per_loangrade = data[data['person_emp_length'].notna()].groupby('loan_grade')[numeric_columns].median()['person_emp_length'].to_dict()

    for val in data['loan_grade'].unique():
        data.loc[(data['person_emp_length'].isna()) & (data['loan_grade'] == val), 'person_emp_length'] = mean_values_per_loangrade[val]
    logger.debug('person_emp_length missing values filled')

    # interest rate column filled with the median because either data is negatively or positively skewed, we should replace missing values with the median. 
    data.loc[data['loan_int_rate'].isna(),'loan_int_rate'] = data['loan_int_rate'].median()
    logger.debug('loan_int_rate missing values filled')
    print(data.isna().sum())

    return data

def scaling_train_data(data:pd.DataFrame):
    numeric_columns = [col for col in data.columns if data[col].dtypes != 'object']

    scaler = StandardScaler()

    numeric_train_data = scaler.fit_transform(data[numeric_columns])
    logger.debug(f"train_data scaled successfully")

    numeric_train_data = pd.DataFrame(numeric_train_data,columns=numeric_columns)

    return numeric_train_data,scaler,numeric_columns

def scaling_test_data(data:pd.DataFrame,scaler:StandardScaler,numeric_columns):
    numeric_test_data = scaler.transform(data[numeric_columns])
    logger.debug(f"test_data scaled successfully")

    numeric_test_data = pd.DataFrame(numeric_test_data,columns=numeric_columns)

    return numeric_test_data

def encoding_categorical_features(data:pd.DataFrame):
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']

    # to reduce the dimension that is going to create after encoding ,putting lower loan grade into same group 'O' 
    data.loc[~data['loan_grade'].isin(['A','B','C']),'loan_grade'] = 'O'
    logger.debug("grade other than A,B and C are grouped together")

    cat_test_data = pd.get_dummies(data[categorical_columns])
    logger.debug("categorical columns encoded")

    return cat_test_data

def save_data(X_train,y_train,X_test,y_test,file_path):
    directory = os.path.join(file_path,'preprocessed')
    os.makedirs(directory,exist_ok=True)

    X_train.to_csv(os.path.join(directory,'X_train.csv'),index=False)
    y_train.to_csv(os.path.join(directory,'y_train.csv'),index=False)
    
    X_test.to_csv(os.path.join(directory,'X_test.csv'),index=False)
    y_test.to_csv(os.path.join(directory,'y_test.csv'),index=False)

    logger.debug(f'features and target data saved to the {directory} folder')

def main():
    X_train,y_train = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\raw\train_data.csv')
    X_test,y_test = load_data(r'D:\MLOPS\DVC\pipeline\End-To-End-Pipeline-Using-DVC\data\raw\test_data.csv')

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    numeric_X_train,scaler,numeric_features = scaling_train_data(X_train)
    numeric_X_test = scaling_test_data(X_test,scaler,numeric_features)

    cat_X_train = encoding_categorical_features(X_train)
    cat_X_test = encoding_categorical_features(X_test)

    X_train = pd.concat([numeric_X_train,cat_X_train],axis=1)
    X_test = pd.concat([numeric_X_test,cat_X_test],axis=1)

    save_data(X_train,y_train,X_test,y_test,'./data')

    logger.debug("Data preprocessing step completed")


if __name__ == "__main__":
    main()
