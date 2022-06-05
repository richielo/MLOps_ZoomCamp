import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import datetime, timedelta
from datetime import date as date_func

from prefect import flow, task, get_run_logger

import mlflow

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    logger = get_run_logger()
    
    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    with mlflow.start_run():
        logger = get_run_logger()
        train_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts) 
        y_train = df.duration.values

        logger.info(f"The shape of X_train is {X_train.shape}")
        logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, squared=False)
        logger.info(f"The MSE of training is: {mse}")
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")       
        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")
        return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    logger = get_run_logger()
    if(date == None):
        date = date_func.today()
    if(isinstance(date, str)):
        date = datetime.strptime(date, '%Y-%m-%d')
    first = date.replace(day = 1)
    last_month = first - timedelta(days = 1)
    last_first = last_month.replace(day = 1)
    last_last_month = last_first - timedelta(days = 1) 
    train_path = './data/fhv_tripdata_{}.parquet'.format(last_last_month.strftime('%Y-%m'))
    val_path = './data/fhv_tripdata_{}.parquet'.format(last_month.strftime('%Y-%m')) 
    logger.info(f"Train path is: {train_path}")
    logger.info(f"Val path is: {val_path}")
    return train_path, val_path
    

@flow
def main(date="2021-08-15"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    train_path, val_path = get_paths(date).result()
    
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

# main()

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow_location='homework.py',
    name="cron_schedule_model_training",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
)
