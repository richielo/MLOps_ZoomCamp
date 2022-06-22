#!/usr/bin/env python
# coding: utf-8

import os
import sys

import uuid
import pickle

from datetime import datetime

import pandas as pd

import pickle

import numpy as np

categorical = ['PUlocationID', 'DOlocationID']

def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr


def save_results( df, y_pred, output_file, year, month):
    df_result = pd.DataFrame()
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predicted_duration'] = y_pred
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

def apply_model(input_file, output_file, year, month):
    df = read_dataframe(input_file)

    dv, lr = load_model()
    dicts = df[categorical].to_dict(orient='records')
    dicts = dv.transform(dicts)
    y_pred = lr.predict(dicts)

    save_results(df, y_pred, output_file, year, month)
    
    print("done applying model, mean prediction: {}".format(np.mean(y_pred)))
    
    return output_file


def get_paths(year, month):
    input_file = f'data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{year:04d}-{month:02d}.parquet'

    return input_file, output_file


def ride_duration_prediction(
        year: int,
        month: int,):
    
    input_file, output_file = get_paths(year, month)

    apply_model(
        input_file=input_file,
        output_file=output_file,
        year = year,
        month = month,
    )


def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3


    ride_duration_prediction(
        year = year,
        month = month,
    )


if __name__ == '__main__':
    run()
