import os
import os.path
import subprocess
import schedule
import time
from datetime import datetime, timedelta, time

import boto3
from tkinter import *
import sqlite3
import warnings
import datetime
import sys
import schedule
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix

import datetime as datetime
import pandas as pd
import numpy as np
from boto3 import s3
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import seaborn as sns

import mlflow
import mlflow.sklearn


def main():
    # Initializing URI
    uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(uri)

    mlflow.tracking.get_tracking_uri()

    # Creating an experiment
    experiment_name = "fraud_detection"
    mlflow.set_experiment(experiment_name)

    alpha = 0.5
    l1_ratio = 0.5

    def train(alpha, l1_ratio):
        # train a model with given parameters
        warnings.filterwarnings("ignore")
        np.random.seed(40)

    # Read the credit card csv file
    d_path = "creditcard.csv"
    X_train, X_test, Y_train, Y_test = initialize_data(d_path)


    with mlflow.start_run():
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, Y_train)

        # Evaluate Metrics
        predicted_qualities = lr.predict(X_test)
        (rmse, mae, r2) = res_metrics(Y_test, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param(key="alpha", value=alpha)
        mlflow.log_param(key="l1_ratio", value=l1_ratio)
        mlflow.log_metric(key="rmse", value=rmse)
        mlflow.log_metrics({"mae": mae, "r2": r2})
        mlflow.log_artifact("pred_Vs_origin.png")
        mlflow.log_artifact(d_path)
        print("Save to: {}".format(mlflow.get_artifact_uri()))

        mlflow.sklearn.log_model(lr, "model")

    root = Tk()
    root.title('Continuous dataset generator')
    root.geometry("400x900")

    # connecting to my existing database
    conn = sqlite3.connect('modeltop.db')
    # create a cursor
    c = conn.cursor()

    # submit command to push to database

    def submit():
        # connect to sqlite3 DB
        conn = sqlite3.connect('modeltop.db')
        # create a cursor
        c = conn.cursor()

        # insert into table created on sqlite3
        c.execute(
            "INSERT INTO fraud_detection values(:time,:V1,:V2,:V3,:V4,:V5,:V6,:V7,:V8,:V9,:V10,:V11,:V12,:V13,:V14,"
            ":V15,:V16,:V17,:V18,:V19,:V20,:V21,:V22,:V23,:V24,:V25,:V26,:V27,:V28,:amount, :class1) ",
            {
                'time': time.get(),
                'V1': v1.get(),
                'V2': v2.get(),
                'V3': v3.get(),
                'V4': v4.get(),
                'V5': v5.get(),
                'V6': v6.get(),
                'V7': v7.get(),
                'V8': v8.get(),
                'V9': v9.get(),
                'V10': v10.get(),
                'V11': v11.get(),
                'V12': v12.get(),
                'V13': v13.get(),
                'V14': v14.get(),
                'V15': v15.get(),
                'V16': v16.get(),
                'V17': v17.get(),
                'V18': v18.get(),
                'V19': v19.get(),
                'V20': v20.get(),
                'V21': v21.get(),
                'V22': v22.get(),
                'V23': v23.get(),
                'V24': v24.get(),
                'V25': v25.get(),
                'V26': v26.get(),
                'V27': v27.get(),
                'V28': v28.get(),
                'amount': amount.get(),
                'class1': class1.get()
            })

        # commit changes
        conn.commit()
        # close connection
        conn.close()

        # clear text box for now
        time.delete(0, END)
        v1.delete(0, END)
        v2.delete(0, END)
        v3.delete(0, END)
        v4.delete(0, END)
        v5.delete(0, END)
        v6.delete(0, END)
        v7.delete(0, END)
        v8.delete(0, END)
        v9.delete(0, END)
        v10.delete(0, END)
        v11.delete(0, END)
        v12.delete(0, END)
        v13.delete(0, END)
        v14.delete(0, END)
        v15.delete(0, END)
        v16.delete(0, END)
        v17.delete(0, END)
        v18.delete(0, END)
        v19.delete(0, END)
        v20.delete(0, END)
        v21.delete(0, END)
        v22.delete(0, END)
        v23.delete(0, END)
        v24.delete(0, END)
        v25.delete(0, END)
        v26.delete(0, END)
        v27.delete(0, END)
        v28.delete(0, END)
        amount.delete(0, END)
        class1.delete(0, END)

    # Creating input boxes
    time = Entry(root, width=30)
    time.grid(row=0, column=1, padx=20)
    v1 = Entry(root, width=30)
    v1.grid(row=1, column=1, padx=20)
    v2 = Entry(root, width=30)
    v2.grid(row=2, column=1, padx=20)
    v3 = Entry(root, width=30)
    v3.grid(row=3, column=1, padx=20)
    v4 = Entry(root, width=30)
    v4.grid(row=4, column=1, padx=20)
    v5 = Entry(root, width=30)
    v5.grid(row=5, column=1, padx=20)
    v6 = Entry(root, width=30)
    v6.grid(row=6, column=1, padx=20)
    v7 = Entry(root, width=30)
    v7.grid(row=7, column=1, padx=20)
    v8 = Entry(root, width=30)
    v8.grid(row=8, column=1, padx=20)
    v9 = Entry(root, width=30)
    v9.grid(row=9, column=1, padx=20)
    v10 = Entry(root, width=30)
    v10.grid(row=10, column=1, padx=20)
    v11 = Entry(root, width=30)
    v11.grid(row=11, column=1, padx=20)
    v12 = Entry(root, width=30)
    v12.grid(row=12, column=1, padx=20)
    v13 = Entry(root, width=30)
    v13.grid(row=13, column=1, padx=20)
    v14 = Entry(root, width=30)
    v14.grid(row=14, column=1, padx=20)
    v15 = Entry(root, width=30)
    v15.grid(row=15, column=1, padx=20)
    v16 = Entry(root, width=30)
    v16.grid(row=16, column=1, padx=20)
    v17 = Entry(root, width=30)
    v17.grid(row=17, column=1, padx=20)
    v18 = Entry(root, width=30)
    v18.grid(row=18, column=1, padx=20)
    v19 = Entry(root, width=30)
    v19.grid(row=19, column=1, padx=20)
    v20 = Entry(root, width=30)
    v20.grid(row=20, column=1, padx=20)
    v21 = Entry(root, width=30)
    v21.grid(row=21, column=1, padx=20)
    v22 = Entry(root, width=30)
    v22.grid(row=22, column=1, padx=20)
    v23 = Entry(root, width=30)
    v23.grid(row=23, column=1, padx=20)
    v24 = Entry(root, width=30)
    v24.grid(row=24, column=1, padx=20)
    v25 = Entry(root, width=30)
    v25.grid(row=25, column=1, padx=20)
    v26 = Entry(root, width=30)
    v26.grid(row=26, column=1, padx=20)
    v27 = Entry(root, width=30)
    v27.grid(row=27, column=1, padx=20)
    v28 = Entry(root, width=30)
    v28.grid(row=28, column=1, padx=20)
    amount = Entry(root, width=30)
    amount.grid(row=29, column=1, padx=20)
    class1 = Entry(root, width=30)
    class1.grid(row=30, column=1, padx=20)

    # creating labels for the above text boxes
    time_label = Label(root, text="Time")
    time_label.grid(row=0, column=0)
    v1_label = Label(root, text="v1")
    v1_label.grid(row=1, column=0)
    v2_label = Label(root, text="v2")
    v2_label.grid(row=2, column=0)
    v3_label = Label(root, text="v3")
    v3_label.grid(row=3, column=0)
    v4_label = Label(root, text="v4")
    v4_label.grid(row=4, column=0)
    v5_label = Label(root, text="v5")
    v5_label.grid(row=5, column=0)
    v6_label = Label(root, text="v6")
    v6_label.grid(row=6, column=0)
    v7_label = Label(root, text="v7")
    v7_label.grid(row=7, column=0)
    v8_label = Label(root, text="v8")
    v8_label.grid(row=8, column=0)
    v9_label = Label(root, text="v9")
    v9_label.grid(row=9, column=0)
    v10_label = Label(root, text="v10")
    v10_label.grid(row=10, column=0)
    v11_label = Label(root, text="v11")
    v11_label.grid(row=11, column=0)
    v12_label = Label(root, text="v12")
    v12_label.grid(row=12, column=0)
    v13_label = Label(root, text="v13")
    v13_label.grid(row=13, column=0)
    v14_label = Label(root, text="v14")
    v14_label.grid(row=14, column=0)
    v15_label = Label(root, text="v15")
    v15_label.grid(row=15, column=0)
    v16_label = Label(root, text="v16")
    v16_label.grid(row=16, column=0)
    v17_label = Label(root, text="v17")
    v17_label.grid(row=17, column=0)
    v18_label = Label(root, text="v18")
    v18_label.grid(row=18, column=0)
    v19_label = Label(root, text="v19")
    v19_label.grid(row=19, column=0)
    v20_label = Label(root, text="v20")
    v20_label.grid(row=20, column=0)
    v21_label = Label(root, text="v21")
    v21_label.grid(row=21, column=0)
    v22_label = Label(root, text="v22")
    v22_label.grid(row=22, column=0)
    v23_label = Label(root, text="v23")
    v23_label.grid(row=23, column=0)
    v24_label = Label(root, text="v24")
    v24_label.grid(row=24, column=0)
    v25_label = Label(root, text="v25")
    v25_label.grid(row=25, column=0)
    v26_label = Label(root, text="v26")
    v26_label.grid(row=26, column=0)
    v27_label = Label(root, text="v27")
    v27_label.grid(row=27, column=0)
    v28_label = Label(root, text="v28")
    v28_label.grid(row=28, column=0)
    amount_label = Label(root, text="Amount")
    amount_label.grid(row=29, column=0)
    class1_label = Label(root, text="class")
    class1_label.grid(row=30, column=0)

    # create submit box
    submit_btn = Button(root, text="Add Record to Database", command=submit)
    submit_btn.grid(row=32, column=0, columnspan=6, pady=10, padx=10, ipadx=100)

    # commit changes
    conn.commit()
    # close connection
    conn.close()
    root.mainloop()

    # Reopen connection to your database now to query the input data into a dataframe so your model can be retrained
    conn = sqlite3.connect('modeltop.db')
    sql_df = pd.read_sql_query("SELECT * FROM fraud_detection", conn)
    print(sql_df)
    sql_df.to_csv('creditssql.csv')

    data1 = pd.read_csv("creditcard.csv")
    append_data = [sql_df, data1]
    append_data_f = pd.concat(append_data)

    append_data_f.to_csv("creditcard.csv")

    s3 = boto3.client("s3")
    s3.upload_file(
        Filename="creditcard.csv",
        Bucket="modeltop",
        Key="updated_data/creditcard.csv",
    )

    # Scheduler

    def get_latest_dataset_path():
        s3_client = boto3.client('s3')
        bucket = 'modeltop'
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Delimiter='/',
            Prefix='updated_data/')
        print(response)
        prefix = response['Contents'][-1]['Key'].split("/")[-1]
        return f's3://{bucket}/{prefix}'

    path = get_latest_dataset_path()
    os.system("python main.py" + " " + path)


def res_metrics(original, pred):
    # compute relevant metrics
    rmse = np.sqrt(mean_squared_error(original, pred))
    mae = mean_absolute_error(original, pred)
    r_sqr = r2_score(original, pred)
    return rmse, mae, r_sqr


def initialize_data(d_path):
    data = pd.read_csv(d_path)

    # declare X and Y as Predictor and target variables

    X = data[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
              'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
              'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
              'Amount']].values

    Y = data['Class'].values

    # This step splits the data into 4 sets. Two for training and other 2 for testing. (80%, 20%) split.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
    return X_train, X_test, Y_train, Y_test


schedule.every(30).days.do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
