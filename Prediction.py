import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
import pathlib
from keras.models import load_model

def fetch_table_data_into_df(table_name, conn):
    return pd.read_sql_query("select * from " + table_name, conn)

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

def main():
    database = r"/home/ekmis/freqtrade/Neural.sqlite"
#    TABLE_NAME = "ZEC_EUR"
    conn = create_connection(database)
    with conn:
        print("Database connected:")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_Array=cursor.fetchall()
        for aantalTables in range(len(table_Array)):
            table = str(table_Array[0]).replace('\'','')
            table = table.replace(',','')
            table = table.replace('(','')
            table = table.replace(')','')
            #print(table)
            df = fetch_table_data_into_df(table, conn)
            df1 = df.drop(columns=['index', 'date', 'open'])
            df2 = df1.tail(1)
            WEIGHTS = table+"Weight"
            files0 = pathlib.Path(WEIGHTS"_model.h5")
            if files0.exists():
                model = load_model(files0)
                files0 = pathlib.Path(WEIGHTS + ".h5")
                if files0.exists():
                    model.load_weights(files0)
                train = df2.replace(np.nan, 0)
                print(model.predict(train))

if __name__ == '__main__':
    main()

