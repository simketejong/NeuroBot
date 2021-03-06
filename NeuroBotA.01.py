import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
import os
import time
import pathlib
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model


import matplotlib.pyplot as plt

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
    conn = create_connection(database)
    zien=1
    dagen = 1
    with conn:
        print("Database connected:")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        Table_Array=cursor.fetchall()
        for aantalTables in range(len(Table_Array)):
            table = str(Table_Array[aantalTables]).replace('\'','')
            table = table.replace(',','')
            table = table.replace('(','')
            table = table.replace(')','')
            print(table)
            zien = 1
            df = fetch_table_data_into_df(table, conn)
            for col in df.columns:
                if col == "sell":
                    df.drop(["sell","buy","gemiddeld","x3","x2","x1","x0"], axis=1)
            lengte=len(df.index)
            sell = []
            buy = []
            gemiddeld = []
            x3 = []
            x2 = []
            x1 = []
            x0 = []
            verschil = []
            for init in range(lengte):
                sell.append(0)
                buy.append(0)
                gemiddeld.append(0)
                x3.append(0)
                x2.append(0)
                x1.append(0)
                x0.append(0)
                verschil.append(0)
            for teller in range(lengte-3):
                verschil1 = []
                tot = teller + (288*dagen)
                df1 = df[df['index'].between(teller, tot)]
                gemid = df1.open.mean()
                gemiddeld[teller] = gemid
#                print("teller = " + str(teller) + " tot = "+ str(tot) + " gemiddeld = " + str(gemid))
                x = (df1.index - teller)
                y = df1.open
                CoefPoly3 = np.polyfit(x, y, 3)
                x0[teller] = CoefPoly3[0]
                x1[teller] = CoefPoly3[1]
                x2[teller] = CoefPoly3[2]
                x3[teller] = CoefPoly3[3]
                X2=[]
                Y2=[]
                plt.clf()
                for toets in range(len(df1.index)):
                    polyfit = CoefPoly3[0]*(toets*toets*toets)+CoefPoly3[1]*(toets*toets)+CoefPoly3[2]*toets+CoefPoly3[3]
                    Y2.append(polyfit)
                    X2.append(teller+toets)
                    verschil[teller]=(polyfit - gemid)
                    verschil1.append(polyfit - gemid)
                MaxArray = np.max(verschil1)
                MinArray = np.min(verschil1)
                if MaxArray > 0:
                    if verschil1[0] > 0:
                        temp1=(verschil1[0] / MaxArray )
                        if temp1 > 1:
                            sell[teller] = 1
                        else:
                            sell[teller]=(verschil1[0] / MaxArray )
                    else:
                        sell[teller]=0
                else:
                    sell[teller]=0
                if MinArray < 0:
                    if verschil1[0] < 0:
                        temp = (verschil1[0] / MaxArray)*-1
                        if temp > 1:
                            buy[teller] = 1
                        else:
                            buy[teller]=(verschil1[0] / MaxArray)*-1
                    else:
                        buy[teller]=0
                else:
                    buy[teller]=0

                if teller == zien*256:
                    plt.plot(y)
                    X1=[teller,tot]
                    Y1=[gemiddeld[teller],gemiddeld[teller]]
                    Max=[gemiddeld[teller]+MaxArray,gemiddeld[teller]+MaxArray]
                    Min=[gemiddeld[teller]+MinArray,gemiddeld[teller]+MinArray]
                    plt.title(table)
                    plt.plot(X1, Y1, label = "average")
                    plt.plot(X2, Y2, label = "polyfit")
                    plt.plot(X1, Max, label = "Max")
                    plt.plot(X1, Min, label = "Min")
                    plt.legend(loc='best')
                    plt.savefig('foo.pdf')
                    zien=zien+1
                    print("Sell = " +str(sell[teller])+ " Buy = "+ str(buy[teller]))

            df4= df.drop(columns=['index', 'date', 'open'])
            df3=pd.Series(sell, name="sell")
            s2=pd.Series(buy, name="buy")
            df3=pd.concat([df3,s2], axis =1)
            train = df4.replace(np.nan, 0)
            target = df3.replace(np.nan, 0)
            model = Sequential()
            model.add(Dense(176, input_dim=88, activation='relu'))
            model.add(Dense(352, activation='relu'))
            model.add(Dense(88, activation='relu'))
            model.add(Dense(2, activation='sigmoid'))
            model.compile(loss='mean_squared_error',
                            optimizer='adam',
                            metrics=['binary_accuracy'])
            WEIGHTS = table+"Weight"
            files0 = pathlib.Path("/tmp/"+WEIGHTS+".h5")
            if files0.exists ():
                model.load_weights(files0)
                print("file exists")
            else:
                print("file doesnt exists")
            model.fit(train, target, epochs=100, verbose=2)
            open("/tmp/"+WEIGHTS+"_wait", 'a').close()
            time.sleep(10)
            model.save_weights(files0,overwrite=True)
            model.save("/tmp/"+WEIGHTS+"_model.h5", overwrite=True)
            if os.path.exists("/tmp/"+WEIGHTS+"_wait"):
                os.remove("/tmp/"+WEIGHTS+"_wait")
#               print(model.predict(train).round())

if __name__ == '__main__':
#    os.system("cp Neural.sqlite.org Neural.sqlite")
    main()

