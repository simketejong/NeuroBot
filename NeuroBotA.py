import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
import os
import pathlib

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
#    TABLE_NAME = "ZEC_EUR"
    conn = create_connection(database)
    teken = 1
    dagen = 1
    with conn:
        print("Database connected:")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        Table_Array=cursor.fetchall()
        for aantalTables in range(len(Table_Array)):
            table = str(Table_Array[0]).replace('\'','')
            table = table.replace(',','')
            table = table.replace('(','')
            table = table.replace(')','')
            print(table)
            df = fetch_table_data_into_df(table, conn)
            for col in df.columns:
                if col == "sell":
                    df.drop(["sell","buy","gemiddeld","x3","x2","x1","x0"], axis=1)
#        for teller in df.index:
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

                if teller == teken*256:
                    plt.plot(y)
                    X1=[teller,tot]
                    Y1=[gemiddeld[teller],gemiddeld[teller]]
                    Max=[gemiddeld[teller]+MaxArray,gemiddeld[teller]+MaxArray]
                    Min=[gemiddeld[teller]+MinArray,gemiddeld[teller]+MinArray]
                    plt.plot(X1, Y1, label = "average")
                    plt.plot(X2, Y2, label = "polyfit")
                    plt.plot(X1, Max, label = "Max")
                    plt.plot(X1, Min, label = "Min")
                    plt.legend(loc='best')
                    plt.savefig('foo.pdf')
                    teken=teken+1
                    print("Sell = " +str(sell[teller])+ " Buy = "+ str(buy[teller]))

            #s1=pd.Series(sell, name="sell")
            #df2=pd.concat([df,s1], axis =1)
            #s2=pd.Series(buy, name="buy")
            #df2=pd.concat([df2,s2], axis =1)
            #s3=pd.Series(gemiddeld, name="gemiddeld")
            #df2=pd.concat([df2,s3], axis =1)
            #s4=pd.Series(x3, name="x3")
            #df2=pd.concat([df2,s4], axis =1)
            #s5=pd.Series(x2, name="x2")
            #df2=pd.concat([df2,s5], axis =1)
            #s6=pd.Series(x1, name="x1")
            #df2=pd.concat([df2,s6], axis =1)
            #s7=pd.Series(x0, name="x0")
            #df2=pd.concat([df2,s7], axis =1)
            #    for col in df2.columns:
            #        print(col)
            #df2 = df2.drop(columns=['index', 'date', 'open'])
            #for col in df2.columns:
            #    print(col)
            #print(str(df2))
            df4= df.drop(columns=['index', 'date', 'open'])
            df3=pd.Series(sell, name="sell")
            s2=pd.Series(buy, name="buy")
            df3=pd.concat([df3,s2], axis =1)
            #s3=pd.Series(gemiddeld, name="gemiddeld")
            #df3=pd.concat([df3,s3], axis =1)
            #s4=pd.Series(x3, name="x3")
            #df3=pd.concat([df3,s4], axis =1)
            #s5=pd.Series(x2, name="x2")
            #df3=pd.concat([df3,s5], axis =1)
            #s6=pd.Series(x1, name="x1")
            #df3=pd.concat([df3,s6], axis =1)
            #s7=pd.Series(x0, name="x0")
            #df3=pd.concat([df3,s7], axis =1)

            WEIGHTS = table+"Weight"
            files0 = pathlib.Path(WEIGHTS+".0")
            if files0.exists ():
                file = open(files0, "rb")
                syn0 = np.load(file)
                print("file exists")
            else:
                print("file doesnt exists")
                syn0 = 2*np.random.random((88,800)) - 1
            files1 = pathlib.Path(WEIGHTS+".1")
            if files1.exists ():
                file = open(files1, "rb")
                syn1 = np.load(file)
                print("file exists")
            else:
                print("file doesnt exists")
                syn1 = 2*np.random.random((800,2)) - 1
            df4 = df4.replace(np.nan, 0)
            df3 = df3.replace(np.nan, 0)
            for teller in range(len(df4)):
                tot = teller + 1
                X = df4[teller:tot].to_numpy()
                print(X.dtype)
#                input("Press Enter to continue...")
                y = df3[teller:tot].to_numpy()
                Y = y.T
                for j in range(3000):
                    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
#                    print(str(l1))
#                    input("Press Enter to continue...")
                    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
                    l2_delta = (y - l2)*(l2*(1-l2))
                    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
                    syn1 += l1.T.dot(l2_delta)
                    syn0 += X.T.dot(l1_delta)
                print ("Output After Training:")
                print("Teller = "+ str(teller)+" error = "+ str(np.mean(np.abs(y - l2))))
            file0 = open(files0, "wb")
            np.save(file0,syn0)
            file0.close
            file1 = open(files1, "wb")
            np.save(file1,syn1)
            file1.close

if __name__ == '__main__':
    os.system("cp Neural.sqlite.org Neural.sqlite")
    main()
#    conn.close()
