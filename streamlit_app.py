pip install matplotlib
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from math import*
import matplotlib
from matplotlib import pyplot as plt   
from matplotlib import style
import numpy as np
from scipy import stats as st
import random
import statistics as stat
import pandas as pd
import seaborn as sns
import matplotlib
import random
import scipy
from scipy.optimize import curve_fit 
from numpy import array, exp
from statistics import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import curve_fit
import itertools
from sklearn.cluster import KMeans
import seaborn.objects as so
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib import cbook
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from shapely.geometry import Point, Polygon
import math
from prettytable import PrettyTable
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
import torch
import torch.nn as nn

matplotlib.use('TkAgg')

win = tk.Tk()
win.title("ТАГАНСКОЕ ЗАПАД")
win.geometry("1000x1300")
#icon = tk.PhotoImage(file="C:\\Users\\79284\\Pictures\\бентонит1.png")
##win.iconphoto(True, icon)

##scale_widget_5 = tk.Scale(win, orient="horizontal", resolution=1, from_=5267300, to=5267850)
##label_1 = tk.Label(win, text="X")
##label_1.grid(row=30, column=0)
##scale_widget_5.grid(row=31, column=0)
##
##scale_widget_6 = tk.Scale(win, orient="horizontal", resolution=1, from_=14716100, to=14717400)
##label_1 = tk.Label(win, text="Y")
##label_1.grid(row=32, column=0)
##scale_widget_6.grid(row=33, column=0)
##
##
##scale_widget_3 = tk.Scale(win, orient="horizontal", resolution=1, from_=14716100, to=14717400)
##label_1 = tk.Label(win, text="Y min")
##label_1.grid(row=30, column=1)
##scale_widget_3.grid(row=31, column=1)
##
##scale_widget_4 = tk.Scale(win, orient="horizontal", resolution=1, from_=14716100, to=14717400)
##label_1 = tk.Label(win, text="Y max")
##label_1.grid(row=32, column=1)
##scale_widget_4.grid(row=33, column=1)
##
##scale_widget_1 = tk.Scale(win, orient="horizontal", resolution=1, from_=5267300, to=5267850)
##label_1 = tk.Label(win, text="X min")
##label_1.grid(row=30, column=9)
##scale_widget_1.grid(row=31, column=9)
##
##scale_widget_2 = tk.Scale(win, orient="horizontal", resolution=1, from_=5267300, to=5267850)
##label_1 = tk.Label(win, text="X max")
##label_1.grid(row=32, column=9)
##scale_widget_2.grid(row=33, column=9)



res101 = tk.Label(win, text = " ")
res101.grid(row = 5, column = 0)
res102 = tk.Label(win, text = " ")
res102.grid(row = 6, column = 0)
res103 = tk.Label(win, text = " ")
res103.grid(row = 7, column = 0)
res104 = tk.Label(win, text = " ")
res104.grid(row = 8, column = 0)
res105 = tk.Label(win, text = " ")
res105.grid(row = 9, column = 0)
res = tk.Label(win, text = " ")
res.grid(row = 11, column = 0)
#res1 = tk.Label(win, text = "-")
##res1.grid(row = 11, column = 10)
res2 = tk.Label(win, text = "")
res2.grid(row = 12, column = 0)
res3 = tk.Label(win, text = " ", fg='red', font = ("Arial Bold", 9))
res3.grid(row = 13, column = 10)
##res22 = tk.Label(win, text = " ")
##res22.grid(row = 14, column = 0)
lbl = Label(win, text="1 - Глубина залегания, м", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=14)  
num0 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num0.grid(row = 15, column = 0)
lbl = Label(win, text="2 - Влажность глины, %", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=16)  
num1 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num1.grid(row = 17, column = 0)
lbl = Label(win, text="3 - Песок, %", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=18)  
num2 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num2.grid(row = 19, column = 0)
lbl = Label(win, text="4 - Индекс набухания, мл/2г", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=20)  
num3 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num3.grid(row = 21, column = 0)
lbl = Label(win, text="5 - Электропроводность, мкСм/см", font=("Arial Bold", 12), fg='blue')  
lbl.grid(column=0, row=22)  
num4 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num4.grid(row = 23, column = 0)
##lbl = Label(win, text="5 - Электропроводность, мкСм", font=("Arial Bold", 12))  
##lbl.grid(column=0, row=24)  
##num5 = tk.Entry(win, width=20, justify='center', font=("Arial Bold", 12))
##num5.grid(row = 25, column = 0)
##lbl = Label(win, text="6 - Индекс набухания, мл/2г", font=("Arial Bold", 12))  
##lbl.grid(column=0, row=26)  
##num5 = tk.Entry(win, width=20, justify='center', font=("Arial Bold", 12))
##num5.grid(row = 27, column = 0)

##lbl = Label(win, text="ВВЕСТИ УРОВЕНЬ СТАТИСТИЧЕСКОЙ ЗНАЧИМОСТИ (percent)", font=("Arial Bold", 10), fg='orange')
##lbl.grid(column=0, row=4)  
##num4 = tk.Entry(win, width=15, fg='orange', justify='center')
##num4.grid(row = 5, column = 0)
##lbl = Label(win, text="ВВЕСТИ ГИПОТЕЗУ О СРЕДНЕМ КВП ПЕРВОЙ СОВОКУПНОСТИ КЕРНОВ", font=("Arial Bold", 10), fg='blue')
##lbl.grid(column=0, row=6)  
##num5 = tk.Entry(win, width=20, fg='blue', justify='center')
##num5.grid(row = 7, column = 0)
lbl = Label(win, text=" 6 - КОЕ, мг-экв/100г ", font=("Arial Bold", 12), fg='blue')
lbl.grid(column=0, row=24)  
num10 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num10.grid(row = 25, column = 0)
lbl = Label(win, text="                                    7 - Монтмориллонит, %                                    ", font=("Arial Bold", 12), fg='blue')
lbl.grid(column=0, row=26)  
num11 = tk.Entry(win, width=27, fg='blue', justify='center', font=("Arial Bold", 12))
num11.grid(row = 27, column = 0)
lbl = Label(win, text="Целевой параметр", font=("Arial Bold", 12), fg='green')
lbl.grid(column=0, row=28)  
num111 = tk.Entry(win, width=20, fg='green', justify='center', font=("Arial Bold", 12))
num111.grid(row = 29, column = 0)
lbl = Label(win, text="Full dataset", font=("Arial Bold", 10), fg='brown')
lbl.grid(column=9, row=15)  
num777 = tk.Entry(win, width=7, fg='brown', justify='center', font=("Arial Bold", 10))
num777.grid(row = 16, column = 9)
lbl = Label(win, text="Цветовая гамма (от 1 до 10)", font=("Arial Bold", 10), fg='brown')
lbl.grid(column=9, row=17)  
num222 = tk.Entry(win, width=9, fg='brown', justify='center', font=("Arial Bold", 10))
num222.grid(row = 18, column = 9)
lbl = Label(win, text="Назначение: ГНБ-'G', Лит-'L', OCMA-'O', Изол-'I', ГОК-'K'", font=("Arial Bold", 10), fg='green')
lbl.grid(column=9, row=19)  
num666 = tk.Entry(win, width=9, fg='green', justify='center', font=("Arial Bold", 10))
num666.grid(row = 20, column = 9)
lbl = Label(win, text="Координаты вручную", font=("Arial Bold", 10), fg='brown')
lbl.grid(column=9, row=21)  
num333 = tk.Entry(win, width=9, fg='brown', justify='center', font=("Arial Bold", 10))
num333.grid(row = 22, column = 9)
lbl = Label(win, text="Глубина Z, м", font=("Arial Bold", 12), fg='brown')
lbl.grid(column=9, row=27)  
num444 = tk.Entry(win, width=10, fg='brown', justify='center', font=("Arial Bold", 11))
num444.grid(row = 28, column = 9)
lbl = Label(win, text="Показывать линейную графику", font=("Arial Bold", 10), fg='black')
lbl.grid(column=9, row=8)  
num555 = tk.Entry(win, width=6, fg='black', justify='center', font=("Arial Bold", 10))
num555.grid(row = 9, column = 9)
lbl = Label(win, text="Area", font=("Arial Bold", 10), fg='blue')
lbl.grid(column=9, row=10)  
num_3 = tk.Entry(win, width=9, fg='blue', justify='center', font=("Arial Bold", 10))
num_3.grid(row = 11, column = 9)

##res10 = tk.Label(win, text = "-")
##res10.grid(row = 15, column = 0)
##res11 = tk.Label(win, text = "-")
##res11.grid(row = 16, column = 0)
##res12 = tk.Label(win, text = "-")
##res12.grid(row = 17, column = 0)
##

res1 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 10))
res1.grid(row = 13, column = 9)
res2 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 10))
res2.grid(row = 14, column = 9)
##res3 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 12))
##res3.grid(row = 22, column = 9)
res28 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 12))
res28.grid(row = 29, column = 9)
res26 = tk.Label(win, text = " ", fg='black', font = ("Arial Bold", 10))
res26.grid(row = 24, column = 9)
resf = tk.Label(win, text = " ", fg='red', font = ("Arial Bold", 10))
resf.grid(row = 26, column = 9)
res30 = tk.Label(win, text = " ", fg='black', font = ("Arial Bold", 10))
res30.grid(row = 33, column = 0)
res31 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 10))
res31.grid(row = 34, column = 0)
res32 = tk.Label(win, text = " ", fg='blue', font = ("Arial Bold", 12))
res32.grid(row = 34, column = 9)
res33 = tk.Label(win, text = " ", fg='green', font = ("Arial Bold", 12))
res33.grid(row = 35, column = 9)
####
####res18 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 12))
####res18.grid(row = 31, column = 0)
####res19 = tk.Label(win, text = " ", fg='red', font = ("Arial Bold", 10))
####res19.grid(row = 32, column = 0)
##
##res21 = tk.Label(win, text = " ", fg='brown', font = ("Arial Bold", 12))
##res21.grid(row = 14, column = 9)
##res22 = tk.Label(win, text = "-", fg='black', font = ("Arial Bold", 10))
##res22.grid(row = 32, column = 0)
##res23 = tk.Label(win, text = "-", fg='blue', font = ("Arial Bold", 10))
##res23.grid(row = 35, column = 0)
##res24 = tk.Label(win, text = "-", fg='blue', font = ("Arial Bold", 10))
##res24.grid(row = 36, column = 0)
##res25 = tk.Label(win, text = "-", fg='blue', font = ("Arial Bold", 10))
##res25.grid(row = 37, column = 0)
##res26 = tk.Label(win, text = "-", fg='red', font = ("Arial Bold", 9))
##res26.grid(row = 35, column = 3)
##res27 = tk.Label(win, text = "-", fg='red', font = ("Arial Bold", 9))
##res27.grid(row = 37, column = 3)

frame1 = tk.LabelFrame(win, text="Excel Data")
frame1.place(height=200, width=440)

# The file/file path text
label_file = tk.Label(win, text="No File Selected")
label_file.place(rely=0, relx=0)


#Treeview Widget
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget

def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=(("xlsx files", "*.xlsx"),("All Files", "*.*")))
    label_file["text"] = filename
    return None

def Load_excel_data():
    
    global aaa
    global bbb
    global df
    for i in tv1.get_children():
        tv1.delete(i)     # clearing screen (table tv1)
    """If the file selected is valid this will load the file into the Treeview"""
    file_path = label_file["text"]
    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename)
    

    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name

    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert   

    
def ddf():
##    try:
    resf.configure(text = " ")

    import cv2
    import imageio
    
    global df

    df_rr = df.copy()
    ## Определяем границы месторождения
    res30.configure(text = " ")
    res31.configure(text = " ")
    res32.configure(text = " ")
    res33.configure(text = " ")
    
    def sd(x):
        x1 = []
        for i in x:
            i = str(i)
            x1.append(i.replace(",", "."))
        return x1
    tg = list(df.columns)
    t_g = list(df.columns)
    X_g = list(df[tg[16]])
    X_g = sd(X_g)
    X_g = [float(item) for item in X_g]
    #a = [x for x in a if str(x) != 'nan']
    Y_g = list(df[tg[17]])
    Y_g = sd(Y_g)
    Y_g = [float(item) for item in Y_g]

##        img = cv2.imread('C:\\west\\west.png', cv2.IMREAD_UNCHANGED)
##        hh, ww = img.shape[:2]
##
##        result = img.copy()
    cc = 0
    at_g = []
    bt_g = []
    for i in range(len(X_g)):
        y_gs = int(Y_g[cc])
        y_ge = int(Y_g[cc+1])
        x_gs = int(X_g[cc])
        x_ge = int(X_g[cc+1])
##            cv2.line(result, (y_gs, x_gs), (y_ge, x_ge), (200, 100, 0, 500), 2)
        
        at_g.append((y_gs, x_gs))
        bt_g.append((y_ge, x_ge))
        print(at_g)
        print(bt_g)
        cc+=1
        if X_g[cc+1] == 0:
            break

    ab_g = []
    for i in range(len(at_g)):
        ab_g.append(at_g[i])
        ab_g.append(bt_g[i])
    x_g = []
    y_g = []
    for i in range(len(ab_g)):
        x_g.append(5267188.06 + 1.41*ab_g[i][1])
        y_g.append(14715755.24 + 1.426*ab_g[i][0])
    w_g = []
    for i in range(len(x_g)):
        w_g.append((x_g[i], y_g[i]))
    Xmin = min(x_g)
    Xmax = max(x_g)
    Ymin = min(y_g)
    Ymax = max(y_g)
    
##        cv2.imshow("RESULT", result)
##        cv2.waitKey(q)


    if list(num444.get().split()):    
        h = round(len(aa)/5)  
    else:
        h = 1
    F = []
    FX = []
    fh = []
    fx = []
    W_ = []
    XMAX = []
    XMIN = []
    YMAX = []
    YMIN = []
    S_R = []
    SS_1 = []
    LEN = []
    KT = []
    MO = []
    MS = []
    W_N = []
    W_F = []
    VO = []
    if list(num444.get().split()):
        if list(num_3.get().split()):
            e = list(num_3.get().split())
            e = [int(item) for item in e]
            es = e[0]
            h = round(len(aa)/es)
            print(h)
        else:
            h = len(aa)
    
    for i in range(h):
        if list(num444.get().split()):
            if list(num_3.get().split()):
                at = aa[round(es*i) : round(es*(i+1))]
                bt = bb[round(es*i) : round(es*(i+1))]
            else:
                at = [aa[i]]
                bt = [bb[i]]
        
            if (len(at) == 1) and (at != bt):
                at_ = [at[0]]
                bt_ = [bt[0]]

                ab = at_+ bt_
            
                x_ = []
                y_ = []
                for i in range(len(ab)):
                    x_.append(5267188.06 + 1.41*ab[i][1])
                    y_.append(14715755.24 + 1.426*ab[i][0])
                w_k = []
                for i in range(len(x_)):
                    w_k.append((x_[i], y_[i]))
                W_N.append(w_k[0])
                W_F.append(w_k[1])
                Xmin = min(x_) 
                Xmax = max(x_) 
                Ymin = min(y_) 
                Ymax = max(y_)

                
                Xmin_ = x_[0] 
                Xmax_ = x_[1] 
                Ymin_ = y_[0] 
                Ymax_ = y_[1]
            
                Xt = 5267188.06 + 1.41*bt[-1][1]
                Yt = 14715755.24 + 1.426*bt[-1][0]
            
                res30.configure(text = "Xmin-Xmax-Ymin-Ymax: %s - %s - %s - %s " % (round(Xmin,1), round(Xmax,1), round(Ymin,1), round(Ymax,1)))
                res31.configure(text = " " )   

                S = (Xmax - Xmin)*(Ymax - Ymin)
                XMIN.append(Xmin)
                XMAX.append(Xmax)
                YMIN.append(Ymin)
                YMAX.append(Ymax)
                
            elif (len(at) == 1) and (at == bt):
                Xt = 5267188.06 + 1.41*bt[-1][1]
                Yt = 14715755.24 + 1.426*bt[-1][0]
                Xmin = Xt
                Xmax = Xmin
                Ymin = Yt
                Ymax = Ymin
                res30.configure(text = " ")
                res31.configure(text = "Координаты точки X-Y: %s - %s" % (round(Xt,1), round(Yt,1)))
                XMIN.append(Xmin)
                XMAX.append(Xmax)
                YMIN.append(Ymin)
                YMAX.append(Ymax)
            else:
                at_ = at
                bt_ = bt

                print('КООРДИНАТЫ НАЧАЛО И КОНЦЫ', at, bt)

                ab = []
                for i in range(len(at_)):
                    ab.append(at_[i])
                    ab.append(bt_[i])
                x_ = []
                y_ = []
                for i in range(len(ab)):
                    x_.append(5267188.06 + 1.41*ab[i][1])
                    y_.append(14715755.24 + 1.426*ab[i][0])
                w_ = []
                for i in range(len(x_)):
                    w_.append((x_[i], y_[i]))
                
                Xmin = min(x_) 
                Xmax = max(x_) 
                Ymin = min(y_) 
                Ymax = max(y_)

                        
                Xt = 5267188.06 + 1.41*bt[-1][1]
                Yt = 14715755.24 + 1.426*bt[-1][0]
                
                W_.append(w_)
                res30.configure(text = "Xmin-Xmax-Ymin-Ymax: %s - %s - %s - %s " % (round(Xmin,1), round(Xmax,1), round(Ymin,1), round(Ymax,1)))
                res31.configure(text = " ")
                XMIN.append(Xmin)
                XMAX.append(Xmax)
                YMIN.append(Ymin)
                YMAX.append(Ymax)
                # НАХОЖДЕНИЕ ПЛОЩАДИ ПОЛИГОНА ИЗ ТОЧЕК
                def explode_xy(xy):
                    xl=[]
                    yl=[]
                    for i in range(len(xy)):
                        xl.append(xy[i][0])
                        yl.append(xy[i][1])
                    return xl,yl

                def shoelace_area(x_list,y_list):
                    a1,a2=0,0
                    x_list.append(x_list[0])
                    y_list.append(y_list[0])
                    for j in range(len(x_list)-1):
                        a1 += x_list[j]*y_list[j+1]
                        a2 += y_list[j]*x_list[j+1]
                    l = abs(a1-a2)/2
                    return l

                xy_e = explode_xy(w_)

                S_r = shoelace_area(xy_e[0],xy_e[1])
                S_R.append(round(S_r,1))
            
        def sd(x):
            x1 = []
            for i in x:
                i = str(i)
                x1.append(i.replace(",", "."))
            return x1
        t = list(df.columns)
        a = list(df[t[0]])
        a = sd(a)
        a = [float(item) for item in a]
        #a = [x for x in a if str(x) != 'nan']
        b = list(df[t[1]])
        b = sd(b)
        b = [float(item) for item in b]
        c = list(df[t[2]])
        c = sd(c)
        c = [float(item) for item in c]
        d = list(df[t[3]])
        d = sd(d)
        d = [float(item) for item in d]
        e = list(df[t[4]])
        e = sd(e)
        e = [float(item) for item in e]
        f = list(df[t[5]])
        f = sd(f)
        f = [float(item) for item in f]
        g = list(df[t[6]])
        g = sd(g)
        g = [float(item) for item in g]
        xx = list(df[t[7]])
        xx = sd(xx)
        xx = [float(item) for item in xx]
        yy = list(df[t[8]])
        yy = sd(yy)
        yy = [float(item) for item in yy]
        zz = list(df[t[9]])
        zz = sd(zz)
        zz = [float(item) for item in zz]
        hh = list(df[t[10]])
        hh = sd(hh)
        hh = [float(item) for item in hh]
        skv = list(df[t[11]])
        skv = sd(skv)
        skv = [float(item) for item in skv]

        # чтение функциональных столбцов

        lit = list(df[t[18]])
        lit = sd(lit)
        lit = [float(item) for item in lit]
        ocma = list(df[t[19]])
        ocma = sd(ocma)
        ocma = [float(item) for item in ocma]
        hid = list(df[t[20]])
        hid = sd(hid)
        hid = [float(item) for item in hid]
        gok = list(df[t[21]])
        gok = sd(gok)
        gok = [float(item) for item in gok]
        gnb = list(df[t[22]])
        gnb = sd(gnb)
        gnb = [float(item) for item in gnb]
  
        df_r = df.copy()

        col = 'coolwarm'
        if list(num222.get().split()):
            color = list(num222.get().split())
            color = [int(item) for item in color]
            if color[0] == 1:
                col = 'viridis'
            elif color[0] == 2:
                col = 'plasma'
            elif color[0] == 3:
                col = 'inferno'
            elif color[0] == 4:
                col = 'magma'
            elif color[0] == 5:
                col = 'cividis'
            elif color[0] == 6:
                col = 'Set1'
            elif color[0] == 7:
                col = 'Blues'
            elif color[0] == 8:
                col = 'Spectral'
            elif color[0] == 9:
                col = 'RdBu_r'
            elif color[0] == 10:
                col = 'bwr_r'
            

##        df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g, t[7]:xx, t[8]:yy, t[10]:hh, t[11]:skv})
##        t = list(df.columns)
##        tt = t.copy()

        # таблица с учетом функциональности
        df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g, t[7]:xx, t[8]:yy, t[9]:zz, t[10]:hh, t[11]:skv, t[18]:lit, t[19]:ocma, t[20]:hid, t[21]:gok, t[22]:gnb})
        t = list(df.columns)
        tt = t.copy()

        ass = pd.DataFrame({'X':x_g, 'Y':y_g})
        ass = ass.dropna()
        ass['h'] = 770
        ass['r'] = 0
        
        print('матрица точек', ass)

        

    ##        one = list(num1.get().split())
    ##        aa = [float(item) for item in one]
    ##        two = list(num2.get().split())
    ##        bb = [float(item) for item in two]
    ##        three = list(num3.get().split())
    ##        cc = [float(item) for item in three]
    ##        four = list(num4.get().split())
    ##        dd = [float(item) for item in four]
    ##        five = list(num10.get().split())
    ##        ee = [float(item) for item in five]
    ##        six = list(num11.get().split())
    ##        ff = [float(item) for item in six]
    ##        zer = list(num0.get().split())
    ##        zz = [float(item) for item in zer]
    ##
    ##        dfx = pd.DataFrame({t[0]:zz, t[1]:aa, t[2]:bb, t[3]:cc, t[4]:dd, t[5]:ee, t[6]:ff})
    ##        dx = dfx.copy()
        dff = df.copy()

        dfv = df.query('Электропроводность > 0')
        dfv['Глубина'].corr(dfv['Электропроводность'])
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = dfv['Глубина']
        values_y = dfv['Электропроводность']
        args, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Электропроводность'].isna(), 'Электропроводность'] = df.loc[df['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]

        dfv = df.query('Монтмориллонит > 0')
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = dfv['Глубина']
        values_y = dfv['Монтмориллонит']
        args_1, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Монтмориллонит'].isna(), 'Монтмориллонит'] = df.loc[df['Монтмориллонит'].isna(), 'Глубина']*args_1[0]+args_1[1]

        dfv = df.query('Песок > 0')
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = dfv['Монтмориллонит']
        values_y = dfv['Песок']
        args_2, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Песок'].isna(), 'Песок'] = df.loc[df['Песок'].isna(), 'Монтмориллонит']*args_2[0]+args_2[1]

        dfv = df.query('КОЕ > 0')
        def mop(values_x,a,b):
            return a * values_x + b 
        values_x = dfv['Монтмориллонит']
        values_y = dfv['КОЕ']
        args_3, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['КОЕ'].isna(), 'КОЕ'] = df.loc[df['КОЕ'].isna(), 'Монтмориллонит']*args_3[0]+args_3[1]

        dfv = df.query('Влажность > 0')
        dfv['Монтмориллонит'].corr(dfv['Влажность'])
        def mop(values_x,a,b,c,d):
            return a * values_x + b 
        values_x = dfv['Монтмориллонит']
        values_y = dfv['Влажность']
        args, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Влажность'].isna(), 'Влажность'] = df.loc[df['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]
        
        dfv = df.query('Индекс > 0')
        dfv['Монтмориллонит'].corr(dfv['Индекс'])
        def mop(values_x,a,b,c,d):
            return a * values_x + b 
        values_x = dfv['Монтмориллонит']
        values_y = dfv['Индекс']
        args, covar = curve_fit(mop, values_x, values_y)
        df.loc[df['Индекс'].isna(), 'Индекс'] = df.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]

        dfg = df.copy()    

        print(df.isna().sum())

        # подготовка данных для определения функциональности
        
        df_p = dfg.iloc[100:105]
        def stand(df, YY, df_p):
            X_train, X_test, y_train, y_test = train_test_split(
            df[['Глубина', 'Влажность', 'Песок', 'Индекс', 'Электропроводность', 'КОЕ', 'Монтмориллонит', 'X', 'Y', YY]].drop(columns=[YY]), 
            df[YY], 
            test_size=0.3,
            random_state=120,
            shuffle=True)
            g = ['Глубина','Влажность', 'Песок', 'Индекс', 'Электропроводность', 'КОЕ', 'Монтмориллонит', 'X', 'Y']
            y_test_ = y_test.copy() 
            X_train_ = X_train.copy()
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[g])
            X_test_scaled = scaler.transform(X_test[g])
            X_train_q = pd.DataFrame(X_train_scaled, columns=g)
            X_test_q = pd.DataFrame(X_test_scaled, columns=g)
    
            X_train = torch.FloatTensor(X_train_q.values)
            X_test = torch.FloatTensor(X_test_q.values)
    
            y_train = torch.FloatTensor(y_train.values)
            y_test = torch.FloatTensor(y_test.values)

            df_p_scaled = scaler.transform(df_p[g])
            df_pp = pd.DataFrame(df_p_scaled, columns=g)
            df_pp = torch.FloatTensor(df_pp.values)
            return X_train_q, X_test_q, X_train, X_test, y_train, y_test, df_pp


        def n_line(n_in_neurons, n_hidden_neurons_1, n_out_neurons, lr, batch_size, drop_rate, X_train, X_test, y_train, y_test, df_n):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.use_deterministic_algorithms(True)    
            
            class Net(nn.Module):
                def __init__(self, n_in_neurons, n_hidden_neurons_1, n_out_neurons):
                    super(Net, self).__init__()
                    
                    self.fc1 = nn.Linear(n_in_neurons, n_hidden_neurons_1)
                    self.dp = nn.Dropout(p=drop_rate)
                    self.act1 = nn.Tanh()

                    self.fc2 = nn.Linear(n_hidden_neurons_1, n_out_neurons)
                    self.act2 = nn.Sigmoid()   
                
                  #  self.fc3 = nn.Linear(n_hidden_neurons_2, n_out_neurons)
                  #  self.act3 = nn.Sigmoid()
                
                   # self.fc4 = nn.Linear(n_hidden_neurons_3, n_hidden_neurons_4)
                   # self.act4 = nn.Sigmoid()

                   # self.fc5 = nn.Linear(n_hidden_neurons_4, n_out_neurons)
                   # self.act5 = nn.ReLU()
                    nn.init.xavier_normal_(self.fc1.weight)
                    nn.init.normal_(self.fc1.bias, mean = 0.5, std = 1)
                    nn.init.xavier_normal_(self.fc2.weight)
                    nn.init.normal_(self.fc2.bias, mean = 0.5, std = 1)
                 #   nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
                 #   nn.init.normal_(self.fc2.bias, mean = 0.5, std = 1)
                 #   nn.init.xavier_normal_(self.fc3.weight)
                 #   nn.init.normal_(self.fc3.bias, mean = 0.5, std = 1)
                  #  nn.init.xavier_normal_(self.fc4.weight)
                  #  nn.init.normal_(self.fc4.bias, mean = 0.5, std = 1)
                  #  nn.init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
                  #  nn.init.normal_(self.fc5.bias, mean = 0.5, std = 1)
                
                def forward(self, x):
                    x = self.fc1(x)
                    x = self.dp(x)
                    x = self.act1(x)

                    x = self.fc2(x)
                    x = self.act2(x)

                 #   x = self.fc3(x)
                 #   x = self.act3(x)
                
                  #  x = self.fc4(x)
                  #  x = self.act4(x)
                
                  #  x = self.fc5(x)
                  #  x = self.act5(x)
                
                    return x
            net = Net(n_in_neurons, n_hidden_neurons_1, n_out_neurons)
            optimizer = torch.optim.Adam(net.parameters(), lr = lr)
            loss = nn.BCELoss()
            num_epochs = 200
            num_batches = ceil(len(X_train)/batch_size)
            Accuracy = []
            Roc_auc = []
            PRED = []
            loss_ = []
            n = []
            roc_train = []
            DF_N = []
            for epoch in range(num_epochs):
                order = np.random.permutation(len(X_train)) 
                for i in range(num_batches):
                    net.train()
                    start_index =  i * batch_size 
                    optimizer.zero_grad()
          
                    batch_indexes = order[start_index:start_index+batch_size] 
                    X_batch = X_train[batch_indexes]
                    y_batch = y_train[batch_indexes]

                    
                    preds = net.forward(X_batch).flatten()
            
                    loss_value = loss(preds, y_batch)

                    loss_value.backward()       

                    optimizer.step()
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    net.eval()
                    test_preds = net.forward(X_test).flatten()
                    accuracy = (torch.round(test_preds) == y_test).float().mean().data
                    roc_auc = roc_auc_score(y_test.detach().numpy(), test_preds.detach().numpy())
                    Accuracy.append((torch.round(test_preds) == y_test).float().mean().data)
                    PRED.append(test_preds)
                    Roc_auc.append(roc_auc)
                    n.append(epoch)
                    
                    preds_ = net.forward(X_train).flatten()
                    loss_.append((torch.round(preds_) == y_train).float().mean().data)
                    roc_train_ = roc_auc_score(y_train.detach().numpy(), preds_.detach().numpy())
                    roc_train.append(roc_train_)
                    DF_N.append(net.forward(df_n).flatten())
            return loss_[Roc_auc.index(max(Roc_auc))], max(Roc_auc), Accuracy[Roc_auc.index(max(Roc_auc))], PRED[Roc_auc.index(max(Roc_auc))], y_test, n[Roc_auc.index(max(Roc_auc))], roc_train[Roc_auc.index(max(Roc_auc))], DF_N[Roc_auc.index(max(Roc_auc))] 
        
            
        ## Функция для факторов значимости функциональных сырьевых моделей

        def drawing(model, X_train_q, y_train):
            model.fit(X_train_q, y_train)
            feature_importance = pd.DataFrame({'Feature': X_train_q.columns, 'Importance': np.abs(model.feature_importances_)})
            feature_importance = feature_importance.sort_values(by = 'Importance', ascending = True)
            sns.set_style('white')
            feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
            plt.show();
        
        
        ## Обучение на исходных данных для предсказания горизонта (сразу делаем обрезание - обучаем на данных только выделенного объекта)
       
    ##        if list(num444.get().split()):
    ##            df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]



        num_columns = ['Глубина', 'Влажность', 'Песок', 'Индекс', 'Электропроводность', 'КОЕ', 'Монтмориллонит', 'X', 'Y']
        num_columns_1 = ['Глубина', 'X', 'Y']
        t_1 = ['Глубина', 'X', 'Y', 'Hor']
        t_2 = num_columns + ['Hor']
        df_h = df[t_1]
        if list(num777.get().split()):
            df_h = df[t_2]
        t1 = df_h.columns
        RANDOM_STATE = 42

        


        

        y = df_h[t1[-1]]
        X = df_h.drop([t1[-1]], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = RANDOM_STATE)
        scaler__ = StandardScaler()
        
        X_train_st = scaler__.fit_transform(X_train)
        X_test_st = scaler__.transform(X_test)
        models = [
             [Lasso(), 'Линейная регрессия Lasso'],
             [Ridge(), 'Линейная регрессия Ridge'],
             [RandomForestRegressor(n_estimators = 200, random_state = 42), 'Случайный лес'],
             [GradientBoostingRegressor(n_estimators = 200, random_state = 42), 'Градиентный бустинг'],
             [DecisionTreeRegressor(random_state = 42), 'Дерево решений']
            ]

        def prediction(mod, X_train, y_train, X_test, y_test, name):
            model = mod
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics(y_test, y_pred, name)
        p_ = []
        w_w = []
        for i in models:
            model = i[0]
            model.fit(X_train_st, y_train)
            y_pred = model.predict(X_test_st)
            p_.append(r2_score(y_test,y_pred))
            w_w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)

        
    ##    data_preprocessor = ColumnTransformer(
    ##    [
    ##        ('num', MinMaxScaler(), num_columns_1)
    ##    ], 
    ##    remainder='passthrough'
    ##    )
    ##    X_train_p = pd.DataFrame(
    ##    data_preprocessor.fit_transform(X_train),
    ##    columns=data_preprocessor.get_feature_names_out()
    ##    )
    ##
    ##    X_test_p = pd.DataFrame(
    ##    data_preprocessor.transform(X_test),
    ##    columns=data_preprocessor.get_feature_names_out()    
    ##    )
    ##    pipe_final = Pipeline(
    ##    [
    ##        ('preprocessor', data_preprocessor),
    ##        ('models', DecisionTreeRegressor(random_state=RANDOM_STATE))
    ##    ]
    ##    )
    ##    param_grid = [
    ##    {
    ##        'models': [DecisionTreeRegressor(random_state=RANDOM_STATE)],
    ##        'models__max_depth': range(2,40),
    ##        'models__max_features': range(2,40),
    ##        'models__min_samples_leaf': range(1, 20),
    ##        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    ##    },
    ##    {
    ##        'models': [RandomForestRegressor(random_state=RANDOM_STATE)],
    ##        'models__max_depth': range(2,40),
    ##        'models__max_features': range(2,40),
    ##        'models__min_samples_leaf': range(1, 20),
    ##        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    ##    },
    ##    {
    ##        'models': [GradientBoostingRegressor(random_state=RANDOM_STATE)],
    ##        'models__max_depth': range(2,40),
    ##        'models__max_features': range(2,40),
    ##        'models__min_samples_leaf': range(1, 20),
    ##        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    ##    }
    ##    ]      
    ##    randomized_search = RandomizedSearchCV(
    ##    pipe_final, 
    ##    param_grid, 
    ##    cv=5,
    ##    scoring='r2',
    ##    random_state=RANDOM_STATE,
    ##    n_jobs=-1
    ##    )
    ##
    ##    
    ##    
    ##    randomized_search.fit(X_train, y_train)
    ##    score_p = randomized_search.best_score_
    ##    y_test_pred =  randomized_search.predict(X_test)
    ##    print(r2_score(y_test, y_test_pred))

    ##    t_1 = ['Глубина', 'X', 'Y', 'Hor']
    ##    df_h = df[t_1]
        
        num_columns_2 = ['Глубина', 'Влажность', 'Песок', 'Индекс', 'Электропроводность', 'КОЕ', 'X', 'Y', 'Монтмориллонит']
        t = ['Глубина', 'X', 'Y', 'Монтмориллонит']
        df = df[t]

        df_sk = dfg[['Скважина', 'Глубина', 'X', 'Y']]
        print('tttttttttttttttttttttttttttttttttttttttt')
        print(len(df_sk), len(df))
     
        df_u = df.copy()

        if list(num111.get().split()):
            seven = list(num111.get().split())
            m = [int(item) for item in seven]
            for i in range(1,8):
                if i == m[0]:
                    df = df.drop(['Монтмориллонит'], axis = 1)
                    hhh = tt[i-1]
                    df = pd.concat([df, dfg[[hhh]]], axis = 1)
                    t = df.columns

            
            t = list(df.columns)

        ## ЛИНЕЙНАЯ ГРАФИКА

        print(df.isna().sum())
        y_ = df[t[-1]]
        X_ = df.drop([t[-1]], axis = 1)
        X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.3, random_state = 42)
        scaler_ = StandardScaler()
        X_train_st_ = scaler_.fit_transform(X_train_)
        X_test_st_ = scaler_.transform(X_test_)

        models_ = [
             [Lasso(), 'Линейная регрессия Lasso'],
             [Ridge(), 'Линейная регрессия Ridge'],
             [RandomForestRegressor(n_estimators = 200, random_state = 42), 'Случайный лес'],
             [GradientBoostingRegressor(n_estimators = 200, random_state = 42), 'Градиентный бустинг'],
             [DecisionTreeRegressor(random_state = 42), 'Дерево решений']
        ]

        def metrics_(y_true, y_pred, title):
            print(title)
            print('MAE: {:.2f}'.format(mean_absolute_error(y_true,y_pred)))
            print('MSE: {:.2f}'.format(mean_squared_error(y_true,y_pred)))
            print('R2: {:.2f}'.format(r2_score(y_true,y_pred)))

        def prediction_(mod, X_train, y_train, X_test, y_test, name):
            model = mod
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics_(y_test, y_pred, name)
        if list(num555.get().split()):
            c = df.corr()
            plt.figure(figsize = (8,8))
            sns.heatmap(c, annot = True, square = True)
            plt.title('Матрица корреляции признаков')
            plt.show()

        for i in models:
            prediction_(i[0],X_train_st_,y_train_,X_test_st_,y_test_,i[1])

        p_ = []
        
        for i in models_:
            model = i[0]
            model.fit(X_train_st_, y_train_)
            y_pred_ = model.predict(X_test_st_)
            p_.append(r2_score(y_test_,y_pred_))
            

        #res26.configure(text = "r2_score: %s" % round(max(p_),2))
        def graph(model, f):
            u = t.copy()
            dft = df.copy()
            dft = dft.drop(t[-1], axis = 1)
            for i in dft.columns:
                dft[i] = round(dft[i].mean(),2)
            del u[-1]
            for i in u:
                dfr = dft.copy()
                for j in range(len(dfr[i])):
                    mi = df[i].min()
                    ma = df[i].max()
                    n = (ma-mi)/(len(dft[i])-1)
                    dfr[i][j] = mi + j*n
                X_st = scaler_.transform(dfr)
                y_pred = model.predict(X_st)
                dfr[t[-1]] = y_pred
                dg = dfr.groupby(i).agg({t[-1]:'mean'})

                def mop(values_x,a,b,c,d):
                    return a * values_x**3 + b * values_x**2 + c * values_x + d
                def mop1(values_x,a,b,c):
                    return a * values_x**2 + b * values_x + c 
                def mop2(values_x,a,b):
                    return a * values_x + b 
                values_x = dfr[i]
                values_y = dfr[t[-1]]
        
                args, covar = curve_fit(mop, values_x, values_y)
                y_pred_1 = mop(values_x, *args)
                r2_1 = r2_score(values_y, y_pred_1)
                args1, covar1 = curve_fit(mop1, values_x, values_y)
                y_pred_2 = mop1(values_x, *args1)
                r2_2 = r2_score(values_y, y_pred_2)
                args2, covar2 = curve_fit(mop2, values_x, values_y)
                y_pred_3 = mop2(values_x, *args2)
                r2_3 = r2_score(values_y, y_pred_3)
                r2 = [r2_1,r2_2,r2_3]
                y = []
                if max(r2) == r2_1:
                    for j in range(len(dfr[i])):
                        y.append(args[0] * dfr[i][j]**3 + args[1] * dfr[i][j]**2 + args[2] * dfr[i][j] + args[3])
                elif max(r2) == r2_2:
                    for j in range(len(dfr[i])):
                        y.append(args1[0] * dfr[i][j]**2 + args1[1] * dfr[i][j] + args1[2])    
                elif max(r2) == r2_3:
                    for j in range(len(dfr[i])):
                        y.append(args2[0] * dfr[i][j] + args2[1]) 
                dgr = pd.DataFrame({'x':list(dfr[i]), 'y':y})
        
                fig, axes = plt.subplots(1, 1, figsize=(8, 5))
                sns.set_style('whitegrid')
                sns.set_palette('bright')
        
                sns.lineplot(dgr.pivot_table(index = 'x', values = 'y', aggfunc = 'mean'), color = 'red')
        
                sns.scatterplot(data=dg)
                #sns.set(rc={'figure.figsize':(8,5)})
                axes.set(xlabel= 'Переменный параметр <{}>'.format(i),
                ylabel='Функция отклика <{}>'.format(t[-1]),
                title ='Усредненная параметрическая диаграмма {} при переменном факторе {} ({})'.format(t[-1], i, f))
                plt.legend(title = '{}'.format(dft.iloc[[0]]), loc=2, bbox_to_anchor=(1, 1), fontsize = 10)
                plt.xticks(rotation = 10)    
                plt.show()

        if list(num555.get().split()):    
            if max(p_) == p_[0]:
                graph(models_[0][0], models_[0][1])
            elif max(p_) == p_[1]:
                graph(models_[1][0], models_[1][1])
            elif max(p_) == p_[2]:
                graph(models_[2][0], models_[2][1])
            elif max(p_) == p_[3]:
                graph(models_[3][0], models_[3][1])
            elif max(p_) == p_[4]:
                graph(models_[4][0], models_[4][1])   
    ##
    ##    
        ## КОННЕЦ ЛИНЕЙНОЙ ГРАФИКИ
            
        ## создаем функцию, которая возвращает расчетную матрицн X-Y-Z-цель в пределах максимального и мин значений X и Y. Также отдает площадб клетки м2
            
        def table(u, Xmin, Xmax, Ymin, Ymax):
            global df
            t = df.columns
            df = df.drop([t[-1]], axis = 1)
            hh = tt[u-1]
            df = pd.concat([df, dfg[[hh]]], axis = 1)
            t = df.columns
    ##            df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]
            y = df[t[-1]]
            X = df.drop([t[-1]], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
            scaler = StandardScaler()
            X_train_st = scaler.fit_transform(X_train)
            X_test_st = scaler.transform(X_test)

            models = [
             [Lasso(), 'Линейная регрессия Lasso'],
             [Ridge(), 'Линейная регрессия Ridge'],
             [RandomForestRegressor(n_estimators = 200, random_state = 42), 'Случайный лес'],
             [GradientBoostingRegressor(n_estimators = 200, random_state = 42), 'Градиентный бустинг'],
             [DecisionTreeRegressor(random_state = 42), 'Дерево решений']
            ]

            def prediction(mod, X_train, y_train, X_test, y_test, name):
                model = mod
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics(y_test, y_pred, name)
            p = []
            w = []
           # if Xmin != Xmax:
##            if at != bt:
            S = (Xmax - Xmin)*(Ymax - Ymin)
            for i in models:
                model = i[0]
                model.fit(X_train_st, y_train)
                y_pred = model.predict(X_test_st)
                p.append(r2_score(y_test,y_pred))
                w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)

            if list(num444.get().split()):
    ##            X_n = [Xmin+20*i for i in range(0,round((Xmax - Xmin)/20))]
    ##            Y_n = [Ymin+20*i for i in range(0,round((Ymax - Ymin)/20))]
    ##
    ##            S_1 = S/(len(X_n)*len(Y_n))
                if (len(at) == 1) and (at != bt):
                    dX = [Xmin+i for i in range(0,round(Xmax - Xmin))]
                    dY = []
                    for i in dX:
                        if Xmax_ == Xmin_:
                            dY.append(Ymin_ + (i - Xmin_)*1)
                        else:
                            dY.append(Ymin_ + (i - Xmin_)*(Ymax_ - Ymin_)/(Xmax_ - Xmin_)) 
                    #S_1 = S/(len(dX)*len(dY))
                    # S_1 = 1
                elif at == bt:
                    dX = [Xt]
                    dY = [Yt]
                    S_1 = 1
                else:
                    X_n = [Xmin+i for i in range(0,round(Xmax - Xmin), 5)]
                    Y_n = [Ymin+i for i in range(0,round(Ymax - Ymin), 5)]
                    S_1 = S/(len(X_n)*len(Y_n))
                   
            else:
                X_n = [Xmin+i for i in range(0,round(Xmax - Xmin), 20)]
                Y_n = [Ymin+i for i in range(0,round(Ymax - Ymin), 20)]

                S_1 = S/(len(X_n)*len(Y_n))
                
            Глубина_n = [0+1*i for i in range(0,36)]

            if list(num444.get().split()):
                if len(at) == 1:
                    dX_N = []
                    for i in dX:
                        for j in range(len(Глубина_n)):
                            dX_N.append(i)
                    dY_N = []
                    for i in dY:
                        for j in range(len(Глубина_n)):
                            dY_N.append(i)
                    H = []
                    for i in range(len(dX)):
                        H += Глубина_n
                    X_F = pd.DataFrame({'Глубина':H, 'X':dX_N, 'Y':dY_N})

##                    Y_1 = []
##                    X_1 = []
##                    for i in range(len(H)):
##                        X_1.append(dX_N[i]+(H[i]*math.tan(1.2))*math.cos(1.5)) 
##                    for i in range(len(H)):
##                        Y_1.append(dY_N[i]+(H[i]*math.tan(1.2))*math.sin(1.5))
##                            
##                    X_F = pd.DataFrame({'Глубина':H, 'X':X_1, 'Y':Y_1})
##                    print('XXXXXXXXXXXXXXXXXX')
##                    print(math.cos(1.5))
                    
                else:
                    Y_N = []
                    for i in range(len(X_n)):
                        Y_N.append(Y_n)
                    X_N = []
                    for i in X_n:
                        for j in range(len(Y_n)):
                            X_N.append(i)
                    Y_NN = []
                    for i in Y_N:
                        Y_NN += i
                    X_NNN = []
                    for i in X_N:
                        for j in range(len(Глубина_n)):
                            X_NNN.append(i)
                    Y_NNN = []
                    for i in Y_NN:
                        for j in range(len(Глубина_n)):
                            Y_NNN.append(i)
                    H = []
                    for i in range(len(Y_NN)):
                        H += Глубина_n
                    X_F = pd.DataFrame({'Глубина':H, 'X':X_NNN, 'Y':Y_NNN})
                    
            else:
                Y_N = []
                for i in range(len(X_n)):
                    Y_N.append(Y_n)
                X_N = []
                for i in X_n:
                    for j in range(len(Y_n)):
                        X_N.append(i)
                Y_NN = []
                for i in Y_N:
                    Y_NN += i
                X_NNN = []
                for i in X_N:
                    for j in range(len(Глубина_n)):
                        X_NNN.append(i)
                Y_NNN = []
                for i in Y_NN:
                    for j in range(len(Глубина_n)):
                        Y_NNN.append(i)
                H = []
                for i in range(len(Y_NN)):
                    H += Глубина_n
                X_F = pd.DataFrame({'Глубина':H, 'X':X_NNN, 'Y':Y_NNN})
            X_F_st = scaler.transform(X_F)
            for i in range(len(p)):
                if p[i] == max(p):
                    model = models[i][0]
                    model.fit(X_train_st, y_train)
                    y_pred_f = model.predict(X_F_st)
            X_F[t[-1]] = y_pred_f
            if list(num444.get().split()):
                if (len(at) == 1) and (at != bt):
                    return X_F, max(p), len(dX)
                else:
                    return X_F, S_1, max(p)
            else:
                return X_F, S_1, max(p)

        ## с помощью table получаем расчетные матрицы (для всего поля или выдел объекта) с полями X-Y-Z-цель 
        if list(num444.get().split()):
            if (len(at) == 1) and (at != bt):
                X_F_1, score_2, Len = table(2, Xmin, Xmax, Ymin, Ymax)
                t11 = X_F_1.columns
            
                X_F_2, score_3, Len = table(3, Xmin, Xmax, Ymin, Ymax)
                t22 = X_F_2.columns    
            
                X_F_3, score_4, Len = table(4, Xmin, Xmax, Ymin, Ymax)
                t33 = X_F_3.columns    
            
                X_F_4, score_5, Len = table(5, Xmin, Xmax, Ymin, Ymax)
                t44 = X_F_4.columns       
            
                X_F_5, score_7, Len = table(7, Xmin, Xmax, Ymin, Ymax)
                t55 = X_F_5.columns     
            
                X_F_6, score_6, Len = table(6, Xmin, Xmax, Ymin, Ymax)
                t66 = X_F_6.columns
                LEN.append(Len)
            else:
                X_F_1, S_1, score_2 = table(2, Xmin, Xmax, Ymin, Ymax)
                t11 = X_F_1.columns
            
                X_F_2, S_1, score_3 = table(3, Xmin, Xmax, Ymin, Ymax)
                t22 = X_F_2.columns    
            
                X_F_3, S_1, score_4 = table(4, Xmin, Xmax, Ymin, Ymax)
                t33 = X_F_3.columns    
            
                X_F_4, S_1, score_5 = table(5, Xmin, Xmax, Ymin, Ymax)
                t44 = X_F_4.columns       
            
                X_F_5, S_1, score_7 = table(7, Xmin, Xmax, Ymin, Ymax)
                t55 = X_F_5.columns     
            
                X_F_6, S_1, score_6 = table(6, Xmin, Xmax, Ymin, Ymax)
                t66 = X_F_6.columns 
        else:
            X_F_1, S_1, score_2 = table(2, Xmin, Xmax, Ymin, Ymax)
            t11 = X_F_1.columns
        
            X_F_2, S_1, score_3 = table(3, Xmin, Xmax, Ymin, Ymax)
            t22 = X_F_2.columns    
        
            X_F_3, S_1, score_4 = table(4, Xmin, Xmax, Ymin, Ymax)
            t33 = X_F_3.columns    
        
            X_F_4, S_1, score_5 = table(5, Xmin, Xmax, Ymin, Ymax)
            t44 = X_F_4.columns       
        
            X_F_5, S_1, score_7 = table(7, Xmin, Xmax, Ymin, Ymax)
            t55 = X_F_5.columns     
        
            X_F_6, S_1, score_6 = table(6, Xmin, Xmax, Ymin, Ymax)
            t66 = X_F_6.columns

##        if (len(at) == 1) and (at != bt):
##            X_F_1, S_1, score_2, Len = table(2, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t11 = X_F_1.columns
##                
##            X_F_2, S_1, score_3, Len = table(3, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t22 = X_F_2.columns    
##                
##            X_F_3, S_1, score_4, Len = table(4, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t33 = X_F_3.columns    
##                
##            X_F_4, S_1, score_5, Len = table(5, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t44 = X_F_4.columns       
##                
##            X_F_5, S_1, score_7, Len = table(7, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t55 = X_F_5.columns     
##                
##            X_F_6, S_1, score_6, Len = table(6, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t66 = X_F_6.columns
##        else:
##            X_F_1, S_1, score_2 = table(2, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t11 = X_F_1.columns
##                
##            X_F_2, S_1, score_3 = table(3, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t22 = X_F_2.columns    
##                
##            X_F_3, S_1, score_4 = table(4, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t33 = X_F_3.columns    
##                
##            X_F_4, S_1, score_5 = table(5, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t44 = X_F_4.columns       
##                
##            X_F_5, S_1, score_7 = table(7, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t55 = X_F_5.columns     
##                
##            X_F_6, S_1, score_6 = table(6, min(df['X']), max(df['X']), min(df['Y']), max(df['Y']))
##            t66 = X_F_6.columns

        t = ['Скважина', 'Глубина', 'X', 'Y', 'Монтмориллонит']
        df = dff[t]


        df_u = df.copy()

        if list(num111.get().split()):
            seven = list(num111.get().split())
            m = [int(item) for item in seven]
            for i in range(1,8):
                if i == m[0]:
                    df = df.drop(['Монтмориллонит'], axis = 1)
                    hhh = tt[i-1]
                    df = pd.concat([df, dfg[[hhh]]], axis = 1)
                    t = df.columns

            t = list(df.columns)

        ## объединяем расчетные поля в итоговую суперматрицу с полями X-Y-Z-цель1-цель2-цель3-...
        
        X_F = X_F_1.copy() 
        X_F['Песок'] = X_F_2[t22[-1]]
        X_F['Индекс'] = X_F_3[t33[-1]]
        X_F['Электропроводность'] = X_F_4[t44[-1]]
        X_F['КОЕ'] = X_F_6[t66[-1]]
        X_F['Монтмориллонит'] = X_F_5[t55[-1]]

        ##    y_pred =  randomized_search.predict(X_F)

        ## в итоговой суперматрице (для матрицы X-Y-Z) рассчитываем горизонт и добавляем в поле Horizont (используется первая модель, обученная на исх данных)  
        
        X_F_h = X_F[list(X_train.columns)]
        
        X_F_st = scaler__.transform(X_F_h)
        for i in range(len(p_)):
            if p_[i] == max(p_):
                model = models[i][0]
                model.fit(X_train_st, y_train)
                y_pred = model.predict(X_F_st)
        X_F['Horizont'] = [round(i,0) for i in list(y_pred)]

        xx = X_F['X']
        yy = X_F['Y']
        X_F = X_F.drop(['X', 'Y'], axis = 1)
        X_F['X'] = xx
        X_F['Y'] = yy

        t_x = X_F.columns
        m_p = max(p_)
        XX_FF = X_F.copy()
        
        ## Проверяем функциональность сырья

        if list(num666.get().split()):
            R = list(num666.get().split())
            if R[0] == 'G':
                X_train_q, X_test_q, X_train_gnb, X_test_gnb, y_train_gnb, y_test_v, df_pp = stand(dfg, 'ГНБ', X_F)
                Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 694, 1, 0.05, round(len(X_train_gnb)/1), 0.39, X_train_gnb, X_test_gnb, y_train_gnb, y_test_v, df_pp)
                df_nn = []
                if len(R)>1:
                   for i in df_n.flatten():
                       if i >= float(R[1]):
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= float(R[1]):
                           ww.append(1)
                       else:
                           ww.append(0) 
                else:
                   for i in df_n.flatten():
                       if i >= 0.5:
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= 0.5:
                           ww.append(1)
                       else:
                           ww.append(0)
                           
                pre_score = precision_score(y_test_v, ww)
                rec_score = recall_score(y_test_v, ww)
                X_F['Class'] = df_nn
                X_F = X_F.query('Class == 1')
                
                if list(num555.get().split()):
                    model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 18, max_features = 20, max_depth = 18)
                    drawing(model, X_train_q, y_train_gnb)
            if R[0] == 'L':
                X_train_q, X_test_q, X_train_lit, X_test_lit, y_train_lit, y_test_v, df_pp = stand(dfg, 'Литейка', X_F)
                Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 396, 1, 0.3, round(len(X_train_lit)/1), 0.56, X_train_lit, X_test_lit, y_train_lit, y_test_v, df_pp)
                df_nn = []
                if len(R)>1:
                   for i in df_n.flatten():
                       if i >= float(R[1]):
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= float(R[1]):
                           ww.append(1)
                       else:
                           ww.append(0) 
                else:
                   for i in df_n.flatten():
                       if i >= 0.5:
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= 0.5:
                           ww.append(1)
                       else:
                           ww.append(0)
                           
                pre_score = precision_score(y_test_v, ww)
                rec_score = recall_score(y_test_v, ww)        
                X_F['Class'] = df_nn
                X_F = X_F.query('Class == 1')
                if list(num555.get().split()):
                    model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 18, max_features = 20, max_depth = 18)
                    drawing(model, X_train_q, y_train_lit)
            if R[0] == 'O':
                X_train_q, X_test_q, X_train_ok, X_test_ok, y_train_ok, y_test_v, df_pp = stand(dfg, 'OCMA', X_F)
                Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 966, 1, 0.05, round(len(X_train_ok)/1), 0.11, X_train_ok, X_test_ok, y_train_ok, y_test_v, df_pp)
                df_nn = []
                if len(R)>1:
                   for i in df_n.flatten():
                       if i >= float(R[1]):
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= float(R[1]):
                           ww.append(1)
                       else:
                           ww.append(0) 
                else:
                   for i in df_n.flatten():
                       if i >= 0.5:
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= 0.5:
                           ww.append(1)
                       else:
                           ww.append(0)
                           
                pre_score = precision_score(y_test_v, ww)
                rec_score = recall_score(y_test_v, ww)
                X_F['Class'] = df_nn
                X_F = X_F.query('Class == 1')
                if list(num555.get().split()):
                    model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 5, max_features = 6, max_depth = 6)
                    drawing(model, X_train_q, y_train_ok)
            if R[0] == 'I':
                X_train_q, X_test_q, X_train_h, X_test_h, y_train_h, y_test_v, df_pp = stand(dfg, 'Гидроизоляция', X_F)
                Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 970, 1, 0.06, round(len(X_train_h)/1), 0.52, X_train_h, X_test_h, y_train_h, y_test_v, df_pp)
                df_nn = []
                if len(R)>1:
                   for i in df_n.flatten():
                       if i >= float(R[1]):
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= float(R[1]):
                           ww.append(1)
                       else:
                           ww.append(0) 
                else:
                   for i in df_n.flatten():
                       if i >= 0.5:
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= 0.5:
                           ww.append(1)
                       else:
                           ww.append(0)
                           
                pre_score = precision_score(y_test_v, ww)
                rec_score = recall_score(y_test_v, ww)
                X_F['Class'] = df_nn
                X_F = X_F.query('Class == 1')
                if list(num555.get().split()):
                    model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 5, max_features = 6, max_depth = 6)
                    drawing(model, X_train_q, y_train_h)
            if R[0] == 'K':
                X_train_q, X_test_q, X_train_g, X_test_g, y_train_g, y_test_v, df_pp = stand(dfg, 'ГОК', X_F)
                Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 780, 1, 0.15, round(len(X_train_g)/1), 0.65, X_train_g, X_test_g, y_train_g, y_test_v, df_pp)
                df_nn = []
                if len(R)>1:
                   for i in df_n.flatten():
                       if i >= float(R[1]):
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= float(R[1]):
                           ww.append(1)
                       else:
                           ww.append(0) 
                else:
                   for i in df_n.flatten():
                       if i >= 0.5:
                           df_nn.append(1)
                       else:
                           df_nn.append(0)
                   ww = []
                   for i in pred.flatten():
                       if i >= 0.5:
                           ww.append(1)
                       else:
                           ww.append(0)
                           
                pre_score = precision_score(y_test_v, ww)
                rec_score = recall_score(y_test_v, ww)
                X_F['Class'] = df_nn
                X_F = X_F.query('Class == 1')
                if list(num555.get().split()):
                    model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 14, max_features = 20, max_depth = 15)
                    drawing(model, X_train_q, y_train_g)
                
        ## вычисляем отклик при условии полного датафрейма 

        if list(num777.get().split()):
            X = dfg.drop(['Hor', 'Z', 'ГОК', 'Гидроизоляция', 'Литейка', 'OCMA', 'ГНБ', 'Скважина', t[-1]], axis = 1)
            y = dfg[t[-1]]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
            scaler = StandardScaler()
            X_train_st = scaler.fit_transform(X_train)
            X_test_st = scaler.transform(X_test)

            models = [
             [Lasso(), 'Линейная регрессия Lasso'],
             [Ridge(), 'Линейная регрессия Ridge'],
             [RandomForestRegressor(n_estimators = 200, random_state = 42), 'Случайный лес'],
             [GradientBoostingRegressor(n_estimators = 200, random_state = 42), 'Градиентный бустинг'],
             [DecisionTreeRegressor(random_state = 42), 'Дерево решений']
            ]

            def prediction(mod, X_train, y_train, X_test, y_test, name):
                model = mod
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics(y_test, y_pred, name)
            p = []
            w = []
            for i in models:
                model = i[0]
                model.fit(X_train_st, y_train)
                y_pred = model.predict(X_test_st)
                p.append(r2_score(y_test,y_pred))
                w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)
            X_F_pp = XX_FF.drop(['Horizont', t[-1]], axis = 1)
            X_F_ss = scaler.transform(X_F_pp)
            for i in range(len(p)):
                if p[i] == max(p):
                    model = models[i][0]
                    model.fit(X_train_st, y_train)
                    y_pred_F = model.predict(X_F_ss)
            X_F_pp[t[-1]] = y_pred_F
            X_F_pp['Horizont'] = X_F['Horizont']
            X_F = X_F_pp.copy()
            t_x = X_F.columns
            m_p = max(p_)

            ## вырезаем из матрицы годную функциональность

            if list(num666.get().split()):
                R = list(num666.get().split())
                if R[0] == 'G':
                    X_train_q, X_test_q, X_train_gnb, X_test_gnb, y_train_gnb, y_test_v, df_pp = stand(dfg, 'ГНБ', X_F)
                    Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 694, 1, 0.05, round(len(X_train_gnb)/1), 0.39, X_train_gnb, X_test_gnb, y_train_gnb, y_test_v, df_pp)
                    df_nn = []
                    if len(R)>1:
                       for i in df_n.flatten():
                           if i >= float(R[1]):
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= float(R[1]):
                               ww.append(1)
                           else:
                               ww.append(0) 
                    else:
                       for i in df_n.flatten():
                           if i >= 0.5:
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= 0.5:
                               ww.append(1)
                           else:
                               ww.append(0)
                               
                    pre_score = precision_score(y_test_v, ww)
                    rec_score = recall_score(y_test_v, ww)
                    X_F['Class'] = df_nn
                    X_F = X_F.query('Class == 1')
                    
                    if list(num555.get().split()):
                        model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 18, max_features = 20, max_depth = 18)
                        drawing(model, X_train_q, y_train_gnb)
                if R[0] == 'L':
                    X_train_q, X_test_q, X_train_lit, X_test_lit, y_train_lit, y_test_v, df_pp = stand(dfg, 'Литейка', X_F)
                    Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 396, 1, 0.3, round(len(X_train_lit)/1), 0.56, X_train_lit, X_test_lit, y_train_lit, y_test_v, df_pp)
                    df_nn = []
                    if len(R)>1:
                       for i in df_n.flatten():
                           if i >= float(R[1]):
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= float(R[1]):
                               ww.append(1)
                           else:
                               ww.append(0) 
                    else:
                       for i in df_n.flatten():
                           if i >= 0.5:
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= 0.5:
                               ww.append(1)
                           else:
                               ww.append(0)
                               
                    pre_score = precision_score(y_test_v, ww)
                    rec_score = recall_score(y_test_v, ww)        
                    X_F['Class'] = df_nn
                    X_F = X_F.query('Class == 1')
                    if list(num555.get().split()):
                        model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 18, max_features = 20, max_depth = 18)
                        drawing(model, X_train_q, y_train_lit)
                if R[0] == 'O':
                    X_train_q, X_test_q, X_train_ok, X_test_ok, y_train_ok, y_test_v, df_pp = stand(dfg, 'OCMA', X_F)
                    Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 966, 1, 0.05, round(len(X_train_ok)/1), 0.11, X_train_ok, X_test_ok, y_train_ok, y_test_v, df_pp)
                    df_nn = []
                    if len(R)>1:
                       for i in df_n.flatten():
                           if i >= float(R[1]):
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= float(R[1]):
                               ww.append(1)
                           else:
                               ww.append(0) 
                    else:
                       for i in df_n.flatten():
                           if i >= 0.5:
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= 0.5:
                               ww.append(1)
                           else:
                               ww.append(0)
                               
                    pre_score = precision_score(y_test_v, ww)
                    rec_score = recall_score(y_test_v, ww)
                    X_F['Class'] = df_nn
                    X_F = X_F.query('Class == 1')
                    if list(num555.get().split()):
                        model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 5, max_features = 6, max_depth = 6)
                        drawing(model, X_train_q, y_train_ok)
                if R[0] == 'I':
                    X_train_q, X_test_q, X_train_h, X_test_h, y_train_h, y_test_v, df_pp = stand(dfg, 'Гидроизоляция', X_F)
                    Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 970, 1, 0.06, round(len(X_train_h)/1), 0.52, X_train_h, X_test_h, y_train_h, y_test_v, df_pp)
                    df_nn = []
                    if len(R)>1:
                       for i in df_n.flatten():
                           if i >= float(R[1]):
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= float(R[1]):
                               ww.append(1)
                           else:
                               ww.append(0) 
                    else:
                       for i in df_n.flatten():
                           if i >= 0.5:
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= 0.5:
                               ww.append(1)
                           else:
                               ww.append(0)
                               
                    pre_score = precision_score(y_test_v, ww)
                    rec_score = recall_score(y_test_v, ww)
                    X_F['Class'] = df_nn
                    X_F = X_F.query('Class == 1')
                    if list(num555.get().split()):
                        model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 5, max_features = 6, max_depth = 6)
                        drawing(model, X_train_q, y_train_h)
                if R[0] == 'K':
                    X_train_q, X_test_q, X_train_g, X_test_g, y_train_g, y_test_v, df_pp = stand(dfg, 'ГОК', X_F)
                    Acc_train, roc_auc_test, Acc_test, pred, test, n, roc_train, df_n = n_line(9, 780, 1, 0.15, round(len(X_train_g)/1), 0.65, X_train_g, X_test_g, y_train_g, y_test_v, df_pp)
                    df_nn = []
                    if len(R)>1:
                       for i in df_n.flatten():
                           if i >= float(R[1]):
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= float(R[1]):
                               ww.append(1)
                           else:
                               ww.append(0) 
                    else:
                       for i in df_n.flatten():
                           if i >= 0.5:
                               df_nn.append(1)
                           else:
                               df_nn.append(0)
                       ww = []
                       for i in pred.flatten():
                           if i >= 0.5:
                               ww.append(1)
                           else:
                               ww.append(0)
                               
                    pre_score = precision_score(y_test_v, ww)
                    rec_score = recall_score(y_test_v, ww)
                    X_F['Class'] = df_nn
                    X_F = X_F.query('Class == 1')
                    if list(num555.get().split()):
                        model = GradientBoostingClassifier(random_state=120, min_samples_leaf= 14, max_features = 20, max_depth = 15)
                        drawing(model, X_train_q, y_train_g)
            
        ## фильтруем суперматрицу и исх данные (X-Y-X-Hor) по номерам горизонтов в ячейке num0
        
        if list(num0.get().split()):
            depth = list(num0.get().split())
            depth = [int(item) for item in depth]
            df_h = df_h.loc[df_h['Hor'].isin(depth)] 
            X_F_ = X_F.loc[X_F['Horizont'].isin(depth)]
        else:
            X_F_ = X_F.copy()

      
        ## готовим двухчленные списки из min и max значений параметров при наличии номера параметра в ячейке num111 

        if list(num111.get().split()):
        
            if list(num1.get().split()):
                vl = list(num1.get().split())
                vl = [float(item) for item in vl]
            else:
                vl = [min(X_F['Влажность']), max(X_F['Влажность'])] 
            if list(num2.get().split()):
                pes = list(num2.get().split())
                pes = [float(item) for item in pes]
            else:
                pes = [min(X_F['Песок']), max(X_F['Песок'])]   
            if list(num3.get().split()):
                ind = list(num3.get().split())
                ind = [float(item) for item in ind]
            else:
                ind = [min(X_F['Индекс']), max(X_F['Индекс'])]   
            if list(num4.get().split()):
                el = list(num4.get().split())
                el = [float(item) for item in el]
            else:
                el = [min(X_F['Электропроводность']), max(X_F['Электропроводность'])]   
            if list(num11.get().split()):
                mont = list(num11.get().split())
                mont = [float(item) for item in mont]
            else:
                mont = [min(X_F['Монтмориллонит']), max(X_F['Монтмориллонит'])]
            if list(num10.get().split()):
                koe = list(num10.get().split())
                koe = [float(item) for item in koe]
            else:
                koe = [min(X_F['КОЕ']), max(X_F['КОЕ'])]

            ## фильтруем суперматрицу в рамках min и max значений параметров при наличии номера параметра в ячейке num111

            X_F_ = X_F_.loc[(X_F_['Влажность'] <= vl[1]) & (X_F_['Влажность'] >= vl[0])]
            X_F_ = X_F_.loc[(X_F_['Песок'] <= pes[1]) & (X_F_['Песок'] >= pes[0])]
            X_F_ = X_F_.loc[(X_F_['Индекс'] <= ind[1]) & (X_F_['Индекс'] >= ind[0])]
            X_F_ = X_F_.loc[(X_F_['Электропроводность'] <= el[1]) & (X_F_['Электропроводность'] >= el[0])]
            X_F_ = X_F_.loc[(X_F_['КОЕ'] <= koe[1]) & (X_F_['КОЕ'] >= koe[0])]
            X_F_ = X_F_.loc[(X_F_['Монтмориллонит'] <= mont[1]) & (X_F_['Монтмориллонит'] >= mont[0])]

            ## оставляем в суперматрице только те горизонты, которые указаны списокм в ячейке num111 (после номера параметра)

            if len(m) > 1:
                g = m.copy()
                g.pop(0)
                X_F_ = X_F_.loc[X_F_['Horizont'].isin(g)]

            ## вырезаем из суперматрицы миниматрицу с ключевым параметром (X-Y-Z-цель) из ячейки num111

            X_F_x = X_F_[['X', 'Y', 'Глубина', tt[m[0]-1]]]
            t_1 = X_F_x.columns
            

        else:

            ## если ключевой параметр не задан в ячейке num111, то вырезаем из исх данных df_h и из суперматрицы горизонты, указанные в num0 
            
            if list(num666.get().split()):
                res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(m_p,2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
            else:
                res26.configure(text = "r2_score: %s" % round(m_p,2))
                
            if list(num0.get().split()):
                depth = list(num0.get().split())
                depth = [int(item) for item in depth]
                df_h = df_h.loc[df_h['Hor'].isin(depth)] 
                X_F_ = X_F.loc[X_F['Horizont'].isin(depth)]
            else:
                X_F_ = X_F.copy()

            ## если указан ключевой параметр в nim111,то из исходных данных (X-Y-Z-цеоль) вырезаем горизонты, указанные списокм в яч num111 после цели
                
        if list(num111.get().split()):
            if len(m) > 1:
                g = m.copy()
                g.pop(0)
                dfff = dff.loc[dff['Hor'].isin(g)]
                df = dfff[['Глубина', 'X', 'Y', tt[m[0]-1]]]
            else:
                dfff = dff.copy()

            ## также из исходных данных X-Y-Z-цель вырезаем горизонты, указанные в ячейке num0
            
            if list(num0.get().split()):
                depth = list(num0.get().split())
                depth = [int(item) for item in depth]
                dfff = dfff.loc[dfff['Hor'].isin(depth)]
                df = dfff[['Глубина', 'X', 'Y', tt[m[0]-1]]]


            ## при наличии цели в num111 фильтруем исходные данные по min и max значениям параметров    
            
            if m[0] == 7:
                if list(num777.get().split()):
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(max(p),2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(max(p),2))
                else:
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(score_7,2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(score_7,2))
##                if list(num11.get().split()):
##                    mont = list(num11.get().split())
##                    mont = [float(item) for item in mont]
##                    df_ = df.loc[(df[t[-1]] <= mont[1]) & (df[t[-1]] >= mont[0])]
##                else:
                df_ = df.copy()
            if m[0] == 6:
                if list(num777.get().split()):
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(max(p),2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(max(p),2))
                else:
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(score_6,2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(score_6,2))
                
##                if list(num10.get().split()):
##                    koe = list(num10.get().split())
##                    koe = [float(item) for item in koe]
##                    df_ = df.loc[(df[t[-1]] <= koe[1]) & (df[t[-1]] >= koe[0])]
##                else:
                df_ = df.copy()
            if m[0] == 5:
                if list(num777.get().split()):
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(max(p),2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(max(p),2))
                else:
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(score_5,2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(score_5,2))
               
##                if list(num4.get().split()):
##                    el = list(num4.get().split())
##                    el = [float(item) for item in el]
##                    df_ = df.loc[(df[t[-1]] <= el[1]) & (df[t[-1]] >= el[0])]
##                else:
                df_ = df.copy()
            if m[0] == 4:
                if list(num777.get().split()):
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(max(p),2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(max(p),2))
                else:
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(score_4,2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(score_4,2))
                
##                if list(num3.get().split()):
##                    ind = list(num3.get().split())
##                    ind = [float(item) for item in ind]
##                    df_ = df.loc[(df[t[-1]] <= ind[1]) & (df[t[-1]] >= ind[0])]
##                else:
                df_ = df.copy()    
            if m[0] == 3:
                if list(num777.get().split()):
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(max(p),2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(max(p),2))
                else:
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(score_3,2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(score_3,2))
                
##                if list(num2.get().split()):
##                    pes = list(num2.get().split())
##                    pes = [float(item) for item in pes]
##                    df_ = df.loc[(df[t[-1]] <= pes[1]) & (df[t[-1]] >= pes[0])]
##                else:
                df_ = df.copy()
            if m[0] == 2:
                if list(num777.get().split()):
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(max(p),2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(max(p),2))
                else:
                    if list(num666.get().split()):
                        res26.configure(text = "r2_score: %s, roc_auc: %s, точность: %s, полнота: %s" % (round(score_2,2), round(roc_auc_test,2), round(pre_score*100), round(rec_score*100)))
                    else:
                        res26.configure(text = "r2_score: %s" % round(score_2,2))
                
##                if list(num1.get().split()):
##                    vl = list(num1.get().split())
##                    vl = [float(item) for item in vl]
##                    df_ = df.loc[(df[t[-1]] <= vl[1]) & (df[t[-1]] >= vl[0])]
##                else:
                df_ = df.copy()

        ## если цель в num111 не указана, то исходные данные не изменны
                    
        else:
            df_ = df.copy()

        kt = int(np.sqrt((Xmax - Xmin)*(Xmax - Xmin) + (Ymax - Ymin)*(Ymax - Ymin)))
        KT.append(kt)

        if list(num111.get().split()):
            if list(num444.get().split()):
                if (len(at) == 1) and (at != bt):
                    u = len(X_F_x)
                    mo = int(u/Len)
                    MO.append(mo)
                    ms = int(u*kt/Len)
                    MS.append(ms)
            FX.append(X_F_x)
            fx.append(df_)
        else:
            if list(num444.get().split()):
                if (len(at) == 1) and (at != bt):
                    u = len(X_F_)
                    mo = int(u/Len)
                    MO.append(mo)
                    ms = int(u*kt/Len)
                    MS.append(ms)
            F.append(X_F_)
            fh.append(df_h)
        if list(num444.get().split()):
            if (len(at) > 1):
                SS_1.append(S_1)
            else:
                SS_1.append(1)
        else:
            SS_1.append(S_1)
        
        df = df_r.copy()

    
    ## при  наличии глубины скв в num444 вырезаем из исходных данных пределы варьирования координат X и Y
        
    
##    if list(num444.get().split()):
##        df_ = df_.loc[(df_['X'] >= Xmin) & (df_['X'] <= Xmax) & (df_['Y'] >= Ymin) & (df_['Y'] <= Ymax)]
##        df_h = df_h.loc[(df_h['X'] >= Xmin) & (df_h['X'] <= Xmax) & (df_h['Y'] >= Ymin) & (df_h['Y'] <= Ymax)]
##        if len(at) > 1:
##            coords = w_
##            poly = Polygon(coords)
##            w_x = list(df_['X'])
##            w_y = list(df_['Y'])
##            w = []
##            for i in range(len(w_x)):
##                w.append((w_x[i], w_y[i]))
##            df_['new'] = w
##            df_.reset_index(inplace = True, drop = True)
##            for i in range(len(w)):
##                if Point(w[i]).within(poly) == False:
##                    df_ = df_.drop(index = i)
##            df_ = df_.drop('new', axis = 1)
##       
##            w_x = list(df_h['X'])
##            w_y = list(df_h['Y'])
##            w = []
##            for i in range(len(w_x)):
##                w.append((w_x[i], w_y[i]))
##            df_h['new'] = w
##            df_h.reset_index(inplace = True, drop = True)
##            for i in range(len(w)):
##                if Point(w[i]).within(poly) == False:
##                    df_h = df_h.drop(index = i)
##            df_h = df_h.drop('new', axis = 1)
##        else:
##            df_ = df_.loc[(df_['X'] >= Xmin) & (df_['X'] <= Xmax) & (df_['Y'] >= Ymin) & (df_['Y'] <= Ymax)]
##            df_h = df_h.loc[(df_h['X'] >= Xmin) & (df_h['X'] <= Xmax) & (df_h['Y'] >= Ymin) & (df_h['Y'] <= Ymax)]

    ## если есть цель в num111, то строим исх диаграмму относительно цели с вырезанными горизонтами и пределами варьирования признаков
        
    if list(num111.get().split()):
        
        coords = w_g
        poly = Polygon(coords)
        w_x = list(df_['X'])
        w_y = list(df_['Y'])
        w = []
        for i in range(len(w_x)):
            w.append((w_x[i], w_y[i]))
        w_u = np.unique(w)
        df_.loc[:, "new"] = w
        df_.reset_index(inplace = True, drop = True)
        for i in range(len(df_['new'])):
            if Point(df_['new'][i]).within(poly) == False:
                df_ = df_.drop(index = i)
        df_ = df_.drop('new', axis = 1)
 
##        x = df_['X']
##        y = df_['Y']
##        z = df_['Глубина'].apply(lambda x: x*(-1))

        
        fig = plt.figure(figsize = (35, 35))
        fig.subplots_adjust(top=1.1, bottom=-.1)
        ax = fig.add_subplot(111, projection='3d')
##        color_map = plt.get_cmap('spring')
##        scatter_plot = ax.scatter3D(x, y, z, c = df_[t[-1]], cmap = col, alpha=1, s=100)


        x1 = ass['X']
        y1 = ass['Y']
        z1 = ass['h']
        ax.plot(x1, y1, z1, linewidth=3)

        df_d = df_.drop_duplicates(subset=['X', 'Y'])
        X_list = list(df_d['X'])
        Y_list = list(df_d['Y'])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Глубина')
        if list(num444.get().split()):
            plt.xlim(Xmin-50, Xmax+50)
            plt.ylim(Ymin-150, Ymax+150)
        
        ax.text(X_list[0], Y_list[0], 761, "9020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[1], Y_list[1], 767, "9420", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[2], Y_list[2], 773, "9820", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[3], Y_list[3], 786, "20420", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[4], Y_list[4], 771, "1019", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[5], Y_list[5], 776, "2219", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[6], Y_list[6], 773, "3020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[7], Y_list[7], 768, "3120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[8], Y_list[8], 764, "920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[9], Y_list[9], 768, "2120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[10], Y_list[10], 763, "2920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[11], Y_list[11], 768, "9620", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[12], Y_list[12], 767, "9220", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[13], Y_list[13], 785, "20220", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[14], Y_list[14], 771, "9920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[15], Y_list[15], 763, "819", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[16], Y_list[16], 761, "2020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[17], Y_list[17], 761, "9120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[18], Y_list[18], 767, "9320", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[19], Y_list[19], 766, "9520", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[20], Y_list[20], 781, "9720", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[21], Y_list[21], 770, "1521", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[22], Y_list[22], 777, "8421", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[23], Y_list[23], 784, "8521", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[24], Y_list[24], 773, "10021", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[25], Y_list[25], 773, "10221", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[26], Y_list[26], 781, "14821", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[27], Y_list[27], 790, "14921", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[28], Y_list[28], 788, "6021", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[29], Y_list[29], 788, "7621", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[30], Y_list[30], 788, "7821", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[31], Y_list[31], 789, "20120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[32], Y_list[32], 783, "20320", fontsize=8, zdir='z', color='black', fontweight='bold')

        ax.text(X_list[33], Y_list[33], 787, "18424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[34], Y_list[34], 789, "18724", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[35], Y_list[35], 782, "18224", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[36], Y_list[36], 791, "17424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[37], Y_list[37], 787, "18124", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[38], Y_list[38], 791, "17524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[39], Y_list[39], 788, "17624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[40], Y_list[40], 785, "19424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[41], Y_list[41], 797, "19324", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[42], Y_list[42], 784, "19524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[43], Y_list[43], 785, "16524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[44], Y_list[44], 787, "16624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[45], Y_list[45], 780, "16424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[46], Y_list[46], 780, "16324", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[47], Y_list[47], 765, "23424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[48], Y_list[48], 768, "23524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[49], Y_list[49], 766, "23624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[50], Y_list[50], 765, "23724", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[51], Y_list[51], 766, "24124", fontsize=8, zdir='z', color='black', fontweight='bold')
        
        
    
   #     plt.colorbar(scatter_plot)
        FX_ = []
        FX_i = []
        for i in range(len(FX)):
            if list(num444.get().split()):
                if len(at) > 1:
                    coords = W_[i]
                    poly = Polygon(coords)
                    w_x = list(FX[i]['X'])
                    w_y = list(FX[i]['Y'])
                    w = []
                    for k in range(len(w_x)):
                        w.append((w_x[k], w_y[k]))
                    w_u = np.unique(w)
                    FX[i]['new'] = w
                    FX[i].reset_index(inplace = True, drop = True)
                    for k in range(len(w)):
                        if Point(FX[i]['new'][k]).within(poly) == False:
                            FX[i] = FX[i].drop(index = k)
                    FX[i] = FX[i].drop('new', axis = 1)
                    FX_.append(FX[i])
                    VO.append(round(SS_1[i]*len(FX[i]),1))
                else:
                    FX[i] = FX[i].loc[(FX[i]['X'] <= XMAX[i]) & (FX[i]['X'] >= XMIN[i]) & (FX[i]['Y'] <= YMAX[i]) & (FX[i]['Y'] >= YMIN[i])]
                    FX_.append(FX[i])
            else:
                print(i)
                FR = FX[i].copy()
                FX_i.append(FR)
                coords = w_g
                poly = Polygon(coords)
                w_x = list(FX[i]['X'])
                w_y = list(FX[i]['Y'])
                w = []
                for k in range(len(w_x)):
                    w.append((w_x[k], w_y[k]))
                w_u = np.unique(w)
                FX[i]["new"] = w
                FX[i].reset_index(inplace = True, drop = True)
                for k in range(len(FX[i]['new'])):
                    if Point(FX[i]['new'][k]).within(poly) == False:
                        FX[i] = FX[i].drop(index = k)
                FX[i] = FX[i].drop('new', axis = 1)
                FX_.append(FX[i])
        

##        for i in range(len(FX_)):  
##            x2 = FX_[i]['X']
##            y2 = FX_[i]['Y']
##            z2 = FX_[i]['Глубина'].apply(lambda x: x*(-1))
##
##            scatter_plot = ax.scatter3D(x2, y2, z2, c = FX_[i][t_1[-1]], cmap = col, alpha=0.15)
        
        
        #res32.configure(text = "Запасы сырья, м3: %s" % round(S_1*len(X_F_x),1))

        def explode_xy(xy):
            xl=[]
            yl=[]
            for i in range(len(xy)):
                xl.append(xy[i][0])
                yl.append(xy[i][1])
            return xl,yl

        def shoelace_area(x_list,y_list):
            a1,a2=0,0
            x_list.append(x_list[0])
            y_list.append(y_list[0])
            for j in range(len(x_list)-1):
                a1 += x_list[j]*y_list[j+1]
                a2 += y_list[j]*x_list[j+1]
            l = abs(a1-a2)/2
            return l
        if list(num444.get().split()):
            if len(at) > 1:
##                xy_e = explode_xy(w_)
##                S_r = shoelace_area(xy_e[0],xy_e[1])
                S = S_R[-1]  
        else:
            xy_e = explode_xy(w_g)
            S = shoelace_area(xy_e[0],xy_e[1])
            
        #res32.configure(text = "Запасы сырья, м3: %s" % round(S_r*36))
        #res33.configure(text = "Площадь выделенного участка месторождения, м2: %s" % round(S_r, 1))
        if list(num444.get().split()):
            if len(at) > 1:
                res32.configure(text = "Запасы сырья, м3: %s" % round(SS_1[-1]*len(FX_[-1]),1))
                
            else:
                kt = int(np.sqrt((Xmax - Xmin)*(Xmax - Xmin) + (Ymax - Ymin)*(Ymax - Ymin)))
                u = len(FX[-1])
                res32.configure(text = "Мощность разреза, м: %s; Площадь среза, м2: %s " % (int((u*kt/(kt*Len))), int((u*kt/Len))))
        else:
            res32.configure(text = "Запасы сырья, м3: %s" % round(SS_1[0]*len(FX_[0]),1))
            
        if list(num444.get().split()):
            if len(at) > 1:
                res33.configure(text = "Площадь выделенного участка, м2: %s" % round(S, 1))
                columns_1 = ["Площадь участка, м2", "Объем запасов, м3", "Xmin", "Xmax", "Ymin", "Ymax"]
                myTable_1 = PrettyTable()

                myTable_1.add_column(columns_1[0], S_R)
                myTable_1.add_column(columns_1[1], VO)
                myTable_1.add_column(columns_1[2], XMIN)
                myTable_1.add_column(columns_1[3], XMAX)
                myTable_1.add_column(columns_1[4], YMIN)
                myTable_1.add_column(columns_1[5], YMAX)
                print(myTable_1)
                
            else:
                res33.configure(text = "Длина среза, м: %s" % kt)
                columns_2 = ['Мощность разреза, м', 'Площадь разреза, м2', 'Длина разреза, м', 'Коорд начала', 'Коорд конца']
                myTable_2 = PrettyTable()
                myTable_2.add_column(columns_2[0], MO)
                myTable_2.add_column(columns_2[1], MS)
                myTable_2.add_column(columns_2[2], KT)
                myTable_2.add_column(columns_2[3], W_N)
                myTable_2.add_column(columns_2[4], W_F)
                print(myTable_2)
                
        else:
            res33.configure(text = "Площадь выделенного участка, м2: %s" % round(S, 1))
     # задаем координату z для поверхности на основе координат скважин       
        df_dd = df_d[['X', 'Y']]
        df_ddd = pd.read_excel('C:\\west\\Каталог координат устьев скважин - Запад.xlsx')
        df_ddd = df_ddd[['X','Y','Z']]
        x10 = df_ddd['X']
        y10 = df_ddd['Y']
    # задаем координаты сетки
        xi = np.linspace(df_ddd['X'].min(), df_ddd['X'].max(), 100)
        yi = np.linspace(df_ddd['Y'].min(), df_ddd['Y'].max(), 100)
        xii, yii = np.meshgrid(xi, yi)
    # задаем уравнение поверхности z(x,y)
        zi = griddata((x10, y10), df_ddd['Z'], (xii, yii), method='linear')
        ##cubic, nearest вместо linear
    # строим поверхность
        my_cmap = plt.get_cmap('cool')
        ax.plot_surface(xii, yii, zi, cmap = my_cmap, edgecolor ='none', alpha=0.3)
    # корректируем глубину скважин под рельеф поверхности
        points = []
        for i in range(len(df_ddd)):
            points.append([list(df_ddd['X'])[i], list(df_ddd['Y'])[i]])
        values = list(df_ddd['Z'])
        cord_r = []
        for i in range(len(df_dd)):
            cord_r.append(([list(df_dd['X'])[i], list(df_dd['Y'])[i]]))
        Z_s = []
        for i in range(len(cord_r)):
            Z_s.append(griddata(points, values, cord_r[i],  method='linear'))
        Z_s = [i[0] for i in Z_s]
        df_dd['Z_s'] = Z_s
     # корректируем координату Z для диаграммы скважин:
        df_n = pd.merge(df_, df_dd, on=['X', 'Y'], how='left')
        df_n['Z_'] = df_n['Z_s'] - df_n['Глубина']

     # ищем соответствие номеров скважин и их геометрии
        
        print('ttttttttttttttttttttttttttttttttttttttttttttttttttttt')
        print(df_n.groupby('Скважина').agg({'Z_':'min'}))
        print(df_d['Скважина'])

     # строим диаграмму скважин:
        x = df_n['X']
        y = df_n['Y']
        z = df_n['Z_'].apply(lambda x: x*(1))
        color_map = plt.get_cmap('spring')
        scatter_plot = ax.scatter3D(x, y, z, c = df_n[t[-1]], cmap = col, alpha=1, s=100)
        plt.colorbar(scatter_plot)
     # корректируем координату Z в модельном представлении
        FX_m = []
        for i in range(len(FX_)):
            FF = FX[i].copy()
            fx = FX_[i].drop_duplicates(subset=['X', 'Y'])[['X', 'Y']]
            cord_rr = []
            for i in range(len(fx)):
                cord_rr.append(([list(fx['X'])[i], list(fx['Y'])[i]]))
            Z_ss = []
            for i in range(len(cord_rr)):
                Z_ss.append(griddata(points, values, cord_rr[i],  method='linear'))
            Z_ss = [j[0] for j in Z_ss]
        
            fx['Z_s'] = Z_ss
            fx_ = pd.merge(FF, fx, on=['X', 'Y'], how='left')
            fx_['Z_'] = fx_['Z_s'] - fx_['Глубина']
            FX_m.append(fx_)

        for i in range(len(FX_m)):  
            x2 = FX_m[i]['X']
            y2 = FX_m[i]['Y']
            z2 = FX_m[i]['Z_'].apply(lambda x: x*(1))
            scatter_plot = ax.scatter3D(x2, y2, z2, c = FX_m[i][t_1[-1]], cmap = col, alpha=0.1)
             
##        fx_ = pd.DataFrame({'X':xii,"Y":yii,'Z':zii})
##        merged_fx = pd.merge(FX_[0], fx_, on=['X', 'Y'], how='left')
##        merged_fx['Z_'] = merged_fx['Z'] - merged_fx['Глубина']
##
##        x2 = merged_fx['X']
##        y2 = merged_fx['Y']
##        z2 = merged_fx['Z_']
##        tt = merged_fx.columns
##        scatter_plot = ax.scatter3D(x2, y2, z2, c = merged_fx[tt[-3]], cmap = col, alpha=1)

        
##        XYZ = []
##        for ii in range(100):
##            for jj in range(100):
##                XYZ.append([xi[ii][jj], yi[ii][jj], zi[ii][jj]])
##            
##        def in_poly_hull_single(poly, point):
##            hull = ConvexHull(poly)
##            new_hull = ConvexHull(np.concatenate((poly, [point])))
##            return np.array_equal(new_hull.vertices, hull.vertices)
##        dop = [[df_d['X'].min(), df_d['Y'].min(), -36], [df_d['X'].max(), df_d['Y'].min(), -36], [df_d['X'].min(), df_d['Y'].max(),-36], [df_d['X'].max(), df_d['Y'].max(), -36]]
##        poly = XYZ + dop
##        point = [5267365.819696969, 14716054.700000001, -3]
##        print(df_d[:3])
##        zz = []
##        x_ = list(df_d['X'])
##        y_ = list(df_d['Y'])
##        z_ = list(df_d['Z'])
##        for i in range(len(df_d)):
##            k = 0
##            for j in range(200):
##                zz_ = z_[i] + k
##                k+=0.5
##                if Delaunay(poly).find_simplex([x_[i], y_[i], zz_]) >= 0 != True:
##                    zz.append(zz_-0.25)
##                    break
                    
            
##        print(in_poly_hull_single(poly, point))
##        print(Delaunay(poly).find_simplex(point) >= 0)
       
        
        if list(num0.get().split()):
            plt.title('Комбинированная 3d-диаграмма фактического и модельного распределения фактора {}, горизонты: {}'.format(t[-1], depth))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('Комбинированная 3d-диаграмма фактического и модельного распределения фактора {}, горизонты: {}'.format(t[-1], g))
            else:
                plt.title('Комбинированная 3d-диаграмма фактического и модельного распределения фактора {}'.format(t[-1]))
        else:
            plt.title('Комбинированная 3d-диаграмма фактического и модельного распределения фактора {}'.format(t[-1]))

        fig2 = plt.figure(figsize = (35, 35))
        fig2.subplots_adjust(top=1.1, bottom=-.1)
        ax1 = fig2.add_subplot(111, projection='3d')
        for i in range(len(FX_m)):  
            x2 = FX_m[i]['X']
            y2 = FX_m[i]['Y']
            z2 = FX_m[i]['Z_'].apply(lambda x: x*(1))
            scatter_plot = ax1.scatter3D(x2, y2, z2, c = FX_m[i][t_1[-1]], cmap = col, alpha=1)

        plt.colorbar(scatter_plot)
        if list(num0.get().split()):
            plt.title('3d-модель распределения фактора {}, горизонты: {}'.format(t_1[-1], str(depth)))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('3d-модель распределения фактора {}, горизонты: {}'.format(t[-1], g))
            else:
                plt.title('3d-модель распределения фактора {}'.format(t_1[-1]))
        else:
            plt.title('3d-модель распределения фактора {}'.format(t_1[-1]))

        ax1.plot_surface(xii, yii, zi, alpha=0.3)

        fig3 = plt.figure(figsize = (55, 55))
        fig3.subplots_adjust(top=1.1, bottom=-.1)
        ax2 = fig3.add_subplot(111, projection='3d')
        scatter_plot = ax2.scatter3D(x, y, z, c = df_n[t_1[-1]], cmap = col, alpha=1, s=100)

        ax2.text(X_list[0], Y_list[0], 761, "9020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[1], Y_list[1], 767, "9420", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[2], Y_list[2], 773, "9820", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[3], Y_list[3], 786, "20420", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[4], Y_list[4], 771, "1019", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[5], Y_list[5], 776, "2219", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[6], Y_list[6], 773, "3020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[7], Y_list[7], 768, "3120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[8], Y_list[8], 764, "920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[9], Y_list[9], 768, "2120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[10], Y_list[10], 763, "2920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[11], Y_list[11], 768, "9620", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[12], Y_list[12], 767, "9220", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[13], Y_list[13], 785, "20220", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[14], Y_list[14], 771, "9920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[15], Y_list[15], 763, "819", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[16], Y_list[16], 761, "2020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[17], Y_list[17], 761, "9120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[18], Y_list[18], 767, "9320", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[19], Y_list[19], 766, "9520", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[20], Y_list[20], 781, "9720", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[21], Y_list[21], 770, "1521", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[22], Y_list[22], 777, "8421", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[23], Y_list[23], 784, "8521", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[24], Y_list[24], 773, "10021", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[25], Y_list[25], 773, "10221", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[26], Y_list[26], 781, "14821", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[27], Y_list[27], 790, "14921", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[28], Y_list[28], 788, "6021", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[29], Y_list[29], 788, "7621", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[30], Y_list[30], 788, "7821", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[31], Y_list[31], 789, "20120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[32], Y_list[32], 783, "20320", fontsize=8, zdir='z', color='black', fontweight='bold')

        ax2.text(X_list[33], Y_list[33], 787, "18424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[34], Y_list[34], 789, "18724", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[35], Y_list[35], 782, "18224", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[36], Y_list[36], 791, "17424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[37], Y_list[37], 787, "18124", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[38], Y_list[38], 791, "17524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[39], Y_list[39], 788, "17624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[40], Y_list[40], 785, "19424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[41], Y_list[41], 797, "19324", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[42], Y_list[42], 784, "19524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[43], Y_list[43], 785, "16524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[44], Y_list[44], 787, "16624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[45], Y_list[45], 780, "16424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[46], Y_list[46], 780, "16324", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[47], Y_list[47], 765, "23424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[48], Y_list[48], 768, "23524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[49], Y_list[49], 766, "23624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[50], Y_list[50], 765, "23724", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[51], Y_list[51], 766, "24124", fontsize=8, zdir='z', color='black', fontweight='bold')
        
        ax2.plot(x1, y1, z1, linewidth=3)
   
##        df_d = df_d[['X', 'Y']]
##        asss = pd.DataFrame({'X':[ass['X'].max(), ass['X'].max(), ass['X'].min(), ass['X'].min()], 'Y': [ass['Y'].max(), ass['Y'].min(), ass['Y'].min(), ass['Y'].max()]})
##        df_d = pd.concat([df_d, asss], axis = 0)
##        df_d['Z'] = [0.5, 0.5, 1, 3, 0.5, 0, 0, 0, 0, 0, 0.5, 3, -3, 2, 2, 0.5, 3, 0, 0, -4, -5, 0, 0, 0.5, 0.3, 1, 1, 3, 3, 0, 0.5, 5, 6, 0, 0, 0.5, 2]
##        x = df_d['X']
##        y = df_d['Y']
##        xi = np.linspace(df_d['X'].min(), df_d['X'].max(), 100)
##        yi = np.linspace(df_d['Y'].min(), df_d['Y'].max(), 100)
##        xi, yi = np.meshgrid(xi, yi)
##        zi = griddata((x10, y10), df_d['Z'], (xi, yi), method='linear')
        ax2.plot_surface(xii, yii, zi, alpha=0.3)
        
        if list(num0.get().split()):
            plt.title('3d-диаграмма фактического распределения фактора {}, горизонты: {}'.format(t[-1], depth))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('3d-диаграмма фактического распределения фактора {}, горизонты: {}'.format(t[-1], g))
            else:
                plt.title('3d-диаграмма фактического распределения фактора {}'.format(t[-1]))
        else:
            plt.title('3d-диаграмма фактического распределения фактора {}'.format(t[-1]))
        
        plt.show()


        
    else:

        coords = w_g
        poly = Polygon(coords)
        w_x = list(df_h['X'])
        w_y = list(df_h['Y'])
        w = []
        for i in range(len(w_x)):
            w.append((w_x[i], w_y[i]))
        w_u = np.unique(w)
        df_h.loc[:, "new"] = w
        df_h.reset_index(inplace = True, drop = True)
        for i in range(len(df_h['new'])):
            if Point(df_h['new'][i]).within(poly) == False:
                df_h = df_h.drop(index = i)
        df_h = df_h.drop('new', axis = 1)

        
##        x = df_h['X']
##        y = df_h['Y']
##        z = df_h['Глубина'].apply(lambda x: x*(-1))

        x1 = ass['X']
        y1 = ass['Y']
        z1 = ass['h']
        
        df_dh = df_h.drop_duplicates(subset=['X', 'Y'])
        X_list = list(df_dh['X'])
        Y_list = list(df_dh['Y'])

        fig = plt.figure(figsize = (35, 35))
        fig.subplots_adjust(top=1.1, bottom=-.1)
        #ax = plt.axes(projection ="3d")
        ax = fig.add_subplot(111, projection='3d')

##        color_map = plt.get_cmap('spring')
##        scatter_plot = ax.scatter3D(x, y, z, c = df_h['Hor'], cmap = col, alpha=1, s=100)
        
        ax.plot(x1, y1, z1, linewidth=3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Глубина')
        if list(num444.get().split()):
            plt.xlim(Xmin-50, Xmax+50)
            plt.ylim(Ymin-300, Ymax+300)
        ax.text(X_list[0], Y_list[0], 761, "9020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[1], Y_list[1], 767, "9420", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[2], Y_list[2], 773, "9820", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[3], Y_list[3], 786, "20420", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[4], Y_list[4], 771, "1019", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[5], Y_list[5], 776, "2219", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[6], Y_list[6], 773, "3020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[7], Y_list[7], 768, "3120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[8], Y_list[8], 764, "920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[9], Y_list[9], 768, "2120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[10], Y_list[10], 763, "2920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[11], Y_list[11], 768, "9620", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[12], Y_list[12], 767, "9220", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[13], Y_list[13], 785, "20220", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[14], Y_list[14], 771, "9920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[15], Y_list[15], 763, "819", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[16], Y_list[16], 761, "2020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[17], Y_list[17], 761, "9120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[18], Y_list[18], 767, "9320", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[19], Y_list[19], 766, "9520", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[20], Y_list[20], 781, "9720", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[21], Y_list[21], 770, "1521", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[22], Y_list[22], 777, "8421", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[23], Y_list[23], 784, "8521", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[24], Y_list[24], 773, "10021", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[25], Y_list[25], 773, "10221", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[26], Y_list[26], 781, "14821", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[27], Y_list[27], 790, "14921", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[28], Y_list[28], 788, "6021", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[29], Y_list[29], 788, "7621", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[30], Y_list[30], 788, "7821", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[31], Y_list[31], 789, "20120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[32], Y_list[32], 783, "20320", fontsize=8, zdir='z', color='black', fontweight='bold')

        ax.text(X_list[33], Y_list[33], 787, "18424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[34], Y_list[34], 789, "18724", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[35], Y_list[35], 782, "18224", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[36], Y_list[36], 791, "17424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[37], Y_list[37], 787, "18124", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[38], Y_list[38], 791, "17524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[39], Y_list[39], 788, "17624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[40], Y_list[40], 785, "19424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[41], Y_list[41], 797, "19324", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[42], Y_list[42], 784, "19524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[43], Y_list[43], 785, "16524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[44], Y_list[44], 787, "16624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[45], Y_list[45], 780, "16424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[46], Y_list[46], 780, "16324", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[47], Y_list[47], 765, "23424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[48], Y_list[48], 768, "23524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[49], Y_list[49], 766, "23624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[50], Y_list[50], 765, "23724", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax.text(X_list[51], Y_list[51], 766, "24124", fontsize=8, zdir='z', color='black', fontweight='bold')

        
##        plt.colorbar(scatter_plot)
        #plt.colorbar(scatter_plot1)

        FX_ = [] 
        for i in range(len(F)):
            if list(num444.get().split()):
                if len(at) > 1:
                    coords = W_[i]
                    poly = Polygon(coords)
                    w_x = list(F[i]['X'])
                    w_y = list(F[i]['Y'])
                    w = []
                    for k in range(len(w_x)):
                        w.append((w_x[k], w_y[k]))
                    w_u = np.unique(w)
                    F[i]['new'] = w
                    F[i].reset_index(inplace = True, drop = True)
                    for k in range(len(w)):
                        if Point(F[i]['new'][k]).within(poly) == False:
                            F[i] = F[i].drop(index = k)
                    F[i] = F[i].drop('new', axis = 1)
                    FX_.append(F[i])
                    VO.append(round(SS_1[i]*len(F[i]),1))
                else:
                    F[i] = F[i].loc[(F[i]['X'] <= XMAX[i]) & (F[i]['X'] >= XMIN[i]) & (F[i]['Y'] <= YMAX[i]) & (F[i]['Y'] >= YMIN[i])]
                    FX_.append(F[i])
            else:
                print(i)
                coords = w_g
                poly = Polygon(coords)
                w_x = list(F[i]['X'])
                w_y = list(F[i]['Y'])
                w = []
                for k in range(len(w_x)):
                    w.append((w_x[k], w_y[k]))
                w_u = np.unique(w)
                F[i]["new"] = w
                F[i].reset_index(inplace = True, drop = True)
                for k in range(len(F[i]['new'])):
                    if Point(F[i]['new'][k]).within(poly) == False:
                        F[i] = F[i].drop(index = k)
                F[i] = F[i].drop('new', axis = 1)
                FX_.append(F[i])

##        for i in range(len(FX_)):  
##            x2 = FX_[i]['X']
##            y2 = FX_[i]['Y']
##            z2 = FX_[i]['Глубина'].apply(lambda x: x*(-1))
##
##            scatter_plot = ax.scatter3D(x2, y2, z2, c = FX_[i]['Horizont'], cmap = col, alpha=0.15)

        def explode_xy(xy):
            xl=[]
            yl=[]
            for i in range(len(xy)):
                xl.append(xy[i][0])
                yl.append(xy[i][1])
            return xl,yl

        def shoelace_area(x_list,y_list):
            a1,a2=0,0
            x_list.append(x_list[0])
            y_list.append(y_list[0])
            for j in range(len(x_list)-1):
                a1 += x_list[j]*y_list[j+1]
                a2 += y_list[j]*x_list[j+1]
            l = abs(a1-a2)/2
            return l

        if list(num444.get().split()):
            if len(at) > 1:
                S = S_R[-1]  
        else:
            xy_e = explode_xy(w_g)
            S = shoelace_area(xy_e[0],xy_e[1])

##            res32.configure(text = "Запасы сырья, м3: %s" % round(S_r*36)
        
        if list(num444.get().split()):
            if len(at) > 1:
                res32.configure(text = "Запасы сырья, м3: %s" % round(SS_1[-1]*len(FX_[-1]),1))
                
            else:
                kt = int(np.sqrt((Xmax - Xmin)*(Xmax - Xmin) + (Ymax - Ymin)*(Ymax - Ymin)))
                u = len(FX_[-1])
                res32.configure(text = "Мощность разреза, м: %s; Площадь среза, м2: %s; длина среза, м: %s " % (int((u*kt/(kt*Len))), int((u*kt/Len)), kt))
        else:
            res32.configure(text = "Запасы сырья, м3: %s" % round(SS_1[0]*len(FX_[0]),1))

        if list(num444.get().split()):
            if len(at) > 1:
                res33.configure(text = "Площадь выделенного участка, м2: %s" % round(S, 1))
                columns_1 = ["Площадь участка, м2", "Объем запасов, м3", "Xmin", "Xmax", "Ymin", "Ymax"]
                myTable_1 = PrettyTable()

                myTable_1.add_column(columns_1[0], S_R)
                myTable_1.add_column(columns_1[1], VO)
                myTable_1.add_column(columns_1[2], XMIN)
                myTable_1.add_column(columns_1[3], XMAX)
                myTable_1.add_column(columns_1[4], YMIN)
                myTable_1.add_column(columns_1[5], YMAX)
                print(myTable_1)
                
            else:
                res33.configure(text = "Длина среза, м: %s" % kt)
                columns_2 = ['Мощность разреза, м', 'Площадь разреза, м2', 'Длина разреза, м', 'Коорд начала', 'Коорд конца']
                myTable_2 = PrettyTable()
                myTable_2.add_column(columns_2[0], MO)
                myTable_2.add_column(columns_2[1], MS)
                myTable_2.add_column(columns_2[2], KT)
                myTable_2.add_column(columns_2[3], W_N)
                myTable_2.add_column(columns_2[4], W_F)
                print(myTable_2)
                
        else:
            res33.configure(text = "Площадь выделенного участка, м2: %s" % round(S, 1))

         # задаем координату z для поверхности на основе координат скважин       
        df_dd = df_dh[['X', 'Y']]
        
        df_ddd = pd.read_excel('C:\\west\\Каталог координат устьев скважин - Запад.xlsx')
        df_ddd = df_ddd[['X','Y','Z']]
        x10 = df_ddd['X']
        y10 = df_ddd['Y']
    # задаем координаты сетки
        xi = np.linspace(df_ddd['X'].min(), df_ddd['X'].max(), 100)
        yi = np.linspace(df_ddd['Y'].min(), df_ddd['Y'].max(), 100)
        xii, yii = np.meshgrid(xi, yi)
    # задаем уравнение поверхности z(x,y)
        zi = griddata((x10, y10), df_ddd['Z'], (xii, yii), method='linear')
        ##cubic, nearest вместо linear
    # строим поверхность
        my_cmap = plt.get_cmap('cool')
        ax.plot_surface(xii, yii, zi, cmap = my_cmap, edgecolor ='none', alpha=0.3)
    # корректируем глубину скважин под рельеф поверхности
        points = []
        for i in range(len(df_ddd)):
            points.append([list(df_ddd['X'])[i], list(df_ddd['Y'])[i]])
        values = list(df_ddd['Z'])
        cord_r = []
        for i in range(len(df_dd)):
            cord_r.append(([list(df_dd['X'])[i], list(df_dd['Y'])[i]]))
        Z_s = []
        for i in range(len(cord_r)):
            Z_s.append(griddata(points, values, cord_r[i],  method='linear'))
        Z_s = [i[0] for i in Z_s]
        df_dd['Z_s'] = Z_s
     # корректируем координату Z для диаграммы скважин:
        df_n = pd.merge(df_h, df_dd, on=['X', 'Y'], how='left')
        df_n['Z_'] = df_n['Z_s'] - df_n['Глубина']

        print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        print(df_n[:5])

     # ищем соответствие номеров скважин и их геометрии
        
##        print('ttttttttttttttttttttttttttttttttttttttttttttttttttttt')
##        print(df_n.groupby('Скважина').agg({'Z_':'min'}))

     # строим диаграмму скважин:
        x = df_n['X']
        y = df_n['Y']
        z = df_n['Z_'].apply(lambda x: x*(1))
        color_map = plt.get_cmap('spring')
        scatter_plot = ax.scatter3D(x, y, z, c = df_n['Hor'], cmap = col, alpha=1, s=100)
        plt.colorbar(scatter_plot)
     # корректируем координату Z в модельном представлении
        FX_m = []
        for i in range(len(FX_)):
            FF = F[i].copy()
            fx = FX_[i].drop_duplicates(subset=['X', 'Y'])[['X', 'Y']]
            cord_rr = []
            for i in range(len(fx)):
                cord_rr.append(([list(fx['X'])[i], list(fx['Y'])[i]]))
            Z_ss = []
            for i in range(len(cord_rr)):
                Z_ss.append(griddata(points, values, cord_rr[i],  method='linear'))
            Z_ss = [j[0] for j in Z_ss]
        
            fx['Z_s'] = Z_ss
            fx_ = pd.merge(FF, fx, on=['X', 'Y'], how='left')
            fx_['Z_'] = fx_['Z_s'] - fx_['Глубина']
            FX_m.append(fx_)

        for i in range(len(FX_m)):  
            x2 = FX_m[i]['X']
            y2 = FX_m[i]['Y']
            z2 = FX_m[i]['Z_'].apply(lambda x: x*(1))
            scatter_plot = ax.scatter3D(x2, y2, z2, c = FX_m[i]['Horizont'], cmap = col, alpha=0.1)




        if list(num0.get().split()):
            plt.title('Комбинированная 3d-диаграмма фактического и модельного распределения горизонтов: {}'.format(depth))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('Комбинированная 3d-диаграмма фактического и модельного распределения горизонтов: {}'.format(g))
            else:
                plt.title('Комбинированная 3d-диаграмма фактического и модельного распределения горизонтов')
        else:
            plt.title('Комбинированная 3d-диаграмма фактического и модельного распределения горизонтов')

        fig2 = plt.figure(figsize = (35, 35))
        fig2.subplots_adjust(top=1.1, bottom=-.1)
        ax1 = fig2.add_subplot(111, projection='3d')
        for i in range(len(FX_m)):   
            x2 = FX_m[i]['X']
            y2 = FX_m[i]['Y']
            z2 = FX_m[i]['Z_'].apply(lambda x: x*(1))
        
            scatter_plot = ax1.scatter3D(x2, y2, z2, c = FX_m[i]['Horizont'], cmap = col, alpha=1)
        
        plt.colorbar(scatter_plot)
        ax1.plot(x1, y1, z1, linewidth=3)
        
        if list(num0.get().split()):
            plt.title('3d-модель распределения фактора горизонтов: {}'.format(str(depth)))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('3d-модель распределения горизонтов: {}'.format(g))
            else:
                plt.title('3d-модель распределения горизонтов')
        else:
            plt.title('3d-модель распределения горизонтов')

        fig3 = plt.figure(figsize = (35, 35))
        fig3.subplots_adjust(top=1.1, bottom=-.1)
        ax2 = fig3.add_subplot(111, projection='3d')
        scatter_plot = ax2.scatter3D(x, y, z, c = df_n['Hor'], cmap = col, alpha=1, s=100)

        ax2.text(X_list[0], Y_list[0], 761, "9020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[1], Y_list[1], 767, "9420", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[2], Y_list[2], 773, "9820", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[3], Y_list[3], 786, "20420", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[4], Y_list[4], 771, "1019", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[5], Y_list[5], 776, "2219", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[6], Y_list[6], 773, "3020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[7], Y_list[7], 768, "3120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[8], Y_list[8], 764, "920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[9], Y_list[9], 768, "2120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[10], Y_list[10], 763, "2920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[11], Y_list[11], 768, "9620", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[12], Y_list[12], 767, "9220", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[13], Y_list[13], 785, "20220", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[14], Y_list[14], 771, "9920", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[15], Y_list[15], 763, "819", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[16], Y_list[16], 761, "2020", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[17], Y_list[17], 761, "9120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[18], Y_list[18], 767, "9320", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[19], Y_list[19], 766, "9520", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[20], Y_list[20], 781, "9720", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[21], Y_list[21], 770, "1521", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[22], Y_list[22], 777, "8421", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[23], Y_list[23], 784, "8521", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[24], Y_list[24], 773, "10021", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[25], Y_list[25], 773, "10221", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[26], Y_list[26], 781, "14821", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[27], Y_list[27], 790, "14921", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[28], Y_list[28], 788, "6021", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[29], Y_list[29], 788, "7621", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[30], Y_list[30], 788, "7821", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[31], Y_list[31], 789, "20120", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[32], Y_list[32], 783, "20320", fontsize=8, zdir='z', color='black', fontweight='bold')

        ax2.text(X_list[33], Y_list[33], 787, "18424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[34], Y_list[34], 789, "18724", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[35], Y_list[35], 782, "18224", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[36], Y_list[36], 791, "17424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[37], Y_list[37], 787, "18124", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[38], Y_list[38], 791, "17524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[39], Y_list[39], 788, "17624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[40], Y_list[40], 785, "19424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[41], Y_list[41], 797, "19324", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[42], Y_list[42], 784, "19524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[43], Y_list[43], 785, "16524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[44], Y_list[44], 787, "16624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[45], Y_list[45], 780, "16424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[46], Y_list[46], 780, "16324", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[47], Y_list[47], 765, "23424", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[48], Y_list[48], 768, "23524", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[49], Y_list[49], 766, "23624", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[50], Y_list[50], 765, "23724", fontsize=8, zdir='z', color='black', fontweight='bold')
        ax2.text(X_list[51], Y_list[51], 766, "24124", fontsize=8, zdir='z', color='black', fontweight='bold')

        ax2.plot(x1, y1, z1, linewidth=3)        
        plt.colorbar(scatter_plot, shrink=0.6)
        if list(num0.get().split()):
            plt.title('3d-диаграмма фактического распределения горизонтов: {}'.format(depth))
        elif list(num111.get().split()):
            if len(m) > 1:
                plt.title('3d-диаграмма фактического распределения горизонтов: {}'.format(g))
            else:
                plt.title('3d-диаграмма фактического распределения горизонтов')
        else:
            plt.title('3d-диаграмма фактического распределения горизонтов')

                   
        plt.show()


    df = df_rr
##    except:
##        resf.configure(text = "Ошибка загрузки данных")
##    finally:
##        df = df_rr        
           
def dim():
    import cv2
    import imageio
    global aa
    global bb


    if list(num333.get().split()):
        print('velryhn')
        def sd(x):
            x1 = []
            for i in x:
                i = str(i)
                x1.append(i.replace(",", "."))
            return x1
        t = list(df.columns)
        X_x = list(df[t[14]])
        X_x = sd(X_x)
        X_x = [float(item) for item in X_x]
        #a = [x for x in a if str(x) != 'nan']
        Y_y = list(df[t[15]])
        Y_y = sd(Y_y)
        Y_y = [float(item) for item in Y_y]

        
        # load transparent image
        img = cv2.imread('C:\\west\\west_n.png', cv2.IMREAD_UNCHANGED)
        hh, ww = img.shape[:2]

        # draw colored line as opaque
        result = img.copy()
        c = 0
        aa = []
        bb = []
        for i in range(len(X_x)):
            y_s = int(round((Y_y[c] - 14715755.24)/1.426, 0))
            y_e = int(round((Y_y[c+1] - 14715755.24)/1.426, 0))
            x_s = int(round((X_x[c] - 5267188.06)/1.41, 0))
            x_e = int(round((X_x[c+1] - 5267188.06)/1.41, 0))
            if (c==0) or c%2 == 0:
                cv2.line(result, (y_s, x_s), (y_e, x_e), (200, 100, 0, 500), 2)
            
                aa.append((y_s, x_s))
                bb.append((y_e, x_e))
            print(aa)
            print(bb)
            c+=1
            if X_x[c+1] == 0:
                break
        print(aa)
        print(bb)

        
        
        cv2.imshow("RESULT", result)
        cv2.waitKey(q)
       # cv2.destroyAllWindows()
        

    else:
        
        class DrawLineWidget(object):
        
            def __init__(self):
                self.original_image = imageio.imread('C://west//west_n.png')
                self.clone = self.original_image.copy()

                cv2.namedWindow('image')
                cv2.setMouseCallback('image', self.extract_coordinates)

                    # List to store start/end points
                self.image_coordinates = []

            def extract_coordinates(self, event, x, y, flags, parameters):
                # Record starting (x,y) coordinates on left mouse button click
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.image_coordinates = [(x,y)]

            # Record ending (x,y) coordintes on left mouse bottom release
                elif event == cv2.EVENT_LBUTTONUP:
                    self.image_coordinates.append((x,y))
                    aa.append(self.image_coordinates[0])
                    bb.append(self.image_coordinates[1])
                    print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

                # Draw line
                    cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (12,150,5), 3)
                    cv2.imshow("image", self.clone) 

            # Clear drawing boxes on right mouse button click
                elif event == cv2.EVENT_RBUTTONDOWN:
                    self.clone = self.original_image.copy()

                
            def show_image(self):
                return self.clone

               
            
        if __name__ == '__main__':
            draw_line_widget = DrawLineWidget()
            aa = []
            bb = []
            t = 0
            while True:
                cv2.imshow('image', draw_line_widget.show_image())
                key = cv2.waitKey(1)

            # Close program with keyboard 'q'
                if key == ord('q'):
                    
                    
                 #   cv2.destroyAllWindows()
                 #   exit(1)
                    break
            print(aa)
    def dx():
        try:
            
            global df
                    
        ##            Xmin = scale_widget_1.get() 
        ##            Xmax = scale_widget_2.get() 
        ##            Ymin = scale_widget_3.get() 
        ##            Ymax = scale_widget_4.get()

            
            at_ = [at[0], at[1], at[2], at[3]]
            bt_ = [bt[0], bt[1], bt[2], bt[3]]
            ab = at_ + bt_
            x_ = []
            y_ = []
            for i in range(len(ab)):
                x_.append(5267188.06 + 1.41*ab[i][1])
                y_.append(14715755.24 + 1.426*ab[i][0])
            Xmin = min(x_) 
            Xmax = max(x_)
            Ymin = min(y_) 
            Ymax = max(y_)

                    
            Xt = 5267188.06 + 1.41*bt[-1][1]
            Yt = 14715755.24 + 1.426*bt[-1][0]

            print(Xmin)
            print(Xmax)
            print(Ymin)
            print(Ymax)
            print(Xt)
            print(Yt)
            res30.configure(text = "Xmin-Xmax-Ymin-Ymax: %s - %s - %s - %s " % (round(Xmin,1), round(Xmax,1), round(Ymin,1), round(Ymax,1)))
            res31.configure(text = "Координаты точки X-Y: %s - %s" % (round(Xt,1), round(Yt,1)))     
                    
            def sd(x):
                x1 = []
                for i in x:
                    i = str(i)
                    x1.append(i.replace(",", "."))
                return x1
            t = list(df.columns)
                    
            a = list(df[t[0]])
            a = sd(a)
            a = [float(item) for item in a]
            #a = [x for x in a if str(x) != 'nan']
            b = list(df[t[1]])
            b = sd(b)
            b = [float(item) for item in b]
            c = list(df[t[2]])
            c = sd(c)
            c = [float(item) for item in c]
            d = list(df[t[3]])
            d = sd(d)
            d = [float(item) for item in d]
            e = list(df[t[4]])
            e = sd(e)
            e = [float(item) for item in e]
            f = list(df[t[5]])
            f = sd(f)
            f = [float(item) for item in f]
            g = list(df[t[6]])
            g = sd(g)
            g = [float(item) for item in g]
            xx = list(df[t[7]])
            xx = sd(xx)
            xx = [float(item) for item in xx]
            yy = list(df[t[8]])
            yy = sd(yy)
            yy = [float(item) for item in yy]

            df_r = df.copy()
            df = pd.DataFrame({t[0]:a, t[1]:b, t[2]:c, t[3]:d, t[4]:e, t[5]:f, t[6]:g, t[7]:xx, t[8]:yy})
                        

            one = list(num1.get().split())
            aa = [float(item) for item in one]
            two = list(num2.get().split())
            bb = [float(item) for item in two]
            three = list(num3.get().split())
            cc = [float(item) for item in three]
            four = list(num4.get().split())
            dd = [float(item) for item in four]
            five = list(num10.get().split())
            ee = [float(item) for item in five]
            six = list(num11.get().split())
            ff = [float(item) for item in six]
            zer = list(num0.get().split())
            zz = [float(item) for item in zer]
                ##    xxx = list(num333.get().split())
                ##    cross = [float(item) for item in xxx]

                ##    dfx = pd.DataFrame({t[0]:zz, t[1]:aa, t[2]:bb, t[3]:cc, t[4]:dd, t[5]:ee, t[6]:ff, t[7]:Xt, t[8]:Yt})
                ##    dx = dfx.copy()
            dff = df.copy()

            dfv = df.query('Электропроводность > 0')
            dfv['Глубина'].corr(dfv['Электропроводность'])
            def mop(values_x,a,b):
                return a * values_x + b 
            values_x = dfv['Глубина']
            values_y = dfv['Электропроводность']
            args, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Электропроводность'].isna(), 'Электропроводность'] = df.loc[df['Электропроводность'].isna(), 'Глубина']*args[0]+args[1]

            dfv = df.query('Монтмориллонит > 0')
            def mop(values_x,a,b):
                return a * values_x + b 
            values_x = dfv['Глубина']
            values_y = dfv['Монтмориллонит']
            args_1, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Монтмориллонит'].isna(), 'Монтмориллонит'] = df.loc[df['Монтмориллонит'].isna(), 'Глубина']*args_1[0]+args_1[1]

            dfv = df.query('Песок > 0')
            def mop(values_x,a,b):
                return a * values_x + b 
            values_x = dfv['Монтмориллонит']
            values_y = dfv['Песок']
            args_2, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Песок'].isna(), 'Песок'] = df.loc[df['Песок'].isna(), 'Монтмориллонит']*args_2[0]+args_2[1]

            dfv = df.query('КОЕ > 0')
            def mop(values_x,a,b):
                return a * values_x + b 
            values_x = dfv['Монтмориллонит']
            values_y = dfv['КОЕ']
            args_3, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['КОЕ'].isna(), 'КОЕ'] = df.loc[df['КОЕ'].isna(), 'Монтмориллонит']*args_3[0]+args_3[1]

            dfv = df.query('Влажность > 0')
            dfv['Монтмориллонит'].corr(dfv['Влажность'])
            def mop(values_x,a,b,c,d):
                return a * values_x + b 
            values_x = dfv['Монтмориллонит']
            values_y = dfv['Влажность']
            args, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Влажность'].isna(), 'Влажность'] = df.loc[df['Влажность'].isna(), 'Монтмориллонит']*args[0]+args[1]
                    
            dfv = df.query('Индекс > 0')
            dfv['Монтмориллонит'].corr(dfv['Индекс'])
            def mop(values_x,a,b,c,d):
                return a * values_x + b 
            values_x = dfv['Монтмориллонит']
            values_y = dfv['Индекс']
            args, covar = curve_fit(mop, values_x, values_y)
            df.loc[df['Индекс'].isna(), 'Индекс'] = df.loc[dff['Индекс'].isna(), 'Монтмориллонит']*args[0]+args[1]

            df = df.loc[(df['X'] >= Xmin) & (df['X'] <= Xmax) & (df['Y'] >= Ymin) & (df['Y'] <= Ymax)]
            tt = t.copy()
            dfg = df.copy()

            t = ['Глубина', 'X', 'Y', 'Монтмориллонит']
            df = df[t]

            if list(num111.get().split()):
                seven = list(num111.get().split())
                m = [float(item) for item in seven]
                for i in range(1,8):
                    if i == m[0]:
                        df = df.drop(['Монтмориллонит'], axis = 1)
                        hh = tt[i-1]
                        df = pd.concat([df, dfg[[hh]]], axis = 1)
                        print(df)
                    
            f = list(df.columns)          
            y = df[f[-1]]
            X = df.drop([f[-1]], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
            scaler = StandardScaler()
            X_train_st = scaler.fit_transform(X_train)
            X_test_st = scaler.transform(X_test)

            models = [
                [Lasso(), 'Линейная регрессия Lasso'],
                [Ridge(), 'Линейная регрессия Ridge'],
                [RandomForestRegressor(n_estimators = 200, random_state = 0), 'Случайный лес'],
                [GradientBoostingRegressor(n_estimators = 200, random_state = 0), 'Градиентный бустинг'],
                [DecisionTreeRegressor(random_state = 0), 'Дерево решений']
                ]

            def metrics(y_true, y_pred, title):
                print(title)
                print('MAE: {:.2f}'.format(mean_absolute_error(y_true,y_pred)))
                print('MSE: {:.2f}'.format(mean_squared_error(y_true,y_pred)))
                print('R2: {:.2f}'.format(r2_score(y_true,y_pred)))

            def prediction(mod, X_train, y_train, X_test, y_test, name):
                model = mod
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics(y_test, y_pred, name)

            p = []
            w = []
            for i in models:
                model = i[0]
                model.fit(X_train_st, y_train)
                y_pred = model.predict(X_test_st)
                p.append(r2_score(y_test,y_pred))
                w.append((mean_absolute_error(y_test, y_pred)+mean_squared_error(y_test, y_pred))/2)
            if max(p) == p[0]:
                print('Максимальная метрика R2-CORE у регрессии LASSO: ', p[0])
            elif max(p) == p[1]:
                print('Максимальная метрика R2-CORE у регрессии Ridge: ', p[1])
            elif max(p) == p[2]:
                print('Максимальная метрика R2-CORE у регрессии RandomForest: ', p[2])
            elif max(p) == p[3]:
                print('Максимальная метрика R2-CORE у регрессии GradientBoosting: ', p[3])  
            elif max(p) == p[4]:
                print('Максимальная метрика R2-CORE у регрессии TreeDecision: ', p[4])      
            if min(w) == w[0]:
                print('Минимальная средняя метрика у регрессии LASSO: ', w[0])
            elif min(w) == w[1]:
                print('Минимальная средняя метрика у регрессии Ridge: ', w[1])
            elif min(w) == w[2]:
                print('Минимальная средняя метрика у регрессии RandomForest: ', w[2])
            elif min(w) == w[3]:
                print('Минимальная средняя метрика у : регрессии GradientBoosting', w[3])
            elif min(w) == w[4]:
                print('Минимальная средняя метрика у : регрессии TreeDecision', w[4])

            Глубина_p = [0+1*i for i in range(36)]
            X_p = Xt
            Y_p = Yt
            X_P = [X_p for i in range(len(Глубина_p))]
            Y_P = [Y_p for i in range(len(Глубина_p))]
            X_PP = pd.DataFrame({'Глубина':Глубина_p, 'X':X_P, 'Y':Y_P})
            X_P_st = scaler.transform(X_PP)
            for i in range(len(p)):
                if p[i] == max(p):
                    model = models[i][0]
                        #model.fit(X_train_st, y_train)
                    y_pred_p = model.predict(X_P_st)
            X_PP[f[-1]] = y_pred_p
            if list(num444.get().split()):
                q = list(num444.get().split())
                Q = [float(item) for item in q]
                X_t = pd.DataFrame({'Глубина':Q[0], 'X':X_P, 'Y':Y_P})
                X_t_st = scaler.transform(X_t)
                    
                for i in range(len(p)):
                    if p[i] == max(p):
                        model = models[i][0]
                            # model.fit(X_train_st, y_train)
                        y_pred_t = model.predict(X_t_st)
                        res28.configure(text = "%s : %s" % (f[-1], round(y_pred_t[0],1)))                

            colorlist = ["darkorange", "gold", "lawngreen", "lightseagreen"]
            newcmp = LinearSegmentedColormap.from_list("testCmap", colors=colorlist, N=256)

            x = X_PP['X']
            y = X_PP['Y']
            z = X_PP['Глубина']

            fig = plt.figure(figsize = (12, 7))
                #ax = plt.axes(projection ="3d")
            ax = fig.add_subplot(111, projection='3d')

            color_map = plt.get_cmap('spring')
            scatter_plot = ax.scatter3D(x, y, z, c = X_PP[f[-1]], cmap = 'coolwarm')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Глубина')
            plt.colorbar(scatter_plot)
            plt.title('3d-модель точечного распределения фактора {}'.format(f[-1]))
            plt.show()
                    
            if (Xt > Xmax) or (Xt < Xmin) or (Yt > Ymax) or (Yt < Ymin):
                resf.configure(text = "Выбранная точка вне модельного поля")
            else:
                resf.configure(text = "  ")

            df = df_r

        except:
            resf.configure(text = "Ошибка ввода. Повторите загрузку данных")
        finally:
            df = df_r

  


button1 = tk.Button(win, text="Browse a File", width=20, command=File_dialog)
button1.grid(row = 0, column = 9 )

button2 = tk.Button(win, text="Load File", width=20, command=Load_excel_data)
button2.grid(row = 4, column = 9 )
    
button8 = tk.Button(win, text = "Распределение по горизонтам", bg="black", fg="white", width=25, command = ddf)
button8.grid(row = 14, column = 9 )

button7 = tk.Button(win, text = "image", bg="black", fg="white", width=20, command = dim)
button7.grid(row = 25, column = 9 )


win.mainloop()












