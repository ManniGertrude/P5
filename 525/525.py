import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import csv
import os
import pandas as pd
from sklearn.metrics import r2_score
import os

def g(x, a1,m1,o1):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    return g1

def gaus0(x, a1, m1, o1,a2,m2,o2):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g2 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    return g1 + g2  

def gaus(x, a1, m1, o1,a2,m2,o2, d,n):
    g2= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g3 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    lin = d*x + n
    return g2 + g3 + lin

def gaus1(x, a1, m1, o1,a2,m2,o2,a3,m3,o3, d, n):
    g2= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g3 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    g1 = a3/(o3*(2*np.pi)**(1/2))*np.exp(-(x-m3)**2/(2*o3)**2)
    lin = d*x + n
    return g2 + g3 + g1 +lin

def gaus4(x, a1, m1, o1,a2,m2,o2,a3,m3,o3,a4,m4,o4, d,n):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g2 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    g3 = a3/(o3*(2*np.pi)**(1/2))*np.exp(-(x-m3)**2/(2*o3)**2)
    g4= a4/(o4*(2*np.pi)**(1/2))*np.exp(-(x-m4)**2/(2*o4)**2) 
    lin = d*x + n
    return g1+g2+g3+g4+ lin

def gaus8(x, a1, m1, o1,a2,m2,o2,a3,m3,o3,a4,m4,o4,a5,m5,o5,a6,m6,o6,a7,m7,o7,a8,m8,o8, d,n):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g2 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    g3 = a3/(o3*(2*np.pi)**(1/2))*np.exp(-(x-m3)**2/(2*o3)**2)
    g4= a4/(o4*(2*np.pi)**(1/2))*np.exp(-(x-m4)**2/(2*o4)**2) 
    g5= a5/(o5*(2*np.pi)**(1/2))*np.exp(-(x-m5)**2/(2*o5)**2)
    g6 = a6/(o6*(2*np.pi)**(1/2))*np.exp(-(x-m6)**2/(2*o6)**2)
    g7= a7/(o7*(2*np.pi)**(1/2))*np.exp(-(x-m7)**2/(2*o7)**2) 
    g8= a8/(o8*(2*np.pi)**(1/2))*np.exp(-(x-m8)**2/(2*o8)**2)
    lin = d*x + n
    return g1+g2+g3+g4+g5+g6+g7+g8+ lin

def gaus9(x, a1, m1, o1,a2,m2,o2,a3,m3,o3,a4,m4,o4,a5,m5,o5,a6,m6,o6,a7,m7,o7,a8,m8,o8,a9,m9,o9, d,n):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g2 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    g3 = a3/(o3*(2*np.pi)**(1/2))*np.exp(-(x-m3)**2/(2*o3)**2)
    g4= a4/(o4*(2*np.pi)**(1/2))*np.exp(-(x-m4)**2/(2*o4)**2) 
    g5= a5/(o5*(2*np.pi)**(1/2))*np.exp(-(x-m5)**2/(2*o5)**2)
    g6 = a6/(o6*(2*np.pi)**(1/2))*np.exp(-(x-m6)**2/(2*o6)**2)
    g7= a7/(o7*(2*np.pi)**(1/2))*np.exp(-(x-m7)**2/(2*o7)**2) 
    g8= a8/(o8*(2*np.pi)**(1/2))*np.exp(-(x-m8)**2/(2*o8)**2)
    g9 = a9/(o9*(2*np.pi)**(1/2))*np.exp(-(x-m9)**2/(2*o9)**2)
    lin = d*x + n
    return g1+g2+g3+g4+g5+g6+g7+g8+g9+ lin

def gausnolin(x, a1, m1, o1,a2,m2,o2,a3,m3,o3,a4,m4,o4,a5,m5,o5,a6,m6,o6,a7,m7,o7,a8,m8,o8):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g2 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    g3 = a3/(o3*(2*np.pi)**(1/2))*np.exp(-(x-m3)**2/(2*o3)**2)
    g4= a4/(o4*(2*np.pi)**(1/2))*np.exp(-(x-m4)**2/(2*o4)**2) 
    g5= a5/(o5*(2*np.pi)**(1/2))*np.exp(-(x-m5)**2/(2*o5)**2)
    g6 = a6/(o6*(2*np.pi)**(1/2))*np.exp(-(x-m6)**2/(2*o6)**2)
    g7= a7/(o7*(2*np.pi)**(1/2))*np.exp(-(x-m7)**2/(2*o7)**2) 
    g8= a8/(o8*(2*np.pi)**(1/2))*np.exp(-(x-m8)**2/(2*o8)**2)

    return g1+g2+g3+g4+g5+g6+g7+g8


path = "/home/riafi/Desktop/Code/Fortgeschrittenenpraktika/525/nurplot"

for filename in os.listdir(path):
    if filename.endswith(".txt"):
        filepath = os.path.join(path, filename)
        with open(filepath, "r") as file:
            contents = np.loadtxt(file, dtype=str)
            contents = contents.astype(np.float64)
            array1 = np.split(contents, 2, axis=1)

            kanal = np.ravel(array1[0], order='C')
            count = np.ravel(array1[1], order='C')

            

            fig, ax=plt.subplots()
            
            p0=np.asarray([50000, 1800, 1000 ,1000, 1500, 100,10,2200,100 ,10000, 7000 ,200, 1,1])
            
            popt, pcov = curve_fit(gaus4, kanal, count, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            print (f'Parameter for {[filename]}', popt, perr)
            rsquaredfil = r2_score(count,gaus4(kanal, *popt))
            print (f'$R^2$ for {[filename]}', rsquaredfil)
            plt.plot(kanal, count, color='cornflowerblue',linestyle ='none', marker ='.', markersize = 2 ,label='Datenpunkte')
            plt.plot(kanal, gaus4(kanal, *popt), color = 'midnightblue',linestyle = '-', label =f'Anpassungskurve')
            plt.grid()

            ax.set(ylabel='Anzahl Ansprecher ', xlabel='Kanalnummer ')
            ax.legend()

            fig.savefig(f"{[filename]}mitplot.pdf")
            plt.show
            




#Natrium links
na1 = np.loadtxt("Na Messung 1 links.txt", dtype=str)
na1 = np.char.replace(na1, ',', '.')
na1 = na1.astype(np.float64)
array = np.split(na1, 3, axis=1)

kanal = np.ravel(array[0], order='C')
count_na_links = np.ravel(array[2], order='C')

fig, ax=plt.subplots()


plt.plot(kanal, count_na_links, color='cornflowerblue',linestyle ='none', marker ='.' , markersize = 2,label='Datenpunkte')
#plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, marker='o', label="Data")

ax.set(ylabel='Anzahl Ansprecher ', xlabel='Kanalnummer ')
ax.legend()

fig.savefig("nalinks.pdf")
plt.show



pathplot = "/home/riafi/Desktop/Code/Fortgeschrittenenpraktika/525/plotundfit"

for filename in os.listdir(pathplot):
    if filename.endswith(".txt"):
        filepath = os.path.join(pathplot, filename)
        with open(filepath, "r") as file:
            contents = np.loadtxt(file, dtype=str)
            contents = contents.astype(np.float64)
            array1 = np.split(contents, 2, axis=1)

            kanal = np.ravel(array1[0], order='C')
            count = np.ravel(array1[1], order='C')

            

            p0=np.asarray([5000000,500,100,1000000, 1000,100,10000,5000,1000,  0,0 ])

            popt, pcov = curve_fit(gaus1, kanal, count, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            print (f'Parameter for {[filename]}', popt, perr)
            rsquaredfil = r2_score(count,gaus1(kanal, *popt))
            print (f'$R^2$ for {[filename]}', rsquaredfil)
        
        
            fig, ax=plt.subplots()

            plt.grid()
            plt.plot(kanal, count, color='cornflowerblue',linestyle ='none', marker ='.', markersize = 2 ,label='Datenpunkte')
            plt.plot(kanal, gaus1(kanal, *popt), color = 'midnightblue',linestyle ='-', label =f'Anpassungskurve')
            ax.set(ylabel='Anzahl Ansprecher ', xlabel='Kanalnummer ')
            ax.legend()

            fig.savefig(f"{[filename]}mitplott.pdf")
            plt.show


            

