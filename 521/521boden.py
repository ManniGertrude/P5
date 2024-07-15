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

path = "/home/riafi/Desktop/Code/Fortgeschrittenenpraktika/521"

for filename in os.listdir(path):
    if filename.endswith("bodenprobe.txt"):
        filepath = os.path.join(path, filename)
        with open(filepath, "r") as file:
            contents = np.loadtxt(file, dtype=str)
            contents = contents.astype(np.float64)
            array1 = np.split(contents, 2, axis=1)

            kanal = np.ravel(array1[0], order='C')[:-1]
            count = np.ravel(array1[1], order='C')[:-1]
            err = np.sqrt(np.abs(count)) +0.00001

            fig,ax = plt.subplots()
            
            plt.errorbar(kanal, count, yerr=err, color = 'black',linestyle ='none', marker ='.', markersize =1 , zorder = 2)
            plt.grid(zorder =1)
                
            ax.set(ylabel='Anzahl Ansprecher ', xlabel='Kanalnummer', title = 'Spektrum der Bodenprobe')
            ax.legend()

            fig.savefig(f"{filename}_einzeln.pdf")
            plt.show



            data = np.loadtxt("LangzeitUntergrund.txt", dtype=str)
            data = data.astype(np.float64)
            array = np.split(data, 2, axis=1)

            kanal = np.ravel(array1[0], order='C')[:-1]
            c = np.ravel(array1[1], order='C')[:-1]
            c_unter  = np.ravel(array[1], order='C')[:-1]
            
            count = c - c_unter
            err1 = np.sqrt(np.abs(count)) +0.0001
            #filter = [ 100,410,800, 2100,2600,3600,4000, 5000,7700, 8500,9900,10300,12500,16000]
            #farbedaten =['indianred', 'coral', 'gold', 'yellowgreen', 'skyblue', 'cornflowerblue', 'darkorchid']
            #farbekurve = ['firebrick', 'orangered', 'darkgoldenrod','gold','olive','lightgreen','seagreen' ,'steelblue', 'navy', 'indigo', 'darkorchid', 'orchid', 'palevioletred']
            #p0 = [[100000000,375,10 ,100000,-100],[10,1100,10,0.01,-100],[10000,2250,1,-0.001,10],[10,3500,10,300,10],[100000,4500, 100,0.01,100],[10,5500,100,0.001,100],[7000000,6200,10,0.007,10], [1000,12000,5000,0.001,100]]
            #a = []
            #b = []
            fig,ax = plt.subplots()
           
                #print (f'{i+1} & {"{:.2f}".format(popt[0])} $\pm$ {"{:.2f}".format(perr[0])} & {"{:.2f}".format(popt[1])} $\pm$ {"{:.2f}".format(perr[1])} & {"{:.3f}".format(popt[2])} $\pm$ {"{:.3f}".format(perr[2])} & {"{:.3f}".format(popt[3])} $\pm$ {"{:.3f}".format(perr[3])} & {"{:.3f}".format(popt[4])} $\pm$ {"{:.3f}".format(perr[4])} & $R^2$ {"{:.3f}".format(rsquaredfil)}')
                #plt.plot(kanal1, gaus(kanal1, *popt), color = farbekurve[i],linestyle = '-',linewidth = 2, label = f'Gaußkurve {i+1}', zorder =3)
            plt.errorbar(kanal[np.where((count>-200))], count[np.where((count>-200))], yerr=err1[np.where((count>-200))], color = 'black',linestyle ='none', marker ='.', markersize =1 , zorder = 2)
            plt.grid(zorder =1)
                
            ax.set(ylabel='Anzahl Ansprecher ', xlabel='Kanalnummer', title = 'Spektrum der Bodenprobe ohne Untergrund')
            ax.legend()

            fig.savefig(f"{filename}_ohne_untergrund.pdf")
            plt.show


            E = (9.366075984686372*kanal - 1.7914300902437184)*10**(-2)

            fig,ax = plt.subplots()
           
                #print (f'{i+1} & {"{:.2f}".format(popt[0])} $\pm$ {"{:.2f}".format(perr[0])} & {"{:.2f}".format(popt[1])} $\pm$ {"{:.2f}".format(perr[1])} & {"{:.3f}".format(popt[2])} $\pm$ {"{:.3f}".format(perr[2])} & {"{:.3f}".format(popt[3])} $\pm$ {"{:.3f}".format(perr[3])} & {"{:.3f}".format(popt[4])} $\pm$ {"{:.3f}".format(perr[4])} & $R^2$ {"{:.3f}".format(rsquaredfil)}')
                #plt.plot(kanal1, gaus(kanal1, *popt), color = farbekurve[i],linestyle = '-',linewidth = 2, label = f'Gaußkurve {i+1}', zorder =3)
            plt.errorbar(E[np.where((count>-200))], count[np.where((count>-200))], yerr=err1[np.where((count>-200))], color = 'black',linestyle ='none', marker ='.', markersize =1 , zorder = 2)
            plt.grid(zorder =1)
                
            ax.set(ylabel='Anzahl Ansprecher ', xlabel='Energie E /keV', title = 'Spektrum der Bodenprobe ohne Untergrund')
            ax.legend()

            fig.savefig(f"{filename}_energie_count.pdf")
            plt.show


            #['firebrick', 'orangered', 'darkgoldenrod','gold','olive','lightgreen','seagreen' ,'steelblue', 'navy', 'indigo', 'darkorchid', 'orchid', 'palevioletred']
            
           

            
            




            
            




            
