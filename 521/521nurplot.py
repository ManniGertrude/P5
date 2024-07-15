import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import r2_score


path = "/home/riafi/Desktop/Code/Fortgeschrittenenpraktika/521"
#einlesen der gemessenen daten
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        filepath = os.path.join(path, filename)
        with open(filepath, "r") as file:
            contents = np.loadtxt(file, dtype=str)
            contents = contents.astype(np.float64)
            array1 = np.split(contents, 2, axis=1)

            kanal = np.ravel(array1[0], order='C')[:-1]
            count = np.ravel(array1[1], order='C')[:-1]
            err = np.sqrt(np.abs(count)) + 0.00001
            fig, ax=plt.subplots()
            
            #graphische darstellungen des gemessenen rohspektrums
            plt.errorbar(kanal, count, yerr = err, color='cornflowerblue',linestyle ='none', marker ='.', markersize = 2 ,label='Datenpunkte')
            plt.grid()

            ax.set(ylabel='Anzahl Ansprecher ', xlabel='Kanalnummer ')
            ax.legend()

            fig.savefig(f"{filename}.pdf")
            plt.show