import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import os
import pandas as pd
from sklearn.metrics import r2_score
import os

path = "/home/riafi/Desktop/Code/Fortgeschrittenenpraktika/521"




#defintion der zur anpassung verwendeten gauskurve           
def gaus(x,a,m,o,d,n ):
    return a/(o*(2*np.pi)**(1/2))*np.exp(-(x-m)**2/(2*o)**2)  + d*x + n


def relativ(x, a,b):
    return (a + b*np.log(x))/x



#einlesen der daten des spektrum
for filename in os.listdir(path):
    if filename.endswith("eu_halbleiter.txt"):
        filepath = os.path.join(path, filename)
        with open(filepath, "r") as file:
            contents = np.loadtxt(file, dtype=str)
            contents = contents.astype(np.float64)
            array1 = np.split(contents, 2, axis=1)

            kanal = np.ravel(array1[0], order='C')[:-1]
            count = np.ravel(array1[1], order='C')[:-1]
            
            #bereiche in denen die gauskurve angepasst wird mit p0 die vorgeschlagenen parameter
            filter = [ 100,410,800, 2100,2600,3600,4000, 5000,7700, 8500,9900,10300,12500,16000]
            farbekurve = ['firebrick', 'orangered', 'darkgoldenrod','gold','olive','lightgreen','seagreen' ,'steelblue', 'navy', 'indigo', 'darkorchid', 'orchid', 'palevioletred']
 
 
            p0=[[ 2.03304213e+05 , 3.71688598e+02 , 1.38456210e+01,7.84246579e-10, -8.90528768e-08],[ 5.56899130e+04,4.23524500e+02,  1.18377925e+01,7.12436862e-01 ,-1.93711512e+02],[ 9.79580683e+04,1.13941589e+03,  9.45890009e+00,-1.55067637e-01,4.55232427e+02],[ 1.79514428e+04, 2.29024103e+03, 9.60253609e+00 ,-1.06668178e-01,3.57069826e+02],[ 4.84145138e+04 ,3.22296313e+03,  9.78905525e+00 ,-3.54271425e-02,1.79197585e+02],[ 3.42128205e+03 ,3.84861972e+03,9.88023154e+00,-1.21204734e-02,9.46506404e+01],[ 4.61144895e+03,4.15653146e+03,  9.62678806e+00 , 1.46982553e-03,4.02783531e+01],[ 1.21730768e+04,  7.29307431e+03,  1.02444988e+01, -5.36811075e-03,7.67028605e+01],[ 3.88002968e+03 , 8.12164193e+03,  1.06215365e+01, -7.58817548e-03,9.43820475e+01],[ 1.18501403e+04 , 9.02761648e+03,  1.06256436e+01, -1.05368309e-02,1.16226496e+02],[ 8.38334459e+03 , 1.01711696e+04,  1.30447228e+01 , 3.04789516e-03,-1.47811402e+01],[ 1.11815297e+04,1.04130546e+04,  1.25852538e+01 ,-3.04363068e-09,3.77571424e-05],[ 1.07814883e+04 , 1.31846092e+04,  7.77529581e+00  ,2.94144852e-10, -4.46244225e-06]] 
            
            b = []
            a =[]
            fig,ax = plt.subplots()
            #schleife in der für jeden bereich die daten der spektren herausgefiltert wird und anschließend an gauskurve angespasst wird
            for i in range (0,len(filter)-1):
                kanal1 = kanal[np.where((kanal > filter[i]) & (kanal< filter[i+1]))]
                count1 = count[np.where((kanal > filter[i]) & (kanal< filter[i+1]))]
                err1 = np.sqrt(count1) + 0.00001
                
                
                popt, pcov = curve_fit(gaus, kanal1, count1, p0=p0[i], sigma=err1, absolute_sigma=True, maxfev = 1000000)
                perr = np.sqrt(np.diag(pcov))
                a.append(popt)
                b.append(perr)
                #berechnung bestimmtheitsmaß
                rsquaredfil = r2_score(count1,gaus(kanal1, *popt))
                
                print (f'{i+1} & {"{:.2f}".format(popt[0])} $\pm$ {"{:.2f}".format(perr[0])} & {"{:.2f}".format(popt[1])} $\pm$ {"{:.2f}".format(perr[1])} & {"{:.3f}".format(popt[2])} $\pm$ {"{:.3f}".format(perr[2])} & {"{:.3f}".format(popt[3])} $\pm$ {"{:.3f}".format(perr[3])} & {"{:.3f}".format(popt[4])} $\pm$ {"{:.3f}".format(perr[4])} & $R^2$ {"{:.3f}".format(rsquaredfil)}')
                #erstellung graphik des angepassten  spektrums
                plt.plot(kanal1, gaus(kanal1, *popt), color = farbekurve[i],linestyle = '-',linewidth = 2, label = f'Gaußkurve {i+1}', zorder =3)
                plt.errorbar(kanal1, count1, yerr=err1, color = 'black',linestyle ='none', marker ='.', markersize =1 , zorder = 2)
                
                plt.grid(zorder =1)
                
            ax.set(ylabel='Anzahl Ansprecher ', xlabel='Kanalnummer', title = 'Spektrum von $^{137}$Cs')
            ax.legend()

            fig.savefig(f"{filename}_einzeln.pdf")
            plt.show

            #speicherung der parameter der rohdatenanpassung 
            parameter = np.asarray(a)
            print (parameter)

            #einlesen der daten der untergrundmessung für den verwendeten detektor
            data = np.loadtxt("ohne_halbleiter.txt", dtype=str)
            data = data.astype(np.float64)
            array = np.split(data, 2, axis=1)

            kanal = np.ravel(array1[0], order='C')[:-1]
            c = np.ravel(array1[1], order='C')[:-1]
            c_unter  = np.ravel(array[1], order='C')[:-1]
            #abzug des untergrunds vom spektrum
            count = c - c_unter
            gesamt_anzahl = np.sum(count)
            gesamt_anzahl_err = np.sqrt(np.sum(np.abs(count)))
            print (f'gesamte Anzahl der gemessenen Anprecher: {gesamt_anzahl} $\pm$ {gesamt_anzahl_err}')
            d = []
            derr = []
            c = []
            cerr = []
            fig,ax = plt.subplots()
            #schleife zur gaußkurvenanpassung in den bereichen für die untergrundbereinigten ansprecher
            for i in range (0,len(filter)-1):
                kanal1 = kanal[np.where((kanal > filter[i]) & (kanal< filter[i+1]))]
                count1 = count[np.where((kanal > filter[i]) & (kanal< filter[i+1]))]
                err1 = np.sqrt(np.abs(count1)) + 0.00001
                #gaußkurvenfit mit den parametern der rohdatenanpassung als iterationsbeginn p0
                popt, pcov = curve_fit(gaus, kanal1, count1, p0=parameter[i], sigma=err1, absolute_sigma=True, maxfev = 1000000)
                perr = np.sqrt(np.diag(pcov))
                d.append(popt[1])
                derr.append(perr[1])
                c.append(popt[2])
                cerr.append(perr[2])
                
                rsquaredfil = r2_score(count1,gaus(kanal1, *popt))
                
                print (f'{i+1} & {"{:.2f}".format(popt[0])} $\pm$ {"{:.2f}".format(perr[0])} & {"{:.2f}".format(popt[1])} $\pm$ {"{:.2f}".format(perr[1])} & {"{:.3f}".format(popt[2])} $\pm$ {"{:.3f}".format(perr[2])} & {"{:.3f}".format(popt[3])} $\pm$ {"{:.3f}".format(perr[3])} & {"{:.3f}".format(popt[4])} $\pm$ {"{:.3f}".format(perr[4])} & $R^2$ {"{:.3f}".format(rsquaredfil)}')
                plt.plot(kanal1, gaus(kanal1, *popt), color = farbekurve[i],linestyle = '-',linewidth = 2, label = f'Gaußkurve {i+1}', zorder =3)
                plt.errorbar(kanal1, count1, yerr=err1, color = 'black',linestyle ='none', marker ='.', markersize =1 , zorder = 2)
                
                plt.grid(zorder =1)
                
            ax.set(ylabel='Anzahl Ansprecher ', xlabel='Kanalnummer', title = 'Spektrum von $^{137}$Cs ohne Untergrund')
            ax.legend()

            fig.savefig(f"{filename}_ohne_untergrund.pdf")
            plt.show
            mu = np.asarray(d)
            merr = np.asarray(derr)
            sigma = np.asarray(c)
            sigmaerr = np.asarray(cerr)
            fwhm = np.zeros(len(filter)-1)
            fwhmerr = np.zeros(len(filter)-1)
            count_kurve = np.zeros(len(filter)-1)
            übergang = []
            übergang_err = []
            #berechnung der halbwertsbreite jeder gaußkurve
            for i in range (0,len(filter)-1):
                fwhm[i] = 2*np.sqrt(2*np.log(2)) *((sigma[i] )/9.365605730383486)
                fwhmerr[i] = np.sqrt(2*np.sqrt(2*np.log(2))*((sigmaerr[i]/9.365605730383486)**2  + ((sigma[i])*4.1387812698069616e-05/(9.365605730383486**2))**2))
                #print (f'FWHM für Gaußkurve{i+1}: {fwhm[i]}$\pm$ {fwhmerr[i]}')
                count_kurve = count[np.where((kanal > (mu[i]-3*np.abs(sigma[i]))) & (kanal< (mu[i]+3*np.abs(sigma[i]))))]
                #print (f'filter {i}:{mu[i]} und {3*sigma[i]}')
                #print (f'filter {i}: {count_kurve}')
                count_übergang = np.sum(count_kurve)
                übergang_c_err = np.sqrt(np.sum(np.abs(count_kurve)))
                übergang.append(count_übergang)
                übergang_err.append(übergang_c_err)
                print (f'peak für Gaußkure {i+1}: {count_übergang} $\pm$ {übergang_c_err}')
            photo =np.asarray(übergang)
            photo_err = np.asarray(übergang_err)
            peaktototal = ((photo[4])/(gesamt_anzahl-photo[2]))
            ptt_err = np.sqrt(((photo_err[4])/(gesamt_anzahl-photo[2]))**2+  (((photo[4])*gesamt_anzahl_err)/(gesamt_anzahl-photo[2])**(2))**2 + (((photo[4])*photo_err[2])/(gesamt_anzahl-photo[2])**(2))**2)
            #print(f'Peak-to-total für Cs: {peaktototal} $\pm$ {ptt_err}')
            N = 21275216
            N_err = 4272629
            effizienz = photo/N
            effizinez_err = np.sqrt((photo_err/N)**2 + (photo*N_err/N**2)**2)  
            #print (f'{"{:.4f}".format({effizienz})} $\pm$ {"{:.4f}".format({effizinez_err})}')  
            I=[28.53 ,  7.55 ,  26.59 , 2.237 , 0.298 , 12.93 ,  4.23 ,  14.51 ,  10.11 ,  13.67 ,  20.87]   
            eff = effizienz[2:]
            eff_err = effizinez_err[2:]
            rel = np.zeros(len(I))
            rel_err = np.zeros(len(I))
            for i in range (len(I)):
                print (f'{"{:.3f}".format(eff[i]*I[i])} $\pm$ {"{:.3f}".format(eff_err[i]*I[i])}')
                rel[i] = eff[i]*I[i]
                rel_err[i] = eff_err[i]*I[i]
            x = np.linspace(10, 1400, 10000)
            E = (21.8 , 244.7 , 344.3 , 411.1 ,  444 ,  778.9 , 867.4 , 964.1 , 1085.8 , 1112.1 , 1408)
            popt_e, pcov_e = curve_fit(relativ, E, rel)
            perr_e = np.sqrt(np.diag(pcov_e))
            print (f'Ausgleichsgerade: $({popt_e[1]}\pm{perr_e[1]}) \cdot FWHM^2  + ({popt_e[0]}\pm{perr_e[0]})$')
            rsquaredfil = r2_score(rel,relativ(E, *popt_e))
            print (f'$R^2$ ', rsquaredfil)
            fig,ax = plt.subplots()
            plt.grid()
            plt.errorbar(E, rel, yerr= rel_err,color = 'darkgoldenrod', marker= '.', linestyle = 'none')
            plt.plot(x ,relativ(x,*popt_e), color = 'gold', linestyle ='-')
            ax.set(ylabel='relative Peakeffizinez \% ', xlabel='Energie /keV', title = 'Kalibrierung Peakeffizienz in Abhängigkeit der Energie')
            fig.savefig(f"{filename}_peakeffizienz.pdf")
            plt.show

        
            
           

            
            




            
            




            
