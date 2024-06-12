import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
import numpy as np


Mb15000V100mAUc = np.array([17.31, 33.56, 44.4, 66.8, 74.3, 9.04, 13.83, 25.11, 2.957, 0.076, 279.1])
Mb15000V100mAUE = np.array([0.162, 0.202, 0.215, 0.230, 0.222, 0.124, 0.155, 0.192, 0.069, 0.039, 0.190])

I_e1 = np.zeros(len(Mb15000V100mAUE))
for i in range (0, len(Mb15000V100mAUE)):
    I_e1[i] = Mb15000V100mAUE[i]/(1*10**9) * 10**9
print(I_e1)

mb15VUcerr = np.zeros(len(Mb15000V100mAUc))
for i in range (0, len(Mb15000V100mAUc)):
    mb15VUcerr[i] = Mb15000V100mAUc[i] * 0.01

Ie1err = np.zeros(len(I_e1))
for i in range(0, len(I_e1)):
    Ie1err[i] = (Mb15000V100mAUE[i]*0.05)/(1*10**9) * 10**9

Mb25000V100mAUc = np.array([7.64, 14.77, 24.47, 34.73, 44.4, 52.7, 63.5, 80.0, 100.2, 135.8, 199.7, 19.82])
Mb25000V100mAUE = np.array([0.320, 0.602, 0.972, 1.289, 1.46, 1.56, 1.63, 1.69, 1.70, 1.70, 1.69, 0.79])

I_e2 = np.zeros(len(Mb25000V100mAUE))
for i in range (0, len(Mb25000V100mAUE)):
    I_e2[i] = Mb25000V100mAUE[i]/(1*10**9) * 10**9
print(I_e2)

mb25VUcerr = np.zeros(len(Mb25000V100mAUc))
for i in range (0, len(Mb25000V100mAUc)):
    mb25VUcerr[i] = Mb25000V100mAUc[i] * 0.01

Ie2err = np.zeros(len(I_e2))
for i in range(0, len(I_e2)):
    Ie2err[i] = (Mb25000V100mAUE[i]*0.05)/(1*10**9) * 10**9

Mb35000V100mAUc = np.array([3.44, 10.91, 16.53, 22.54, 34.12, 46.5, 57.5, 68.8, 81.1, 102.5, 130.6, 173.4, 196.6, 230.8, 279.5])
Mb35000V100mAUE = np.array([0.25, 0.68, 1.02, 1.49, 2.17, 3.00, 3.58, 3.95, 4.26, 4.50, 4.69, 4.76, 4.84, 4.79, 4.82 ])

I_e3 = np.zeros(len(Mb35000V100mAUE))
for i in range (0, len(Mb35000V100mAUE)):
    I_e3[i] = Mb35000V100mAUE[i]/(1*10**9) * 10**9
print(I_e3)

mb35VUcerr = np.zeros(len(Mb35000V100mAUc))
for i in range (0, len(Mb35000V100mAUc)):
    mb35VUcerr[i] = Mb35000V100mAUc[i] * 0.01

Ie3err = np.zeros(len(I_e3))
for i in range(0, len(I_e3)):
    Ie3err[i] = (Mb35000V100mAUE[i]*0.05)/(1*10**9) * 10**9


fig, ax= plt.subplots()

plt.grid()

plt.errorbar(Mb15000V100mAUc, I_e1, xerr= mb15VUcerr, yerr= Ie1err, color='rebeccapurple', marker='+',capsize=2,  linestyle='none', label= 'U = 15kV')
plt.errorbar(Mb25000V100mAUc, I_e2, xerr= mb25VUcerr, yerr= Ie2err, color='slateblue', marker='+',capsize=2,  linestyle='none', label= 'U = 25kV')
plt.errorbar(Mb35000V100mAUc, I_e3, xerr= mb35VUcerr, yerr= Ie3err,color='midnightblue', marker='+',capsize=2,  linestyle='none', label= 'U = 35kV' )
ax.set(ylabel='Ionisationsstrom $I_C$/nA', xlabel='Kondensatorspannung $U_C$/V')

ax.legend()

fig.savefig('P5\\529\\Moionisationsstrom.pdf')
plt.show


Cu15000V100mAUc = np.array([19.49, 51.6, 37.05, 68.4, 82.9, 100.0, 134.4, 160.9, 205.3, 240.7, 277.8])
Cu15000V100mAUE = np.array([0.142, 0.418, 0.289, 0.545, 0.619, 0.655, 0.706, 0.715, 0.725, 0.734, 0.738])

I_e1cu = np.zeros(len(Cu15000V100mAUE))
for i in range (0, len(Cu15000V100mAUE)):
    I_e1cu[i] = Cu15000V100mAUE[i]/(1*10**9) * 10**8
print(I_e1cu)

cu15VUcerr = np.zeros(len(Cu15000V100mAUc))
for i in range (0, len(Cu15000V100mAUc)):
    cu15VUcerr[i] = Cu15000V100mAUc[i] * 0.01

Ie1cuerr = np.zeros(len(I_e1cu))
for i in range(0, len(I_e1cu)):
    Ie1cuerr[i] = (Cu15000V100mAUE[i]*0.05)/(1*10**8) * 10**9

Cu25000V100mAUc = np.array([22.14, 41.1, 51.3, 68.0, 82.2, 105.1, 140.8, 159.2, 204.2, 278.4, 245.5])
Cu25000V100mAUE = np.array([0.225, 0.286, 0.655, 0.927, 1.183, 1.569, 2.047, 2.189, 2.402, 2.569, 2.505])

I_e2cu = np.zeros(len(Cu25000V100mAUE))
for i in range (0, len(Cu25000V100mAUE)):
    I_e2cu[i] = Cu25000V100mAUE[i]/(1*10**8) * 10**9
print(I_e2cu)

cu25VUcerr = np.zeros(len(Cu25000V100mAUc))
for i in range (0, len(Cu25000V100mAUc)):
    cu25VUcerr[i] = Cu25000V100mAUc[i] * 0.01

Ie2cuerr = np.zeros(len(I_e2cu))
for i in range(0, len(I_e2cu)):
    Ie2cuerr[i] = (Cu25000V100mAUE[i]*0.05)/(1*10**8) * 10**9

Cu35000V100mAUc = np.array([17.11, 34.44, 24.84, 46.7, 60.5, 75.2, 88.1, 95.8, 121.3, 140.5, 166.9, 181.5, 201.6, 231.1, 278.1])
Cu35000V100mAUE = np.array([0.19, 0.447, 0.298, 0.665, 0.953, 1.277, 1.572, 1.76, 2.386, 2.817, 3.35, 3.65, 3.918, 4.22, 4.49])

I_e3cu = np.zeros(len(Cu35000V100mAUE))
for i in range (0, len(Cu35000V100mAUE)):
    I_e3cu[i] = Cu35000V100mAUE[i]/(1*10**8) * 10**9
print(I_e3cu)

cu35VUcerr = np.zeros(len(Cu35000V100mAUc))
for i in range (0, len(Cu35000V100mAUc)):
    cu35VUcerr[i] = Cu35000V100mAUc[i] * 0.01

Ie3cuerr = np.zeros(len(I_e3cu))
for i in range(0, len(I_e3cu)):
    Ie3cuerr[i] = (Cu35000V100mAUE[i]*0.05)/(1*10**8) * 10**9

fig, ax= plt.subplots()

plt.grid()   

plt.errorbar(Cu15000V100mAUc, I_e1cu, xerr= cu15VUcerr, yerr= Ie1cuerr, color='rebeccapurple', marker='+',capsize=2,  linestyle='none', label= 'U = 15kV')
plt.errorbar(Cu25000V100mAUc, I_e2cu, xerr= cu25VUcerr, yerr= Ie2cuerr, color='slateblue', marker='+',capsize=2,  linestyle='none', label= 'U = 25kV')
plt.errorbar(Cu35000V100mAUc, I_e3cu, xerr= cu35VUcerr, yerr= Ie3cuerr,color='midnightblue', marker='+',capsize=2,  linestyle='none', label= 'U = 35kV' )
ax.set(ylabel='Ionisationsstrom $I_C$/nA', xlabel='Kondensatorspannung $U_C$/V')

ax.legend()

fig.savefig('P5\\529\\cuionisationsstrom.pdf')
plt.show

t= 23+273.15
terr = 1
p = 1006
perr = 3
V= 1.24*10**(-4)
ro0= 1.293
p0=1013
t0=273

dm = ro0*(t0/t)*(p/p0)*V
dmerr = np.sqrt((ro0*(perr/p0)*(t0/t)*V)**2 + (ro0*(p/p0)*(t0*terr/(t**2))*V)**2)


Mb35000V279dot8VUcI = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Mb35000V279dot8VUcUE = np.array([0.55, 1.03, 1.51, 1.98, 2.45, 2.92, 3.41, 3.86, 4.3, 4.76])

jmoi = np.zeros(len(Mb35000V279dot8VUcUE))
for i in range (0, len(Mb35000V279dot8VUcUE)):
    jmoi[i] = (Mb35000V279dot8VUcUE[i]/(1*10**9))/dm *10**6
print(jmoi)

moIerr = np.zeros(len(Mb35000V279dot8VUcI))
for i in range (0, len(Mb35000V279dot8VUcI)):
    moIerr[i] = Mb35000V279dot8VUcI[i] * 0.05

jmoierr = np.zeros(len(jmoi))
for i in range(0, len(jmoi)):
    jmoierr[i] = np.sqrt((((Mb35000V279dot8VUcUE[i]*0.05)/(1*10**9))/dm)**2 + (((Mb35000V279dot8VUcUE[i])/(1*10**9)*dmerr)/(dm**2))**2)*10**6


Mb100mA279dot8VUcU = np.array([2.5, 5.0, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35])
Mb100mA279dot8VUcUE = np.array([0.05, 0.05, 0.05, 0.05, 0.07, 0.16, 0.34, 0.6, 1.03, 1.6, 2.27, 3.03, 3.85, 4.8])

jmou = np.zeros(len(Mb100mA279dot8VUcUE))
for i in range (0, len(Mb100mA279dot8VUcUE)):
    jmou[i] = (Mb100mA279dot8VUcUE[i]/(1*10**9))/dm *10**6
print(jmou)

moUerr = np.zeros(len(Mb100mA279dot8VUcU))
for i in range (0, len(Mb100mA279dot8VUcU)):
    moUerr[i] = Mb100mA279dot8VUcU[i] * 0.05

jmouerr = np.zeros(len(jmou))
for i in range(0, len(jmou)):
    jmouerr[i] = np.sqrt((((Mb100mA279dot8VUcUE[i]*0.05)/(1*10**9))/dm)**2 + (((Mb100mA279dot8VUcUE[i])/(1*10**9)*dmerr)/(dm**2))**2)*10**6



Cu35000V278dot8VUcI = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Cu35000V278dot8VUcUE = np.array([0.54, 1.02, 1.50, 1.95, 2.4, 2.85, 3.27, 3.68, 4.08, 4.47])

jcui = np.zeros(len(Cu35000V278dot8VUcUE))
for i in range (0, len(Cu35000V278dot8VUcUE)):
    jcui[i] = (Cu35000V278dot8VUcUE[i]/(1*10**8))/dm *10**6
print(jcui)

cuIerr = np.zeros(len(Cu35000V278dot8VUcI))
for i in range (0, len(Cu35000V278dot8VUcI)):
    cuIerr[i] = Cu35000V278dot8VUcI[i] * 0.05

jcuierr = np.zeros(len(jcui))
for i in range(0, len(jcui)):
    jcuierr[i] = np.sqrt((((Cu35000V278dot8VUcUE[i]*0.05)/(1*10**8))/dm)**2 + (((Cu35000V278dot8VUcUE[i])/(1*10**8)*dmerr)/(dm**2))**2) * 10**6



Cu100mA278dot8VUcU = np.array([2.5, 5.0, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35])
Cu100mA278dot8VUcUE = np.array([0.02, 0.02, 0.04, 0.15, 0.39, 0.72, 1.14, 1.56, 2.04, 2.52, 3.01, 3.54, 4.00, 4.45])

jcuu = np.zeros(len(Cu100mA278dot8VUcUE))
for i in range (0, len(Cu100mA278dot8VUcUE)):
    jcuu[i] = (Cu100mA278dot8VUcUE[i]/(1*10**8))/dm *10**6
print(jcuu)

cuUerr = np.zeros(len(Cu100mA278dot8VUcU))
for i in range (0, len(Cu100mA278dot8VUcU)):
    cuUerr[i] = Cu100mA278dot8VUcU[i] * 0.05

jcuuerr = np.zeros(len(jcuu))
for i in range(0, len(jcuu)):
    jcuuerr[i] = np.sqrt((((Cu100mA278dot8VUcUE[i]*0.05)/(1*10**8))/dm)**2 + (((Cu100mA278dot8VUcUE[i])/(1*10**8)*dmerr)/(dm**2))**2) * 10**6


# def ionenzuequi(j):
#     """
#     Returns temperature in Celsius given Fahrenheit temperature.
#     """
#     return j * 32,4


# def make_plot():

#     # Define a closure function to register as a callback
#     def convert_ax_c_to_sievert(ax_f):
#         """
#         Update second axis according to first axis.
#         """
#         y1, y2 = ax_f.get_ylim()
#         ax_c.set_ylim(ionenzuequi(y1*10**3), ionenzuequi(y2))
#         ax_c.figure.canvas.draw()

#     fig, ax_f = plt.subplots()
#     ax_c = ax_f.twinx()

#     # automatically update ylim of ax2 when ylim of ax1 changes.
#     ax_f.callbacks.connect("ylim_changed", convert_ax_c_to_sievert)
#     plt.grid()   
#     plt.errorbar(Cu35000V278dot8VUcI, jcui, xerr= cuIerr, yerr= jcuierr, color='rebeccapurple', marker='+',capsize=2,  linestyle='none', label= 'Kupfer')
#     plt.errorbar(Mb35000V279dot8VUcI, jmoi, xerr= moIerr, yerr= jmoierr,color='slateblue', marker='+',capsize=2,  linestyle='none', label= 'Molybdän' )
#     #ax_f.plot(np.linspace(-40, 120, 100))
#     #ax_f.set_xlim(0, 100)

#     #ax_f.set_title('Two scales: Fahrenheit and Celsius')
#     ax_f.set_ylabel('mittlere Ionendosisleistung <$j$>/ $\mu$Akg$^{-1}$')
#     ax_c.set_ylabel('Äquivalenzdosis $\mu$Svs$^{-1}$')
#     ax.set_xlabel('Emissionsstrom $I_E$/mA')
#     ax.legend()

#     fig.savefig('P5\\529\\dosisstrom.pdf')

#     plt.show()

#     fig, ax_f = plt.subplots()
#     ax_c = ax_f.twinx()

#     # automatically update ylim of ax2 when ylim of ax1 changes.
#     ax_f.callbacks.connect("ylim_changed", convert_ax_c_to_sievert)
#     plt.grid()   
#     plt.errorbar(Cu100mA278dot8VUcU, jcuu, xerr= cuUerr, yerr= jcuuerr, color='rebeccapurple', marker='+',capsize=2,  linestyle='none', label= 'Kupfer')
#     plt.errorbar(Mb100mA279dot8VUcU, jmou, xerr= moUerr, yerr= jmouerr,color='slateblue', marker='+',capsize=2,  linestyle='none', label= 'Molybdän' )
#     #ax_f.plot(np.linspace(-40, 120, 100))
#     #ax_f.set_xlim(0, 100)

#     #ax_f.set_title('Two scales: Fahrenheit and Celsius')
#     ax_f.set_ylabel('mittlere Ionendosisleistung <$j$>/ $\mu$Akg$^{-1}$')
#     ax_c.set_ylabel('Äquivalenzdosis $\mu$Svs$^{-1}$')
#     ax.set_xlabel('Beschleunigungsspannung $U$/kV')
#     ax.legend()

#     fig.savefig('P5\\529\\dosisspannung.pdf')

#     plt.show()



# #make_plot()


# fig, ax= plt.subplots()

# plt.grid()   

# plt.errorbar(Cu35000V278dot8VUcI, jcui, xerr= cuIerr, yerr= jcuierr, color='rebeccapurple', marker='+',capsize=2,  linestyle='none', label= 'Kupfer')
# plt.errorbar(Mb35000V279dot8VUcI, jmoi, xerr= moIerr, yerr= jmoierr,color='slateblue', marker='+',capsize=2,  linestyle='none', label= 'Molybdän' )
# ax.set(ylabel='mittlere Ionendosisleistung <$j$>/ $\mu$Akg$^{-1}$', xlabel='Emissionsstrom $I_E$/mA')

# ax.legend()

# fig.savefig('P5\\529\\dosisstrom.pdf')
# plt.show

fig, ax= plt.subplots()
ax_1 = ax.twinx()
plt.grid()   
plt.errorbar(Cu100mA278dot8VUcU, jcuu, xerr= cuUerr, yerr= jcuuerr, color='rebeccapurple', marker='+',capsize=2,  linestyle='none', label= 'Kupfer')
plt.errorbar(Mb100mA279dot8VUcU, jmou, xerr= moUerr, yerr= jmouerr,color='slateblue', marker='+',capsize=2,  linestyle='none', label= 'Molybdän' )
ax.set(ylabel='mittlere Ionendosisleistung <$j$>/ $\mu$Akg$^{-1}$')
plt.set(xlabel='Beschleunigungsspannung $U$/kV')
ax_1.set(ylabel='<$h$>/ $mSv/s$')
ax_1.tick_params(axis='y')
plt.legend()
fig.savefig("P5\\529\\dosisspannung.pdf")
plt.show
