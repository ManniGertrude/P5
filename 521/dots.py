
# simpler Code, der alle Punkte in dots.txt in Kommata tauscht
def Ersetzer(dateiname):
    with open(dateiname, 'r') as datei:
        data = datei.read()

    data_neu = data.replace('.', ',')

    with open(dateiname, 'w') as datei:
        datei.write(data_neu)

Ersetzer('C:\\Users\\kontr\\Desktop\\Github\\P5\\521\\dots.txt')