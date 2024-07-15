def ersetze_punkt_mit_komma(dateiname):
    with open(dateiname, 'r') as datei:
        inhalt = datei.read()

    neuer_inhalt = inhalt.replace('.', ',')

    with open(dateiname, 'w') as datei:
        datei.write(neuer_inhalt)

ersetze_punkt_mit_komma('C:\\Users\\kontr\\Desktop\\Github\\P5\\521\\dots.txt')