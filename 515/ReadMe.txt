Diese Datei erklärt den Aufbau aller Dateien in https://github.com/ManniGertrude/P5/tree/main/515: 

Es gibt 4 Arten von Dateien: 
"Parameter Dateien" ist der Ordner, in dem alle ausgewerteten .root Dateien für die Parametersuche der Driftkammer stecken.
"Root Dateien" ist der Ordner, in dem alle unbearbeiteten .root Dateien liegen, die vor der Langzeitmessung gemessen wurden.
"4x_name.pdf" sind Vektorgrafiken, die bei der Auswertung entstanden sind. Sie sind geordnet nach Aufgabenteilen und haben Bezeichnungen, um sie zuzuordnen.
"analysis_name.c" sind verschiedene Versionen der selben Auswertungsdatei. Die neuste und mächtigste ist analysis.c, jedoch sind im Auswertungsprozess viele andere Dateien als Backup entstanden.
Begriffserklärung im Namen:
    - TOT = TimeOverThreshold bzw. Zeit über der Signalschwelle
    - DN = Drahtnummer
    - DT = Driftzeit
    - HITS = Trefferanzahl
    - ODB = Orts-Driftzeitbeziehung
    - anfang = Vor der Filteranwendung
    - fixed = Nach der Filteranwendung
    - lin = lineare Skala
    - log = logarithmische Skala
    - 0x-0y = Von Draht x bis Draht y
Abschließend gibt es noch einige Dateien wie "makefile" oder "plot.C", welche nicht in die Kategorien einsortierbar sind und einen eigenen Nützen haben
"plot.C" erlaubt es, mehrere Plots in einem Histrogramm zu betrachten
"messung_b201.root" ist die Langzeitmessung
"run_240502_164450.root" ist die letzte Messung der Quelle mit den finalen Parametern
"515.py" ist eine frühere Version der Fitfunktion der Spannungsuntersuchung
"Driftstrom_mit/ohne.csv" sind die Werte aus der Vermessung der Spannung
