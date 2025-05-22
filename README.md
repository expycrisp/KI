Datenauswertung mit KI

Projektziel
Ziel dieses Projekts ist es, mit Hilfe eines tiefen neuronalen Netzes vorherzusagen, ob ein Studierender an Depressionen leidet. Grundlage bilden bereinigte Umfragedaten, die mittels eines Multilayer Perceptrons (MLP) ausgewertet werden.


Verwendete Technologien
    • Programmiersprache: Python
    • Frameworks & Bibliotheken:
        ◦ TensorFlow / Keras (Modellerstellung und Training)
        ◦ scikit-learn (Datenaufteilung)
        ◦ pandas (Datenverarbeitung)
        ◦ matplotlib (Visualisierung)
    • Datenquelle: Bereinigter Umfragedatensatz über studentische Depression




Modellarchitektur
Das trainierte Modell ist ein Multilayer Perceptron (MLP), ein vollständig verbundenes neuronales Netz zur binären Klassifikation.
Die Architektur des Modells ist flexibel konfigurierbar und besteht aus:
    • Einer Eingabeschicht, angepasst an die Anzahl der Features im Datensatz
    • Einer vom Nutzer bestimmten Anzahl an versteckten Schichten (Dense Layers) mit wählbarer Aktivierungsfunktion
    • Einer Ausgabeschicht mit einem Neuron und einer sigmoid-Aktivierungsfunktion (für binäre Entscheidung)
Verwendet wird der Binary Crossentropy Loss, da das Ziel die Klassifikation in zwei Gruppen ist (depressiv / nicht depressiv).



Ordnerstruktur und Datenfluss
KI/
├── Netz.py
├── Datasets/
│   ├── cleaned_student_depression_dataset.csv
│   └── cleaned_student_depression_dataset_outcome.csv
├── Gespeicherte_Scalars/
│   └── (optional: passende Scaler)
├── Gespeicherte_Netze/
│   └── gespeichertes_model.keras
    • Netz.py: Hauptskript für die Erstellung, das Training und die Evaluierung des Modells.
    • Datasets/: Enthält die vorbereiteten und bereinigten Datensätze.
    • Gespeicherte_Scalars/: (Optional) Ablage für Scaler zur Vorverarbeitung.
    • Gespeicherte_Netze/: Ablageort für trainierte Modelle im .keras-Format.
Der Code lädt die bereinigten Eingabedaten und Zielwerte, teilt diese in Trainings-, Validierungs- und Testmengen auf und trainiert anschließend das Netzwerk. Das fertige Modell wird gespeichert und auf den Testdaten evaluiert.


Trainingsverlauf & Visualisierung
Während des Trainings wird der Verlauf des Trainings- und Validierungsfehlers aufgezeichnet und graphisch dargestellt. Dies dient der Bewertung möglicher Overfitting-Tendenzen und der allgemeinen Lernfähigkeit des Netzes.

🔍 Modellparameter (dynamisch einstellbar)
Die folgenden Hyperparameter werden zu Beginn des Skripts vom Nutzer interaktiv eingegeben:
    • Anzahl der versteckten Schichten
    • Anzahl der Neuronen pro Schicht
    • Aktivierungsfunktion (z. B. relu, sigmoid, tanh)
    • Anzahl der Epochen
Optional könnte auch ein Learning Rate Scheduler aktiviert werden, der die Lernrate bei stagnierender Validierungsleistung reduziert.

Ergebnisse
Nach dem Training wird das Modell auf den Testdaten evaluiert. Dabei werden der finale Loss-Wert sowie die Genauigkeit (Accuracy) ausgegeben. Zusätzlich wird das Modell zur späteren Wiederverwendung abgespeichert.

 Lernziele & persönlicher Nutzen
    • Praktische Anwendung von Keras zur Erstellung neuronaler Netze
    • Vertieftes Verständnis der Zusammenhänge zwischen Modellarchitektur und Lernverhalten
    • Übung im Vorverarbeiten und Trennen von Datensätzen
    • Nutzung eigener Datensätze für einen realitätsnahen Use Case
    • Aufbau einer wiederverwendbaren Codebasis für Klassifikationsprobleme

