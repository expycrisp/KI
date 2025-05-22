import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
#df_c = df_c.drop(columns=['id']) # Löscht die ID-Spalte
#df[['Category']] # Zeigt die Kategorie-Spalte

#df[0:k] #Gibt die ersten k einträge zurück 

#print(n)
#print(len(df[df['Profession'] == 'Student'])) # Anzahl der Studenten
#print(len(df[df['Job Satisfaction'] == 0])) # Anzahl der Studenten mit Jobzufriedenheit 0']]))

# Einträge mit Job Satisfaction ungleich 0
#print(df[df['Job Satisfaction'] != 0])
#df_c = df[df['Category'] == 'Option'] # Filtert Kategorie nach Optionen (überall, wo diese Option in der Kategorie vorkommt)

# Einträge mit Profession ungleich 'Student'
#print(df[df['Profession'] != 'Student'])

## Datenaufbereitung
def convert_sleep_duration(sleep_str):
    if sleep_str == "'Less than 5 hours'":
        return 4.5
    elif sleep_str == "'5-6 hours'":
        return 5.5
    elif sleep_str == "'7-8 hours'":
        return 7.5
    elif sleep_str == "'More than 8 hours'":
        return 9.0
    else:
        return np.nan
    
    
    
def convert_dietary_habits(dietary_str):
    if dietary_str == "Healthy":
        return 0
    elif dietary_str == "Moderate":
        return 1
    elif dietary_str == "Unhealthy":
        return 2
    else:
        return np.nan
    
    
    
    
    
# Lese Daten
file_path = '/home/jan/Programmieren/Ordentlich/KI/Datasets/student_depression_dataset.csv'

Kategorien = ['id', 'Gender' ,'Age,City', 'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction' , 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 'Depression']
df = pd.read_csv(file_path)

df_c = df.drop(columns=['id', 'City'])  # 'Depression' behalten wir für späteres yscaler = MinMaxScaler()












## Daten in numerische Werte aufbereiten

# Gender in 0 und 1 umwandeln
df_c['Gender'] = df_c['Gender'].apply(lambda x: 0 if x == 'Male' else 1)

# Sleep Duration in Stunden umwandeln, NaN Werte entfernen
df_c['Sleep Duration'] = df_c['Sleep Duration'].apply(convert_sleep_duration)

df_c = df_c.dropna(subset=['Sleep Duration']) # Entfernt Zeilen mit NaN-Werten in der Spalte 'Sleep Duration'

# Konvertiere die Dietary Habits in numerische Werte
df_c['Dietary Habits'] = df_c['Dietary Habits'].apply(convert_dietary_habits)
df_c = df_c.dropna(subset=['Dietary Habits']) # Entfernt Zeilen mit NaN-Werten in der Spalte 'Dietary Habits'


# Konvertiere die Degree in numerische Werte
# Filter out rows where 'Degree' is 'Others'
df_c = df_c[df_c['Degree'] != 'Others']

# Assign a numeric value to each unique 'Degree'
degree_mapping = {degree: idx for idx, degree in enumerate(df_c['Degree'].unique())}
df_c['Degree'] = df_c['Degree'].map(degree_mapping) # Jeder der 27 Einträge erhält einen eigenen Wert

# Konvertiere Suicidal Thoughts und Financial Stress in 0 und 1
df_c['Have you ever had suicidal thoughts ?'] = df_c['Have you ever had suicidal thoughts ?'].apply(lambda x: 0 if x == 'Yes' else 1)
df_c['Family History of Mental Illness'] = df_c['Family History of Mental Illness'].apply(lambda x: 0 if x == 'Yes' else 1)


# Mapping unique professions to numeric values
unique_values = df_c['Profession'].unique()
profession_mapping = {profession: idx for idx, profession in enumerate(unique_values)}
df_c['Profession'] = df_c['Profession'].map(profession_mapping)


# Konvertiere Financial Stress zu numerische Werte
df_c = df_c[df_c['Financial Stress'] != '?']






# Separate the target variable 'Depression' for later use
y = df_c['Depression']
X = df_c.drop(columns=['Depression'])  # 'Depression' behalten wir für späteres yscaler = MinMaxScaler()



# Normieren
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)




# Speichern der bereinigten Daten in einer neuen CSV-Datei
pd.DataFrame(X_scaled, columns=X.columns).to_csv('/home/jan/Programmieren/Ordentlich/KI/Datasets/cleaned_student_depression_dataset.csv', index=False)
y.to_csv('/home/jan/Programmieren/Ordentlich/KI/Datasets/cleaned_student_depression_dataset_outcome.csv', index=False)
print("Bereinigte Daten wurden erfolgreich gespeichert.")




# Scalar speichern
with open("/home/jan/Programmieren/Ordentlich/KI/Gespeicherte_Scalars/student_depression_dataset.pkl", "wb") as f:
    pickle.dump(scaler, f)
    

    
    