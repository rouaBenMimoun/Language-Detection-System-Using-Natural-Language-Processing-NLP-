import pandas as pd
data = pd.read_csv("Language_Detection.csv")
data.head(10)
language_counts=data["Language"].value_counts()
language_percentages = (language_counts / len(data)) * 100
print(language_percentages)

#separation variable dependante et indépendante

X = data["Text"]
y = data["Language"]

 #Encodage de la variable ‘Langue’

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


#pretraitement du texte 
import re
data_list = []
for text in X:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)

#Sac de mots (Bag of words)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
a=X.shape
print ("taille de vecteur des mots est" ,a)
#Fractionnement de la base de données en données d’apprentissage et en données de test
from sklearn.model_selection import train_test_split
import numpy as np
y = y.astype(np.int8)
X = X.astype(np.int16)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Entraînement du modèle et prédiction (classification)
import pickle
filename = 'language_detection_model.sav'
model = pickle.load(open(filename, 'rb'))
y_pred = model.predict(x_test)


#etude de performance

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
ac = 100*accuracy_score(y_test, y_pred)
print("accuracy" , ac)
cm = confusion_matrix(y_test, y_pred)
cm=100*cm/ cm.astype(float).sum(axis=1)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

#prediction 
def predict(text):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The langauge is in",lang[0])

predict("¿Este ejercicio le brindó una introducción al procesamiento del lenguaje natural?")