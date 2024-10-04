import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Charger les données
data = pd.read_csv('data/emails.csv')

# Afficher les premières lignes pour comprendre les données
print(data.head())

# Séparer les caractéristiques (features) et les cibles (targets)
X = data['text']  # Colonne contenant le texte des e-mails
y = data['spam']   # Colonne indiquant si c'est spam ou non (0 ou 1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorisation du texte
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

# Entraîner le modèle
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Enregistrer le modèle et le vectoriseur
with open('Naive_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Modèle et vectoriseur enregistrés avec succès !")
