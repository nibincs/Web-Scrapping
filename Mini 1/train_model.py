import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]

df.columns = ['label', 'message']

# Convert labels into numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Input and Output
X = df['message']
y = df['label']

# Convert text into vectors
cv = CountVectorizer()

X = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

print("Model and Vectorizer saved successfully!")