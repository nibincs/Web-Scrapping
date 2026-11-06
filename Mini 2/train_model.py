import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("sleep_dataset.csv")

encoders = {}

categorical_columns = [
    "caffeine",
    "mood",
    "interruptions",
    "sleep_quality"
]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

X = df.drop("sleep_quality", axis=1)
y = df["sleep_quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy : {accuracy * 100:.2f}%")

joblib.dump(model, "sleep_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model saved successfully")