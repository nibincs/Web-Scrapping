from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("sleep_model.pkl")
encoders = joblib.load("encoders.pkl")


def get_tip(screen_time, exercise, stress):
    if screen_time > 120:
        return "Reduce screen time before bed for better sleep quality."
    elif exercise < 20:
        return "Try increasing your daily exercise duration."
    elif stress > 7:
        return "Practice meditation or relaxation techniques before sleeping."
    else:
        return "Excellent sleep habits. Keep it up!"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    sleep_duration = float(request.form["sleep_duration"])
    caffeine = request.form["caffeine"]
    exercise = float(request.form["exercise"])
    screen_time = float(request.form["screen_time"])
    stress = int(request.form["stress"])
    mood = request.form["mood"]
    interruptions = request.form["interruptions"]

    caffeine_encoded = encoders["caffeine"].transform([caffeine])[0]
    mood_encoded = encoders["mood"].transform([mood])[0]
    interruptions_encoded = encoders["interruptions"].transform([interruptions])[0]

    input_data = pd.DataFrame(
        [[
            sleep_duration,
            caffeine_encoded,
            exercise,
            screen_time,
            stress,
            mood_encoded,
            interruptions_encoded
        ]],
        columns=[
            "sleep_duration",
            "caffeine",
            "exercise",
            "screen_time",
            "stress",
            "mood",
            "interruptions"
        ]
    )

    prediction = model.predict(input_data)[0]

    quality = encoders["sleep_quality"].inverse_transform([prediction])[0]

    if quality.lower() == "good":
        score = 95
    elif quality.lower() == "average":
        score = 70
    else:
        score = 40

    tip = get_tip(screen_time, exercise, stress)

    return render_template(
        "index.html",
        prediction=quality,
        score=score,
        tip=tip,
        sleep_duration=sleep_duration,
        caffeine=caffeine,
        exercise=exercise,
        screen_time=screen_time,
        stress=stress,
        mood=mood,
        interruptions=interruptions
    )


if __name__ == "__main__":
    app.run(debug=True)