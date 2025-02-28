import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# URL de l'API FastAPI
FASTAPI_URL = "http://127.0.0.1:8000/predict"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Récupérer les données du formulaire
        input_data = {
            "account_length": float(request.form["account_length"]),
            "international_plan": request.form["international_plan"],
            "voice_mail_plan": request.form["voice_mail_plan"],
            "number_vmail_messages": int(request.form["number_vmail_messages"]),
            "total_day_minutes": float(request.form["total_day_minutes"]),
            "total_day_calls": int(request.form["total_day_calls"]),
            "total_eve_minutes": float(request.form["total_eve_minutes"]),
            "total_eve_calls": int(request.form["total_eve_calls"]),
            "total_night_minutes": float(request.form["total_night_minutes"]),
            "total_night_calls": int(request.form["total_night_calls"]),
            "total_intl_minutes": float(request.form["total_intl_minutes"]),
            "total_intl_calls": int(request.form["total_intl_calls"]),
            "customer_service_calls": int(request.form["customer_service_calls"]),
        }

        # Envoyer les données à l'API FastAPI
        response = requests.post(FASTAPI_URL, json=input_data)
        if response.status_code == 200:
            prediction = response.json()["prediction"][0]
        else:
            prediction = "Erreur lors de la prédiction"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
