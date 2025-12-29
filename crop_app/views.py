from django.shortcuts import render
import numpy as np
import pickle
import os

# Load ML model safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

if isinstance(model,list):
    model = model[0];


def home(request):
    result = None

    if request.method == "POST":
        try:
            N = float(request.POST.get('N'))
            P = float(request.POST.get('P'))
            K = float(request.POST.get('K'))
            temperature = float(request.POST.get('temperature'))
            humidity = float(request.POST.get('humidity'))
            ph = float(request.POST.get('ph'))
            rainfall = float(request.POST.get('rainfall'))

            # Convert inputs to numpy array
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Prediction
            result = model.predict(data)[0]

        except Exception as e:
            result = f"Error: {e}"

    return render(request, "index.html", {"result": result})
