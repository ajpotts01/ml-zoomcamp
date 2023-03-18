import pickle

import numpy as np
from flask import Flask, request, jsonify, Response

INPUT_PATH = "../models/model_reg=%s.bin"
final_reg_factor: float = 0.1

# Time to load!
# Note: This requires scikit-learn to be installed
with open(INPUT_PATH % final_reg_factor, "rb") as file_model_input:
    (loaded_vect, loaded_model) = pickle.load(file_model_input)


app = Flask("churn")


def model_prediction(customer: dict) -> tuple[float, float]:
    X: np.ndarray = loaded_vect.transform([customer])
    prediction: float = loaded_model.predict_proba(X)[0, 1]
    churn: float = prediction >= 0.5

    return prediction, churn


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    customer: dict = request.get_json()

    prediction, churn = model_prediction(customer=customer)

    response = dict(churn_probability=prediction, churn=churn)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
