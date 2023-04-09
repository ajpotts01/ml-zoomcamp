# Non-standard libs
import bentoml
from bentoml.io import JSON
from bentoml import Model, Runner, Service

CURRENT_MODEL_TAG = "credit_risk_model:latest"

# TODO: Make this not top-level
model_ref: Model = bentoml.xgboost.get(CURRENT_MODEL_TAG)
dv = model_ref.custom_objects['dictVectorizer']

model_runner: Runner = model_ref.to_runner()

model_service: Service = bentoml.Service(
    "credit_risk_classifier", runners=[model_runner]
)


@model_service.api(input=JSON(), output=JSON())
def classify(application_data):
    vector = dv.transform(application_data)

    model_prediction = model_runner.predict.run(vector)

    prediction = model_prediction[0]
    print(f"{prediction=}")
    if prediction > 0.5:
        result = {"status": "Declined"}
    else:
        result = {"status": "Approved"}

    return result