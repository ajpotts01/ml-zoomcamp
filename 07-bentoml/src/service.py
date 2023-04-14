# Non-standard libs
import bentoml
from bentoml.io import JSON, NumpyNdarray
from bentoml import Model, Runner, Service
from pydantic import BaseModel

# {
#   "seniority": 7,
#   "home": "other",
#   "time": 48,
#   "age": 34,
#   "marital": "married",
#   "records": "no",
#   "job": "freelance",
#   "expenses": 45,
#   "income": 37.0,
#   "assets": 0.0,
#   "debt": 0.0,
#   "amount": 800,
#   "price": 941
# }

class CreditApplication(BaseModel):
  seniority: int
  home: str
  time: int
  age: int
  marital: str
  records: str
  job: str
  expenses: int
  income: float
  assets: float
  debt: float
  amount: int
  price: int


CURRENT_MODEL_TAG = "credit_risk_model:latest"

# TODO: Make this not top-level
model_ref: Model = bentoml.xgboost.get(CURRENT_MODEL_TAG)
dv = model_ref.custom_objects['dictVectorizer']

model_runner: Runner = model_ref.to_runner()

model_service: Service = bentoml.Service(
    "credit_risk_classifier", runners=[model_runner]
)


@model_service.api(input=JSON(pydantic_model=CreditApplication), output=JSON())
def classify(credit_application: CreditApplication):
    application_data = credit_application.dict()
    vector = dv.transform(application_data)

    model_prediction = model_runner.predict.run(vector)

    prediction = model_prediction[0]
    print(f"{prediction=}")
    if prediction > 0.5:
        result = {"status": "Declined"}
    else:
        result = {"status": "Approved"}

    return result