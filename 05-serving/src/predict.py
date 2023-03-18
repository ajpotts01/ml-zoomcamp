import pickle


INPUT_PATH = "../models/model_reg=%s.bin"
final_reg_factor: float = 0.1
# Time to load!
# Note: This requires scikit-learn to be installed
with open(INPUT_PATH % final_reg_factor, "rb") as file_model_input:
    (loaded_vect, loaded_model) = pickle.load(file_model_input)

test_customer = dict(
    gender="female",
    seniorcitizen=0,
    partner="yes",
    dependents="no",
    phoneservice="no",
    multiplelines="no_phone_service",
    internetservice="dsl",
    onlinesecurity="no",
    onlinebackup="yes",
    deviceprotection="no",
    techsupport="no",
    streamingtv="no",
    streamingmovies="no",
    contract="month-to-month",
    paperlessbilling="yes",
    paymentmethod="electronic_check",
    tenure="1",
    monthlycharges=29.85,
    totalcharges=29.85
)


X = loaded_vect.transform([test_customer])
prediction = loaded_model.predict_proba(X)[0, 1]
print(prediction)