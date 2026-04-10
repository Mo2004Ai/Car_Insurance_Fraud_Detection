from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Vehicle Insurance Fraud Detection API")

# تحميل الموديلات وملفات المعالجة
model_bundle = joblib.load("final_models.pkl")
preprocess = joblib.load("preprocess.pkl")
model = model_bundle["voting_model"]
threshold = model_bundle["thresholds"]["voting"]

# تعريف هيكل البيانات القادمة من المستخدم
class ClaimData(BaseModel):
    policy_state: str
    policy_deductible: float
    policy_annual_premium: float
    insured_age: int
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    incident_state: str
    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    bodily_injuries: int
    witnesses: int
    police_report_available: str
    claim_amount: float
    total_claim_amount: float

@app.post("/predict")
def predict_fraud(data: ClaimData):
    # تحويل البيانات القادمة إلى DataFrame
    input_dict = data.dict()
    new_data = pd.DataFrame([input_dict])

    # 1. معالجة القيم المفقودة
    new_data["authorities_contacted"] = new_data["authorities_contacted"].fillna("other")

    # 2. Label Encoding
    label_features = [col for col in preprocess["label_encoders"].keys() if col != 'fraud_reported']
    for col in label_features:
        new_data[col] = preprocess["label_encoders"][col].transform(new_data[col])
    
    label_df = new_data[label_features]

    # 3. One-Hot Encoding
    ohe_parts = []
    for col, enc in preprocess["onehot_encoders"].items():
        arr = enc.transform(new_data[[col]])
        cols = enc.get_feature_names_out([col])
        ohe_parts.append(pd.DataFrame(arr, columns=cols))
    ohe_df = pd.concat(ohe_parts, axis=1)

    # 4. تجميع الأرقام
    num_cols = [
        'policy_deductible', 'policy_annual_premium', 'insured_age',
        'incident_hour_of_the_day', 'number_of_vehicles_involved',
        'bodily_injuries', 'witnesses', 'claim_amount', 'total_claim_amount'
    ]
    X_new = pd.concat([new_data[num_cols], ohe_df, label_df], axis=1)

    # 5. Scaling
    std_cols = [
        'policy_deductible', 'policy_annual_premium', 'insured_age',
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
        'claim_amount', 'total_claim_amount'
    ]
    minmax_cols = ['incident_hour_of_the_day']

    X_new[std_cols] = preprocess["std_scaler"].transform(X_new[std_cols])
    X_new[minmax_cols] = preprocess["minmax_scaler"].transform(X_new[minmax_cols])

    #ترتيب الأعمدة بنفس ترتيب التدريب
    expected_columns = [col for col in preprocess["columns"] if col != 'fraud_reported']
    X_new = X_new[expected_columns]

    # التوقع
    proba = float(model.predict_proba(X_new)[:, 1][0])
    pred = int(proba >= threshold)
    result_status = "Fraud" if pred == 1 else "Legit"

    #  النتيجة
    return {
        "prediction": pred,
        "probability": round(proba, 4),
        "status": result_status
    }
#
# 
#  C:\Users\moham\AppData\Local\Programs\Python\Python313\python.exe -m uvicorn main:app --reload