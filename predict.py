import pandas as pd
import joblib

# 1. Input Data
new_data = pd.DataFrame([{
    "policy_state": "GA",
    "policy_deductible": 400,
    "policy_annual_premium": 1430.78,
    "insured_age": 74,
    "insured_sex": "MALE",
    "insured_education_level": "High School",
    "insured_occupation": "Manager",
    "incident_type": "Parked Car",
    "collision_type": "Front",
    "incident_severity": "Total Loss",
    "authorities_contacted": "Police",
    "incident_state": "MI",
    "incident_hour_of_the_day": 6,
    "number_of_vehicles_involved": 1,
    "bodily_injuries": 4,
    "witnesses": 0,
    "police_report_available": "Yes",
    "claim_amount": 8161.36,
    "total_claim_amount": 11677.6
}])

# 2. Load Models & Preprocessors
model_bundle = joblib.load("final_models.pkl")
preprocess = joblib.load("preprocess.pkl")

# اختيار الموديل والعتبة
model = model_bundle["voting_model"]
threshold = model_bundle["thresholds"]["voting"]

# 3. Preprocessing

# معالجة القيم المفقودة 
new_data["authorities_contacted"] = new_data["authorities_contacted"].fillna("other")

# Label Encoding 
label_features = [col for col in preprocess["label_encoders"].keys() if col != 'fraud_reported']
for col in label_features:
    new_data[col] = preprocess["label_encoders"][col].transform(new_data[col])

label_df = new_data[label_features]

# One-Hot Encoding
ohe_parts = []
for col, enc in preprocess["onehot_encoders"].items():
    arr = enc.transform(new_data[[col]])
    cols = enc.get_feature_names_out([col])
    ohe_parts.append(pd.DataFrame(arr, columns=cols))

ohe_df = pd.concat(ohe_parts, axis=1)

# Numerical Columns
num_cols = [
    'policy_deductible', 'policy_annual_premium', 'insured_age',
    'incident_hour_of_the_day', 'number_of_vehicles_involved',
    'bodily_injuries', 'witnesses', 'claim_amount', 'total_claim_amount'
]

#تجميع البيانات
X_new = pd.concat([new_data[num_cols], ohe_df, label_df], axis=1)

#  Scaling
std_cols = [
    'policy_deductible', 'policy_annual_premium', 'insured_age',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'claim_amount', 'total_claim_amount'
]
minmax_cols = ['incident_hour_of_the_day']

X_new[std_cols] = preprocess["std_scaler"].transform(X_new[std_cols])
X_new[minmax_cols] = preprocess["minmax_scaler"].transform(X_new[minmax_cols])

# ترتيب الأعمدة بنفس ترتيب التدريب
expected_columns = [col for col in preprocess["columns"] if col != 'fraud_reported']
X_new = X_new[expected_columns]

# 4. Predict
proba = model.predict_proba(X_new)[:, 1]
pred = (proba >= threshold).astype(int)

print(f"Probability: {proba[0]:.4f}")
print(f"Prediction: {pred[0]}")

if pred[0] == 1:
    print("Result:  Fraud ")
else:
    print("Result:  Legit ")