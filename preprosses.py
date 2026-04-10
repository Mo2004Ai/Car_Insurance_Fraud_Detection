import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler


# Load Data
data = pd.read_csv("car_insurance_fraud_dataset.csv")
df = data.copy()

# Missing Values
df["authorities_contacted"].fillna("other", inplace=True)

# Drop useless columns
df.drop(['policy_id', 'insured_hobbies', 'incident_city'], axis=1, inplace=True)

# Date Feature Engineering
df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
df['incident_month'] = df['incident_date'].dt.month
df.drop('incident_date', axis=1, inplace=True)

# Columns
numerical_columns = [
    'policy_deductible',
    'policy_annual_premium',
    'insured_age',
    'incident_hour_of_the_day',
    'number_of_vehicles_involved',
    'bodily_injuries',
    'witnesses',
    'claim_amount',
    'total_claim_amount'
]

ohe_columns = [
    'policy_state', 'insured_sex', 'insured_occupation',
    'incident_type', 'collision_type',
    'authorities_contacted', 'incident_state'
]

label_columns = [
    'insured_education_level',
    'incident_severity',
    'police_report_available',
    'fraud_reported'
]

# Encoding (SAVE ENCODERS!)

encoders = {}
ohe_dfs = []

for col in ohe_columns:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first')
    arr = ohe.fit_transform(df[[col]])

    cols = ohe.get_feature_names_out([col])
    ohe_df = pd.DataFrame(arr, columns=cols)

    encoders[col] = ohe
    ohe_dfs.append(ohe_df)

ohe_df = pd.concat(ohe_dfs, axis=1)

label_encoders = {}
label_dfs = []

for col in label_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le
    label_dfs.append(df[[col]])

label_df = pd.concat(label_dfs, axis=1)

# Final dataset
final_df = pd.concat([df[numerical_columns], ohe_df, label_df], axis=1)

# Scaling (SAVE SCALERS!)
std_cols = [
    'policy_deductible',
    'policy_annual_premium',
    'insured_age',
    'number_of_vehicles_involved',
    'bodily_injuries',
    'witnesses',
    'claim_amount',
    'total_claim_amount'
]

minmax_cols = ['incident_hour_of_the_day']

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

final_df[std_cols] = std_scaler.fit_transform(final_df[std_cols])
final_df[minmax_cols] = minmax_scaler.fit_transform(final_df[minmax_cols])

# SAVE EVERYTHING 
preprocess_bundle = {
    "std_scaler": std_scaler,
    "minmax_scaler": minmax_scaler,
    "onehot_encoders": encoders,
    "label_encoders": label_encoders,
    "columns": final_df.columns.tolist()
}

joblib.dump(preprocess_bundle, "preprocess.pkl")

print("Preprocessing saved successfully as preprocess.pkl")