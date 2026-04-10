import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler



#===================================================================================================

#تحميل البيانات 
data = pd.read_csv(r"D:\prgrames\program.py\programes\rwad\projeact2ml\car_insurance_fraud_dataset.csv")
df=data.copy()  # عمل نسخة من البيانات
#عرض البيانات

#=====================================================================================================

#تحليل البيانات
#EDX
print(df.head())  
#print(df.info())  # عندنا 24 عمود 
msno.matrix(df  , figsize=(10, 5) , fontsize=6)  # عرض البيانات المفقودة
#العمود دا الي ففيه قيم مفقوده  authorities_contacted 
plt.show()

print(df["authorities_contacted"].isnull().sum())      # عدد القيم المفقودة في كل عمود  7564 نسبه القيم المفقوده 25.5 %
print(df["authorities_contacted"].nunique()) #3
print(df["authorities_contacted"].value_counts())  # توزيع القيم في العمود دا

"""
Fire         7569
Police       7498
Ambulance    7369
"""

#تحديد انواع الاعمده 
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
categorical_columns = [
    'policy_id',# دا هيتحذف 
    'policy_state',#one-hot encoding
    'insured_sex',#one-hot encoding
    'insured_education_level',#label encoding
    'insured_occupation',#one-hot encoding
    'insured_hobbies', # دا هيتحذف
    'incident_date',# دا هيتحول لميعاد
    'incident_type',#one-hot encoding
    'collision_type',#one-hot encoding
    'incident_severity',#label encoding
    'authorities_contacted',#one-hot encoding
    'incident_state',#one-hot encoding
    'incident_city',#يتحذف
    'police_report_available',#label encoding
    'fraud_reported'#label encoding
]
#==========================================================================================================

#preprossing 

# التعامل  مع القيم المفوده 

df["authorities_contacted"].fillna("other" , inplace=True)  # في الاول هنملي باي كلمه مجرد حاجه نحسب عليها 
#print(df.groupby("authorities_contacted")["fraud_reported"].value_counts() ) # توزيع الاحتيال حسب الجهات المتصلة بها
"""
Ambulance              N                 6744  91.5%
                       Y                  625  8.5%
                       sum                7369

Fire                   N                 6935  91.6%
                       Y                  634  8.4%
                       sum                7569
Police                 N                 6915  92.2%
                       Y                  583  7.8%
                       sum                7498

other                  N                 5966  78.5%
                       Y                 1598  21.5%
                       sum               7564
"""
#الافضل نملئ القيم المفقوده بكلمه other لانها بتحتوي علي نسبه احتيال عاليه 21.5 %
print(df["authorities_contacted"].value_counts())  # توزيع القيم في العمود دا بعد التعامل مع القيم المفقوده
print(df.info())
msno.matrix(df  , figsize=(10, 5) , fontsize=6) # عرض البيانات المفقودة بعد التعامل معها
plt.show()

print(df['policy_state'].nunique())
print(df['insured_education_level'].nunique())
print(df['insured_occupation'].nunique())
print(df['incident_type'].nunique())
print(df['collision_type'].nunique())
print(df['incident_severity'].nunique())
print(df['incident_state'].nunique())    
print(df['incident_city'].nunique())

# الحذف الاعمده الغير مهمه

df.drop(['policy_id', 'insured_hobbies', 'incident_city'], axis=1, inplace=True)

#تحويل incident_date لميعاد

df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')

# استخراج الشهر  من incident_date

df['incident_month'] = df['incident_date'].dt.month 

# حذف العمود الاصلي بعد استخراج المعلومات

df.drop('incident_date', axis=1, inplace=True)  
print(df.info())

#=================================================================================================

#تحويل البيانات الي ارقام 

# الأعمدة التصنيفية للـ One-Hot
ohe_columns = [
    'policy_state', 'insured_sex', 'insured_occupation',
    'incident_type', 'collision_type', 'authorities_contacted', 'incident_state'
]

# الأعمدة للتصنيف بالـ Label
label_columns = [
    'insured_education_level', 'incident_severity', 'police_report_available', 'fraud_reported'
]

# ---- One-Hot Encoding ----
ohe_dataframes = []
for col in ohe_columns:
    onehot = OneHotEncoder(sparse_output=False, drop='first')
    encoded = onehot.fit_transform(df[[col]])  # لازم 2D
    col_names = onehot.get_feature_names_out([col])
    ohe_df = pd.DataFrame(encoded, columns=col_names)
    ohe_dataframes.append(ohe_df)

# دمج كل One-Hot DataFrames في واحد

ohe_df = pd.concat(ohe_dataframes, axis=1)

# ---- Label Encoding ----
label_dataframes = []
for col in label_columns:
    le = LabelEncoder()
    encoded = le.fit_transform(df[col]).reshape(-1,1)
    label_df = pd.DataFrame(encoded, columns=[col])
    label_dataframes.append(label_df)

label_df = pd.concat(label_dataframes, axis=1)

# ---- دمج كل شيء مع الأعمدة الرقمية ----
final_df = pd.concat([df[numerical_columns], ohe_df, label_df], axis=1)

print(final_df.info())


#scaling

# ---- StandardScaler للأعمدة الكبيرة أو المتباينة ----
std_scaler_columns = [
    'policy_deductible',
    'policy_annual_premium',
    'insured_age',
    'number_of_vehicles_involved',
    'bodily_injuries',
    'witnesses',
    'claim_amount',
    'total_claim_amount'
]

# ---- MinMaxScaler للـ incident_hour_of_the_day ----

minmax_scaler_columns = ['incident_hour_of_the_day']

# تطبيق StandardScaler

std_scaler = StandardScaler()
final_df[std_scaler_columns] = std_scaler.fit_transform(final_df[std_scaler_columns])

# تطبيق MinMaxScaler

minmax_scaler = MinMaxScaler()
final_df[minmax_scaler_columns] = minmax_scaler.fit_transform(final_df[minmax_scaler_columns])
#print(final_df[numerical_columns].head())
#print(final_df.head())
#print(final_df.info())

#=================================================================================================
#تحليل البيانات بعد المعالجة
#رسم كلاس احتيال ام لا 
plt.figure(figsize=(6,4))
final_df['fraud_reported'].value_counts().plot(kind='bar')
plt.title('Distribution of Fraud Reported')
plt.xlabel('Fraud Reported')
plt.ylabel('Count')     
plt.xticks(rotation=0)
plt.show()
#عندنا مشكلة عدم توازن  class0>>>>>>>>>>class1




