import streamlit as st
import requests

# رابط الـ API الخاص بـ FastAPI
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Insurance Fraud Detection", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle Insurance Claim Fraud Detection")
st.write("أدخل تفاصيل بوليصة التأمين والحادث بالأسفل للتحقق مما إذا كانت المطالبة سليمة أم يُشتبه في كونها احتيال.")

# إنشاء نموذج الإدخال (Form)
with st.form("prediction_form"):
    st.subheader("📋 معلومات البوليصة والمؤمن عليه")
    col1, col2, col3 = st.columns(3)
    with col1:
        policy_state = st.selectbox("Policy State", ["OH", "IN", "IL", "GA", "MI", "NY"]) 
        policy_deductible = st.number_input("Policy Deductible", value=500.0)
        policy_annual_premium = st.number_input("Annual Premium", value=1100.50)
    with col2:
        insured_age = st.number_input("Insured Age", min_value=18, max_value=100, value=45)
        insured_sex = st.selectbox("Sex", ["MALE", "FEMALE"])
        insured_education_level = st.selectbox("Education Level", ["High School", "College", "Masters", "JD", "MD", "PhD", "Associate"])
    with col3:
        insured_occupation = st.selectbox("Occupation", ["Manager", "Prof-specialty", "Sales", "Exec-managerial", "Craft-repair", "Transport-moving", "Other-service", "Priv-house-serv", "Armed-Forces"])

    st.subheader(" تفاصيل الحادث والمطالبة")
    col4, col5, col6 = st.columns(3)
    with col4:
        incident_type = st.selectbox("Incident Type", ["Multi-vehicle Collision", "Single Vehicle Collision", "Vehicle Theft", "Parked Car"])
        collision_type = st.selectbox("Collision Type", ["Front Collision", "Rear Collision", "Side Collision"])
        incident_severity = st.selectbox("Incident Severity", ["Minor Damage", "Total Loss", "Major Damage", "Trivial Damage"])
        authorities_contacted = st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"])
    with col5:
        incident_state = st.selectbox("Incident State", ["NY", "SC", "WV", "VA", "NC", "PA", "OH"])
        incident_hour_of_the_day = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=14)
        number_of_vehicles_involved = st.number_input("Vehicles Involved", min_value=1, value=2)
        bodily_injuries = st.number_input("Bodily Injuries", min_value=0, value=0)
    with col6:
        witnesses = st.number_input("Witnesses", min_value=0, value=2)
        police_report_available = st.selectbox("Police Report Available", ["Yes", "No"])
        claim_amount = st.number_input("Claim Amount", value=1500.00)
        total_claim_amount = st.number_input("Total Claim Amount", value=2100.00)

    # زر الإرسا
    submit_button = st.form_submit_button("Predict (توقع النتيجة)")

if submit_button:
    # تجميع البيانات في Dictionary
    data = {
        "policy_state": policy_state,
        "policy_deductible": policy_deductible,
        "policy_annual_premium": policy_annual_premium,
        "insured_age": insured_age,
        "insured_sex": insured_sex,
        "insured_education_level": insured_education_level,
        "insured_occupation": insured_occupation,
        "incident_type": incident_type,
        "collision_type": collision_type,
        "incident_severity": incident_severity,
        "authorities_contacted": authorities_contacted,
        "incident_state": incident_state,
        "incident_hour_of_the_day": incident_hour_of_the_day,
        "number_of_vehicles_involved": number_of_vehicles_involved,
        "bodily_injuries": bodily_injuries,
        "witnesses": witnesses,
        "police_report_available": police_report_available,
        "claim_amount": claim_amount,
        "total_claim_amount": total_claim_amount
    }

    # إرسال البيانات للـ API
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            status = result['status']
            probability = result['probability'] * 100

            st.markdown("---")
            if status == "Legit":
                st.success(f" المطالبة تبدو سليمة (نسبة الاشتباه: {probability:.2f}%)")
            else:
                st.error(f" تحذير: احتمالية عالية للاحتيال! (النسبة: {probability:.2f}%)")
        else:
            st.error(f"حدث خطأ في الـ API: الكود {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("لم يتمكن التطبيق من الاتصال بالـ Backend. تأكد من تشغيل ملف (main.py) الخاص بـ FastAPI أولاً.")

