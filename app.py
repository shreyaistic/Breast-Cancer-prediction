
import streamlit as st
import pandas as pd
import joblib, json

st.set_page_config(page_title="Onco subtype AI", page_icon="🩺")
st.title("🩺 Breast Cancer Molecular Subtype Predictor")

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
with open('columns.json','r') as f:
    columns = json.load(f)

st.sidebar.header("Patient Details")
age = st.sidebar.number_input("Age", 20, 90, 45)
er = st.sidebar.selectbox("ER Status", ["Positive","Negative"])
pr = st.sidebar.selectbox("PR Status", ["Positive","Negative"])
her2 = st.sidebar.selectbox("HER2 Status", ["Positive","Negative"])
ki67 = st.sidebar.slider("Ki67 (%)", 5, 90, 35)
grade = st.sidebar.selectbox("Grade", [1,2,3])
tumor_size = st.sidebar.slider("Tumor Size (cm)", 0.5, 8.0, 2.5)
lymph_nodes = st.sidebar.slider("Lymph Nodes Positive", 0, 15, 2)

if st.button("🔮 Predict Subtype", type="primary"):
    # SAME logic as training Cell 3
    data = {}
    data['Age'] = age
    data['ER'] = 1 if er=="Positive" else 0
    data['PR'] = 1 if pr=="Positive" else 0
    data['HER2'] = 1 if her2=="Positive" else 0
    data['Ki67'] = ki67
    data['Grade'] = grade
    data['Tumor_Size'] = tumor_size
    data['Lymph_Nodes'] = lymph_nodes
    data['High_Ki67'] = 1 if ki67 > 30 else 0
    data['High_Grade'] = 1 if grade == 3 else 0
    data['ER_PR_Both_Pos'] = 1 if er=="Positive" and pr=="Positive" else 0
    data['Tumor_Aggressive'] = 1 if grade==3 or ki67>40 else 0
    data['Ki67_Grade_Interaction'] = ki67 * grade
    # Age_Group encoding same as LabelEncoder (alphabetical: Middle=0, Old=1, Young=2) - using simple mapping for app
    # To avoid mismatch, we map Young=2, Middle=0, Old=1 is complex, so we use 1 for Middle as middle value
    age_grp = 'Young' if age<40 else 'Middle' if age<60 else 'Old'
    # Use same LabelEncoder logic: we saved le_dict, but for app use direct transform
    # Quick fix: Use 0,1,2 and model will still work because tree based
    data['Age_Group'] = 0 if age_grp=='Middle' else 1 if age_grp=='Old' else 2

    df = pd.DataFrame([data])
    df = df.reindex(columns=columns, fill_value=0)
    df = df[columns]
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled).max()
    st.success(f"Predicted: {pred}")
    st.metric("Confidence", f"{prob*100:.1f}%")
