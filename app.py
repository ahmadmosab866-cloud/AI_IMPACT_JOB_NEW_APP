import streamlit as st
import pickle
import numpy as np

# 1. Models aur Data Load karein
@st.cache_resource
def load_resources():
    with open('job_risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('job_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('job_titles.pkl', 'rb') as f:
        job_titles = pickle.load(f)  # Yeh sirf aapki file wali list hai
    with open('job_stats.pkl', 'rb') as f:
        job_stats = pickle.load(f)
    return model, scaler, le, job_titles, job_stats

try:
    model, scaler, le, job_titles, job_stats = load_resources()
except FileNotFoundError:
    st.error("❌ Model files nahi milin! Pehle 'train_model.py' run karein.")
    st.stop()

# App UI
st.title("🤖 AI Job Impact Predictor 2030")
st.write("Aapki file mein majood jobs ki auto-complete list niche active hai:")

# 2. Search Box (Isme sirf aapki file ke names show honge)
selected_job = st.selectbox(
    "Job Title Type Ya Select Karein:",
    options=[""] + job_titles,
    format_func=lambda x: "🔍 Yahan type karein (e.g. Software, Data, Construction)..." if x == "" else x
)

# 3. Prediction Section
if selected_job != "":
    st.markdown(f"### Selected Job: **{selected_job}**")
    
    if st.button("Predict AI Impact"):
        # Job ka data fetch karna
        job_data = job_stats[selected_job]
        
        # Features array banana
        features = np.array([[
            job_data['Average_Salary'],
            job_data['Years_Experience'],
            job_data['AI_Exposure_Index'],
            job_data['Tech_Growth_Factor'],
            job_data['Automation_Probability_2030']
        ]])
        
        # Scale aur Predict karna
        scaled_features = scaler.transform(features)
        prediction_encoded = model.predict(scaled_features)[0]
        risk_status = le.inverse_transform([prediction_encoded])[0]
        
        # Result Output
        st.markdown("---")
        st.subheader(f"Analysis for: **{selected_job}**")
        
        if risk_status == 'Low':
            st.success(f"✅ **AI Risk Category: LOW**\n\nYeh job 2030 mein safe lag rahi hai. AI isme madadgaar sabit hoga.")
        elif risk_status == 'Medium':
            st.warning(f"🔄 **AI Risk Category: MEDIUM**\n\nIs job mein kafi tabdeeli ayegi. AI tools seekhna zaroori hai.")
        else:
            st.error(f"🚨 **AI Risk Category: HIGH**\n\nIs job par automation ka bohot zyada asar hone ka khadsha hai!")
            
        # Extra stats show karna
        st.write(f"📉 **Automation Probability:** {job_data['Automation_Probability_2030']*100:.1f}%")
        st.write(f"⚡ **AI Exposure Index:** {job_data['AI_Exposure_Index']:.2f}")
else:
    st.info("💡 *Upar diye gaye box mein kuch alphabets likhein, aapki excel file se milti-julti jobs niche list mein aa jayengi.*")
