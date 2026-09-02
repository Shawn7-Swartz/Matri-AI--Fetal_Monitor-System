import streamlit as st
import cv2
import random
import time
from PIL import Image
import numpy as np
import tempfile

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Matri AI : Fetal & Maternal Monitoring", page_icon="🤱", layout="wide")
st.title("🤱 Matri AI : Fetal & Maternal Health Monitoring")
st.markdown("Simulated **real-time fetal monitoring** using video playback with random predictions.")

tab1, tab2, tab3 = st.tabs([" Ultrasound Analysis", " Real-Time Monitoring", " Maternal Health"])

#================= TAB 1: Dummy Ultrasound Analysis =================
with tab1:
    st.subheader("Upload Ultrasound (Image or DICOM)")
    uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "dcm"])
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="Uploaded Ultrasound", use_container_width=True)
        st.info("✅ Simulated Prediction: Optimal (95%)")

#================= TAB 2: Real-Time Monitoring =================
#================= TAB 2: Real-Time Monitoring =================
with tab2:
    st.subheader("Real-Time Fetal Monitoring (Simulated Predictions)")

uploaded_file = st.file_uploader("Choose a video", type=["mp4"])

if uploaded_file:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Open video with OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    if not cap.isOpened():
        st.error("❌ Unable to open video.")
    else:
        stframe = st.empty()
        pred_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            
            # Display frame
            stframe.image(img, channels="RGB")
            
            # Simulate prediction
            simulated_pred = random.choice(["Optimal", "Non_Optimal"])
            simulated_conf = round(random.uniform(0.5, 1.0), 2)
            pred_text.markdown(f"**Prediction:** {simulated_pred}  |  **Confidence:** {simulated_conf*100:.1f}%")
            
            # Control video playback speed
            cv2.waitKey(30)
        
        cap.release()
        st.success("✅ Video playback finished.")

#================= TAB 3: Maternal Vitals =================
with tab3:
    st.subheader("Maternal Vitals Dashboard")
    bp = st.number_input("Blood Pressure (systolic mmHg)", min_value=70, max_value=200, value=120)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
    if st.button("Analyze Maternal Health"):
        alerts = []
        if bp > 140 or bp < 90:
            alerts.append("⚠️ Abnormal blood pressure detected.")
        if heart_rate > 110 or heart_rate < 60:
            alerts.append("⚠️ Abnormal heart rate detected.")
        if not alerts:
            alerts.append("✅ Maternal vitals are normal.")
        for alert in alerts:
            st.warning(alert) if "⚠️" in alert else st.success(alert)

st.markdown("---")
st.markdown("💗 *Matri AI: Simulated fetal and maternal monitoring demo.*")
