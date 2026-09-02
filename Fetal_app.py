import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import json
import time
import pydicom
import matplotlib.pyplot as plt
import io
import random
import tempfile
import tensorflow as tf
tf.keras.backend.clear_session()
from ultralytics import YOLO


@st.cache_resource
def load_all_models():
    model1 = load_model("fetal_position_model.h5")  
    model2 = load_model("fetal_vgg16_model.h5")     
    with open("label_map.json", "r") as f:
        label_map = json.load(f)
    return model1, model2, label_map

@st.cache_resource
def load_model1():
    return load_model("fetal_position_model.h5", compile=False)

@st.cache_resource
def load_model2():
    return load_model("fetal_vgg16_model.h5", compile=False)

@st.cache_resource
def load_head_detector():
    return YOLO("best.pt")

head_model = load_head_detector()
model1 = load_model1()
model2 = load_model2()

with open("label_map.json", "r") as f:
    label_map = json.load(f)


# ---------- Prediction ----------
def predict_fetal_position(model, img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)

    if pred.shape[-1] == 1:
        conf = float(pred[0][0])
        class_name = "Optimal" if conf > 0.5 else "Non_Optimal"
        conf_adj = conf if class_name == "Optimal" else 1 - conf
    else:
        class_index = np.argmax(pred)
        class_name = "Optimal" if class_index == 1 else "Non_Optimal"
        conf_adj = float(np.max(pred))
    return class_name, conf_adj


# ---------- Grad-CAM ----------
def generate_gradcam(model, img, layer_name=None):
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if layer_name is None:
        layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]

    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs[0].shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[0][:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img = np.uint8(255 * img_array[0])
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img


# ---------- Alerts ----------
def fetal_alert(pred_class):
    if pred_class.lower() == "optimal":
        return "✅ Fetal is in an OPTIMAL position (Head Down)"
    elif pred_class.lower() == "non_optimal":
        return "⚠️ Non-optimal fetal position detected — Possible Breech or Transverse lie."
    else:
        return "⚠️ Unable to determine fetal position confidently."

def non_optimal_guidelines():
    st.markdown("### ⚕️ Clinical Guidance for Non-Optimal Fetal Positions:")
    st.markdown("""
    **1. Breech Position:**  
    • Higher chance of **C-section delivery**.  
    • **Prevention:** Pelvic tilt exercises, gentle prenatal yoga.  
    • **Monitoring:** Regular ultrasound to observe fetal rotation.

    **2. Transverse Position:**  
    • **Procedure:** *External Cephalic Version (ECV)* —  
      a safe manual method to reposition the baby (36–37 weeks).  
    • **Precautions:** Stay hydrated, rest, and avoid stress.  
    """)


def maternal_alert(bp, heart_rate):
    alerts = []
    if bp > 140 or bp < 90:
        alerts.append("⚠️ Abnormal blood pressure detected.")
    if heart_rate > 110 or heart_rate < 60:
        alerts.append("⚠️ Abnormal heart rate detected.")
    return alerts if alerts else ["✅ Maternal vitals are normal."]

def compute_contraction_level(prev_gray, gray, bbox):
    x, y, w, h = bbox

    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_curr = gray[y:y+h, x:x+w]

    if roi_prev.size == 0 or roi_curr.size == 0:
        return 45.0  

    diff = cv2.absdiff(roi_prev, roi_curr)

    variation = np.mean(diff)

    # Normalize to physiological-looking pressure
    contraction_level = 40 + min(variation / 8, 20)

    # Add natural flicker 
    contraction_level += np.random.normal(0, 0.6)

    
    contraction_level = max(40, min(contraction_level, 60))

    return float(contraction_level)

def generate_ctg_signals(contraction_level, t):
    """
    Generates realistic CTG-like signals
    """

    # --- Fetal Heart Rate (baseline + variability)
    baseline_fhr = 140

    variability = 5 * np.sin(0.5 * t) + np.random.normal(0, 1.5)

    # mild response to contractions
    contraction_effect = (contraction_level - 50) * 0.2

    fhr = baseline_fhr + variability - contraction_effect
    fhr = np.clip(fhr, 110, 170)

    # --- Uterine Contraction waveform
    uc_wave = contraction_level + 2 * np.sin(0.8 * t)
    uc_wave = np.clip(uc_wave, 0, 100)

    return fhr, uc_wave


def plot_ctg(fhr_values, uc_values):
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))

    # FHR Trace
    ax[0].plot(fhr_values)
    ax[0].set_ylim(100, 180)
    ax[0].set_title("Fetal Heart Rate (bpm)")
    ax[0].set_ylabel("bpm")
    ax[0].grid(True)

    # UC Trace
    ax[1].plot(uc_values)
    ax[1].set_ylim(0, 100)
    ax[1].set_title("Uterine Contractions (mmHg)")
    ax[1].set_ylabel("mmHg")
    ax[1].grid(True)

    plt.tight_layout()
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Matri AI : Fetal & Maternal Monitoring", page_icon="🤱", layout="wide")
st.title("🤱 Matri AI : Fetal & Maternal Health Monitoring")
st.markdown("Perform **real-time fetal monitoring** or **analyze ultrasound (DICOM/Image)** for AI-driven insights.")
st.html("""
<style>
body { background-color:#0b0f17; color:#e6edf3; }

h1,h2,h3 { color:#7dd3fc !important; }

[data-testid="stMetric"]{
 background:linear-gradient(145deg,#0f172a,#020617);
 border:1px solid #1e293b;
 padding:15px;
 border-radius:14px;
 box-shadow:0 0 15px rgba(0,255,255,0.15);
 text-align:center;
}

.stTabs [role="tab"]{
 font-size:18px;
 padding:10px;
}

img{
 border-radius:14px;
 border:2px solid #1f2937;
}

section[data-testid="stSidebar"]{
 background-color:#020617;
}

</style>
""")

st.markdown("<hr style='border:1px solid #1f2937'>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([" Ultrasound Analysis", " Real-Time Monitoring", " Maternal Health"])

# ========== TAB 1 ==========
with tab1:
    st.subheader("Upload Ultrasound (Image or DICOM)")
    uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png", "dcm"])

    if uploaded_file:
        if uploaded_file.name.endswith(".dcm"):
            dicom_data = pydicom.dcmread(uploaded_file)
            img = dicom_data.pixel_array
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = Image.fromarray(np.uint8(img)).convert("RGB")
        else:
            img = Image.open(uploaded_file).convert("RGB")

        st.image(img, caption="Uploaded Ultrasound", use_container_width=True)

        pred1, conf1 = predict_fetal_position(model1, img)
        pred2, conf2 = predict_fetal_position(model2, img)

        # ✅ High confidence prediction logic
        if pred1.lower() == "optimal" and conf1 >= 0.70:
            final_pred, final_conf = pred1, conf1
            st.markdown("### 🧩 Prediction 1 (High Confidence)")
            st.write(f"**Class:** {final_pred}")
            st.write(f"**Confidence:** {final_conf*100:.2f}%")
        else:
            final_pred, final_conf = (pred1, conf1) if conf1 >= conf2 else (pred2, conf2)
            st.markdown("### 🧩 Final Prediction")
            st.write(f"**Class:** {final_pred}")
            st.write(f"**Confidence:** {final_conf*100:.2f}%")

        st.info(fetal_alert(final_pred))
        if final_pred == "Non_Optimal":
            non_optimal_guidelines()

 


with tab2:
    st.subheader("Real-Time Fetal Monitoring (Simulated Predictions)")

    uploaded_file = st.file_uploader("Choose a video", type=["mp4"])
    ctg_fhr = []
    ctg_uc = []
    t = 0

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("❌ Unable to open video.")
        else:
            col1,col2 =st.columns([1.2,1])
            with col1:
                stframe = st.empty()
            with col2:
                chart = st.empty()
                pred_text = st.empty()
            
            
            contraction_values = []
            prev_gray = None
            contraction_level = 45.0  
            metric1, metric2, metric3 = st.columns([1,1,1], gap="large")
            pressure_card = metric1.empty()
            confidence_card = metric2.empty()
            status_card = metric3.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:

                    results = head_model(frame, conf=0.3)[0]

                    if len(results.boxes) > 0:
                        box = results.boxes[0].xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)

                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0,255,255), 2)
                        cv2.putText(frame_rgb, "Fetal Head", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                        bbox = (x1, y1, x2-x1, y2-y1)

                        contraction_level = compute_contraction_level(prev_gray, gray, bbox)

                    else:
                        contraction_level = 45.0
                   
                fhr, uc_val = generate_ctg_signals(contraction_level, t)

                ctg_fhr.append(fhr)
                ctg_uc.append(uc_val)

                fig = plot_ctg(ctg_fhr[-120:], ctg_uc[-120:])
                chart.pyplot(fig)

                t += 0.1
                
                prev_gray = gray
                img = Image.fromarray(frame_rgb)
                stframe.image(img, channels="RGB")

                simulated_pred = "Optimal" if contraction_level < 55 else "Non_Optimal"

                simulated_conf = 0.70 + abs(50 - contraction_level) / 50
                simulated_conf = min(simulated_conf, 0.95)
                pressure_card.metric("Uterine Pressure", f"{contraction_level:.1f} mmHg", "Live")
                confidence_card.metric("Model Confidence", f"{simulated_conf*100:.1f}%", "AI")
                status_card.metric("Fetal Status", simulated_pred)

                if contraction_level > 58:
                    alert_msg = "⚠ Excessive Uterine Contraction Detected"
                    st.warning(alert_msg)
                else:
                    alert_msg = "✅ Uterine Activity Normal"

                pred_text.markdown(
                    f"**Prediction:** {simulated_pred} | "
                    f"**Confidence:** {simulated_conf*100:.1f}% | "
                    f"**Uterine Activity Index:** {contraction_level:.2f} mmHg"
                )

                time.sleep(0.03)

            cap.release()
            st.success("✅ Video playback finished.")


# ========== TAB 3 ==========
with tab3:
    st.subheader("Maternal Vitals Dashboard")
    bp = st.number_input("Blood Pressure (systolic mmHg)", min_value=70, max_value=200, value=120)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
    if st.button("Analyze Maternal Health"):
        alerts = maternal_alert(bp, heart_rate)
        for alert in alerts:
            st.warning(alert) if "⚠️" in alert else st.success(alert)


st.markdown("---")
st.markdown("💗 *Matri AI: A holistic AI platform for fetal and maternal monitoring.*")
