import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import time
import pandas as pd
from datetime import datetime
import tempfile
from PIL import Image

# --- 1. PAGE SETUP & MASSIVE HEADING CSS ---
st.set_page_config(page_title="AI Fire Guard Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    
    /* MASSIVE BOLD WHITE MAIN TITLE */
    .main-title {
        text-align: center;
        color: #FFFFFF !important;
        font-size: 100px !important; 
        font-weight: 900 !important;
        text-transform: uppercase;
        margin-top: -60px;
        margin-bottom: 0px;
        text-shadow: 10px 10px 20px #000000; 
        font-family: 'Arial Black', Gadget, sans-serif;
        line-height: 1.2;
    }
    
    .sub-title {
        text-align: center;
        color: #00FFFF !important;
        font-size: 28px !important; 
        margin-top: -10px;
        margin-bottom: 60px;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: capitalize;
    }

    /* ANALYTICS BOXES & NUMBERS */
    [data-testid="stMetricValue"] {
        font-size: 60px !important; 
        font-weight: 800 !important;
        color: #FF4B4B !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 28px !important; 
        color: #FFFFFF !important;
    }
    
    .stMetric { 
        background-color: #1e2130; 
        padding: 40px !important; 
        border-radius: 20px; 
        border: 2px solid #333;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    }

    /* Sidebar Radio Button Font Size */
    .st-eb { font-size: 24px !important; }
    </style>
    """, unsafe_allow_html=True)


# --- 2. INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Area'])

if 'last_speech_time' not in st.session_state:
    st.session_state.last_speech_time = 0

@st.cache_resource
def load_models():
    return YOLO('yolov8n.pt'), YOLO('best.pt')

model_coco, model_fire = load_models()

def speak_warning_async(text):
    """Non-blocking Background Threaded Voice Warning"""
    def target():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=target, daemon=True).start()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("🔥 AI Guard Control")
app_slide = st.sidebar.radio("Navigate Dashboard", [
    "📊 Real-Time Analytics", 
    "🛡️ Risk & Safety Analysis", 
    "📁 Evidence & Logs"
])
input_source = st.sidebar.selectbox("Input Source", ["Webcam/Live", "Upload Evidence"])
conf_val = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.35) # Default lowered to catch critical fire easily

# --- 4. CENTERED HEADERS ---
st.markdown('<p class="main-title">AI FIRE GUARD PRO</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Intelligent Surveillance & Fire Spread Monitoring System</p>', unsafe_allow_html=True)

# --- 5. CORE AI ENGINE (RELIABLE REALTIME PARSING) ---
def process_ai(frame):
    res_h = model_coco(frame, classes=[0], conf=0.45, verbose=False)
    res_f = model_fire(frame, conf=conf_val, verbose=False)
    
    annotated = frame.copy()
    heatmap = np.zeros_like(frame)
    f_det, s_det, h_det = False, False, len(res_h[0].boxes) > 0
    f_area, s_density = 0, 0

    for box in res_f[0].boxes:
        c = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf_score = float(box.conf[0])
        
        w, h = (c[2] - c[0]), (c[3] - c[1])
        if cls == 0: # Fire
            f_det = True
            f_area += (w * h)
            color = (0, 0, 255) # Red for Fire
            # Continuous overlay rendering on the heatmap
            cv2.rectangle(heatmap, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 0, 255), -1)
        else: # Smoke
            s_det = True
            s_density += (w * h) / 1000
            color = (128, 128, 128) # Grey for Smoke
            
        # Draw bounding boxes and text percentages dynamically
        cv2.rectangle(annotated, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), color, 3)
        cv2.putText(annotated, f"{model_fire.names[cls].upper()} {conf_score:.0%}", 
                    (int(c[0]), int(c[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Human bounding boxes overlay
    for box in res_h[0].boxes:
        ch = box.xyxy[0].tolist()
        cv2.rectangle(annotated, (int(ch[0]), int(ch[1])), (int(ch[2]), int(ch[3])), (0, 255, 0), 2)
        cv2.putText(annotated, "HUMAN", (int(ch[0]), int(ch[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Fixed: Red Heatmap blend triggers without dropping frames
    if f_det:
        annotated = cv2.addWeighted(annotated, 0.65, heatmap, 0.35, 0)
            
    return annotated, f_det, s_det, h_det, f_area, s_density

# --- 6. LAYOUT UTILITIES ---
col_video, col_info = st.columns([2, 1])
st_frame = col_video.empty()

with col_info:
    m1 = st.empty()
    m2 = st.empty()
    st.markdown("---")
    panel_header = st.empty()
    panel_content = st.empty()

# --- 7. UNIFIED ANALYTICS RENDERING SYSTEM ---
def update_dashboard_ui(f_det, s_det, h_det, f_area, s_dens):
    """Handles real-time metric fluctuations and speech alerts instantly"""
    m1.metric("Fire Area", f"{int(f_area)} px²")
    m2.metric("Smoke Density", f"{int(s_dens)} units")
    
    if app_slide == "📊 Real-Time Analytics":
        panel_header.markdown("### 📈 Fire Spread Trend")
        new_data = pd.DataFrame({'Time': [datetime.now()], 'Area': [f_area]})
        st.session_state.history = pd.concat([st.session_state.history, new_data]).tail(20)
        panel_content.line_chart(st.session_state.history.set_index('Time')['Area'])

    elif app_slide == "🛡️ Risk & Safety Analysis":
        panel_header.markdown("### 🚨 Safety Monitor")
        risk = "CRITICAL" if (f_det and h_det) else "HIGH" if f_det else "SAFE"
        color = "red" if risk == "CRITICAL" else "orange" if risk == "HIGH" else "green"
        panel_content.markdown(f"## Risk Status: <span style='color:{color}'>{risk}</span><br><br><b>Human Presence:</b> {'YES' if h_det else 'NO'}", unsafe_allow_html=True)

    elif app_slide == "📁 Evidence & Logs":
        panel_header.markdown("### 📅 Incident Logs")
        if f_det:
            panel_content.error(f"🔥 Alert: Fire detected at {datetime.now().strftime('%H:%M:%S')}")
        else:
            panel_content.success("System normal. No anomaly logged.")

    # Optimized Alarm System: Trigger alarm every 3 seconds to avoid UI stuttering or thread locks
    if f_det:
        current_time = time.time()
        if current_time - st.session_state.last_speech_time > 3:
            msg = "Emergency! Human in danger." if h_det else "Warning! Fire detected."
            speak_warning_async(msg)
            st.session_state.last_speech_time = current_time

# --- 8. RUN LOGIC ---
def run_system():
    if input_source == "Webcam/Live":
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            proc, f_det, s_det, h_det, f_area, s_dens = process_ai(frame)
            update_dashboard_ui(f_det, s_det, h_det, f_area, s_dens)
            st_frame.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
            time.sleep(0.01)
        cap.release()
    else:
        uploaded = st.sidebar.file_uploader("Upload Image or Video Evidence", type=['mp4', 'avi', 'jpg', 'png', 'jpeg'])
        if not uploaded: return
        
        if uploaded.type.startswith('image'):
            img = Image.open(uploaded)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            proc, f_det, s_det, h_det, f_area, s_dens = process_ai(frame)
            st_frame.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
            update_dashboard_ui(f_det, s_det, h_det, f_area, s_dens)
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                proc, f_det, s_det, h_det, f_area, s_dens = process_ai(frame)
                update_dashboard_ui(f_det, s_det, h_det, f_area, s_dens)
                st_frame.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                time.sleep(0.01)
            cap.release()

if __name__ == "__main__":
    run_system()