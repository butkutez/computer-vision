import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from styles import CSS_CODE, HEADER_HTML, get_result_orb

# ==========================================
# MODEL DATA & GLOBAL SETTINGS
# Pre-calculated notebook results are stored in a dictionary 
# for centralized access across the UI components.
# ==========================================
MODEL_RESULTS = {
    "normal": {"precision": 0.91, "recall": 0.85, "f1": 0.88},
    "pneumonia": {"precision": 0.91, "recall": 0.95, "f1": 0.93},
    "accuracy": 0.91,
    "matrix": [[199, 35], [20, 370]]}

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="PNEUMA AI", page_icon="💠", layout="wide", initial_sidebar_state="collapsed")

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = "expanded"

if 'show_biometrics' not in st.session_state:
    st.session_state.show_biometrics = False

# 2. STYLING INJECTION
# Applies custom CSS from styles.py for the cinematic glassmorphism look.
st.markdown(CSS_CODE, unsafe_allow_html=True)

# 3. CORE LOGIC
# Loads the Keras model using caching to optimize memory usage.
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('COMPUTER-VISION/pneumonia_model.keras')

model = load_my_model()

# 4. UI HEADER
# Displays branding and animated subtitle imported from styles.py.
st.markdown("<h1 style='text-align: center; color: white; font-weight: 100; letter-spacing: 10px;'>PNEUMA AI</h1>", unsafe_allow_html=True)
st.markdown(HEADER_HTML, unsafe_allow_html=True)

if 'diagnosed' not in st.session_state:
    st.session_state.diagnosed = False
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

# 5. MAIN UPLOAD & DIAGNOSTIC SECTION
col_space_1, col_uploader, col_space_2 = st.columns([1, 1, 1])

with col_uploader:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Detects if a new file is uploaded to reset diagnostic state.
        if uploaded_file.name != st.session_state.last_file:
            st.session_state.diagnosed = False      
            st.session_state.last_file = uploaded_file.name 

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
        with col_btn_2:
            if not st.session_state.diagnosed:
                if st.button("INITIATE DIAGNOSTIC SEQUENCE"):
                    st.session_state.diagnosed = True
                    st.rerun() 

            if st.session_state.diagnosed:
                # Image preprocessing for model input (150x150).
                size = (150, 150)
                img_processed = ImageOps.fit(image, size)
                img_array = np.asarray(img_processed).astype('float32') / 255.0
                img_reshape = np.expand_dims(img_array, axis=0)
                
                # Executes inference.
                prediction = model.predict(img_reshape)
                confidence = prediction[0][0]

                # Result determination logic.
                orb_color = "#E63946" if confidence > 0.5 else "#00f2ff"
                result_text = "PNEUMONIA DETECTED" if confidence > 0.5 else "SYSTEM STATUS: NORMAL"
                conf_val = confidence*100 if confidence > 0.5 else (1-confidence)*100

                # Renders the result component.
                st.markdown(get_result_orb(orb_color, result_text, conf_val), unsafe_allow_html=True)   
    else:
        st.session_state.diagnosed = False
        st.session_state.last_file = None

if 'show_matrix' not in st.session_state:
    st.session_state.show_matrix = False               

# 6. SIDEBAR
with st.sidebar:
    st.markdown("## SYSTEM METRICS")
    st.markdown("---")
    st.markdown("### PERFORMANCE") 
    c1, c2 = st.columns(2)

    rec_val = f"{MODEL_RESULTS['pneumonia']['recall']*100:.0f}%"
    pre_val = f"{MODEL_RESULTS['pneumonia']['precision']*100:.0f}%"

    c1.metric("Recall", rec_val)
    c2.metric("Precision", pre_val)
    st.markdown("---")
    st.write("**Model:** CNN-ResNet")
    acc_val = f"{MODEL_RESULTS['accuracy']*100:.0f}%"
    st.write(f"**Accuracy:** {acc_val}")
    st.markdown("---")

    if st.button("VIEW SYSTEM ANALYTICS"):
        st.session_state.show_analytics = True
        st.session_state.sidebar_state = "collapsed"
        st.rerun()

# 7. MERGED ANALYTICS DISPLAY (FINAL BALANCED VERSION)
if st.session_state.get('show_analytics', False):
    st.markdown("---")
    
    # MAIN HEADER
    st.markdown("<h2 style='text-align: center; color: white; font-weight: 100; letter-spacing: 10px; text-transform: uppercase;'>SYSTEM ANALYTICS</h2>", unsafe_allow_html=True)

    col_left, spacer, col_right = st.columns([4, 0.5, 4])

    with col_left:
        # SUB-TITLE
        st.markdown("<h3 style='text-align: center; color: white; font-weight: 100; letter-spacing: 5px; text-transform: uppercase; font-size: 1rem;'>CONFUSION MATRIX</h3>", unsafe_allow_html=True)
        
        # --- CONFUSION MATRIX ---
        z = MODEL_RESULTS["matrix"]
        fig_matrix = ff.create_annotated_heatmap(
            z, x=["NORMAL", "PNEUMONIA"], y=["NORMAL", "PNEUMONIA"], 
            colorscale=[[0, '#0a1a1a'], [1, '#00f2ff']],
            showscale=False
        )
        fig_matrix.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(t=20, b=100, l=100, r=20),
            xaxis=dict(tickfont=dict(size=14, color='white'), title=dict(text="P R E D I C T E D", font=dict(size=15, color='#00f2ff'))),
            yaxis=dict(tickfont=dict(size=14, color='white'), title=dict(text="A C T U A L", font=dict(size=15, color='#00f2ff')), autorange="reversed")
        )

        for i in range(len(fig_matrix.layout.annotations)):
            fig_matrix.layout.annotations[i].font.size = 20
            fig_matrix.layout.annotations[i].font.color = 'white'

        st.plotly_chart(fig_matrix, use_container_width=True, config={'displayModeBar': False})

        # EXPLANATION TEXT (LEFT BOX - General Matrix Info)
        st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 25px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(15px); min-height: 280px;'>
            <p style='color: #00f2ff; font-size: 0.9rem; letter-spacing: 2px; margin-bottom: 10px; font-weight: bold;'>DIAGNOSTIC ARCHITECTURE</p>
            <p style='color: white; font-size: 15px; opacity: 0.9; line-height: 1.6;'>
                A <b>Confusion Matrix</b> is the primary tool for evaluating AI performance. It maps the relationship between predicted outcomes and actual clinical truths. 
                <br><br>
                The vertical axis shows the ground truth (Actual), while the horizontal axis shows what the AI predicted. 
                High values in the diagonal boxes indicate a high-performing model, while values in the off-diagonal areas represent diagnostic errors (False Positives and False Negatives).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # SUB-TITLE
        st.markdown("<h3 style='text-align: center; color: white; font-weight: 100; letter-spacing: 5px; text-transform: uppercase; font-size: 1rem;'>BIOMETRIC METRICS</h3>", unsafe_allow_html=True)

        # --- BIOMETRIC RADAR ---
        categories = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
        tn, fp, fn, tp = MODEL_RESULTS["matrix"][0][0], MODEL_RESULTS["matrix"][0][1], MODEL_RESULTS["matrix"][1][0], MODEL_RESULTS["matrix"][1][1]
        pne_values = [MODEL_RESULTS["pneumonia"]["precision"], MODEL_RESULTS["pneumonia"]["recall"], MODEL_RESULTS["pneumonia"]["f1"], MODEL_RESULTS["accuracy"], tn/(tn+fp)]
        
        categories_closed = categories + [categories[0]]
        values_closed = pne_values + [pne_values[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values_closed, theta=categories_closed, fill='toself',
            fillcolor='rgba(0, 242, 255, 0.2)',
            line=dict(color='#00f2ff', width=3)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255, 255, 255, 0.1)', showticklabels=False),
                angularaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='white', size=14))
            ),
            showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(t=20, b=80, l=80, r=80)
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

        # EXPLANATION TEXT (RIGHT BOX - Definitions)
        st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 25px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(15px); min-height: 280px;'>
            <p style='color: #00f2ff; font-size: 0.9rem; letter-spacing: 2px; margin-bottom: 10px; font-weight: bold;'>METRIC PARAMETERS</p>
            <p style='color: white; font-size: 15px; opacity: 0.9; line-height: 1.6;'>
                The <b>Biometric Radar</b> provides a geometric representation of model stability. 
                A larger, more symmetrical shape indicates a well-balanced model.
                <br><br>
                <b>Key Metric Definitions:</b>
            </p>
            <style>
                /* This specifically targets the bullet point dots */
                .white-bullets li::marker {
                    color: white !important;
                }
            </style>
            <ul class="white-bullets" style='color: white; font-size: 15px; padding-left: 20px;'>
                <li style='margin-bottom: 8px;'><b>Accuracy:</b> Overall correctness across all diagnostic categories.</li>
                <li style='margin-bottom: 8px;'><b>Precision:</b> The percentage of pneumonia detections that were actually correct.</li>
                <li style='margin-bottom: 8px;'><b>Recall:</b> The system's ability to find and identify all pneumonia cases present.</li>
                <li style='margin-bottom: 8px;'><b>F1-Score:</b> The harmonic mean providing a balance between Precision and Recall.</li>
                <li><b>Specificity:</b> The ability to correctly rule out healthy lungs as 'Normal'.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # CENTERED CLOSE BUTTON
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn_center, _ = st.columns([1, 1, 1])
    with col_btn_center:
        if st.button("CLOSE ANALYTICS", use_container_width=True):
            st.session_state.show_analytics = False
            st.session_state.sidebar_state = "expanded"
            st.rerun()