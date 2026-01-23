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
    "normal": {"precision": 0.94, "recall": 0.28, "f1": 0.43},
    "pneumonia": {"precision": 0.70, "recall": 0.99, "f1": 0.82},
    "accuracy": 0.72,
    "matrix": [[66, 168], [4, 386]]}

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="PNEUMA AI", page_icon="💠", layout="centered", initial_sidebar_state="collapsed")

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
    return tf.keras.models.load_model('pneumonia_model.keras')

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
with st.container():
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

    if st.button("CONFUSION MATRIX"):
        st.session_state.show_matrix = True
        st.session_state.show_biometrics = False 
        st.session_state.sidebar_state = "collapsed"
        st.rerun()

    if st.button("BIOMETRICS MAP"):
        st.session_state.show_biometrics = True
        st.session_state.show_matrix = False 
        st.session_state.sidebar_state = "collapsed"
        st.rerun()

# 7. CONFUSION MATRIX DISPLAY
if st.session_state.show_matrix:
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: white; font-weight: 100; letter-spacing: 10px; text-transform: uppercase; font-size: 1.25rem;'>CONFUSION MATRIX</h2>", unsafe_allow_html=True)

    z = MODEL_RESULTS["matrix"]
    x_labels = ["NORMAL", "PNEUMONIA"]
    y_labels = ["NORMAL", "PNEUMONIA"]
    colorscale = [[0, '#0a1a1a'], [1, '#00f2ff']]

    fig = ff.create_annotated_heatmap(
        z, x=x_labels, y=y_labels, 
        annotation_text=[[str(val) for val in row] for row in z],
        colorscale=colorscale,
        showscale=False
    )

    fig.data[0].opacity = 0.4  
    fig.data[0].xgap = 3      
    fig.data[0].ygap = 3

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis=dict(title=dict(text="P R E D I C T E D", font=dict(size=10, color='#00f2ff')), side="bottom"),
        yaxis=dict(title=dict(text="A C T U A L", font=dict(size=10, color='#00f2ff')), autorange="reversed")
    )

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 18
        fig.layout.annotations[i].font.color = 'white'
        fig.layout.annotations[i].font.family = "Inter"

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    col_x, col_center, col_y = st.columns([1, 1, 1])
    with col_center:
        if st.button("CLOSE MATRIX", use_container_width=True):
            st.session_state.show_matrix = False
            st.session_state.sidebar_state = "expanded"
            st.rerun()

# 8. BIOMETRIC ANALYSIS DISPLAY
if st.session_state.show_biometrics:
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: white; font-weight: 100; letter-spacing: 10px; text-transform: uppercase; font-size: 1.25rem;'>BIOMETRIC ANALYSIS</h2>", unsafe_allow_html=True)

    categories = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
    tn, fp, fn, tp = MODEL_RESULTS["matrix"][0][0], MODEL_RESULTS["matrix"][0][1], MODEL_RESULTS["matrix"][1][0], MODEL_RESULTS["matrix"][1][1]
    spec_val = tn / (tn + fp)

    pne_values = [
        MODEL_RESULTS["pneumonia"]["precision"], 
        MODEL_RESULTS["pneumonia"]["recall"], 
        MODEL_RESULTS["pneumonia"]["f1"],
        MODEL_RESULTS["accuracy"],
        spec_val
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pne_values, theta=categories, fill='toself',
        fillcolor='rgba(0, 242, 255, 0.2)',
        line=dict(color='#00f2ff', width=2),
        marker=dict(color='#00f2ff', size=8),
        name='Pneumonia Model'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            domain=dict(x=[0, 1], y=[0, 1]), 
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='rgba(255, 255, 255, 0.1)', ),
            angularaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='white', size=15, family="Inter"), )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=40, r=40), 
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    with col_c2:
        if st.button("EXIT BIOMETRICS", use_container_width=True):
            st.session_state.show_biometrics = False
            st.session_state.sidebar_state = "expanded"
            st.rerun()