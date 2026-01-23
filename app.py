import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Update these numbers whenever you retrain your model in your Notebook
MODEL_RESULTS = {
    "normal": {"precision": 0.94, "recall": 0.28, "f1": 0.43},
    "pneumonia": {"precision": 0.70, "recall": 0.99, "f1": 0.82},
    "accuracy": 0.72,
    "matrix": [[66, 168], [4, 386]]
}

# 1. PAGE CONFIG
st.set_page_config(page_title="PNEUMA AI", page_icon="💠", layout="centered", initial_sidebar_state="collapsed")

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = "expanded"

if 'show_biometrics' not in st.session_state:
    st.session_state.show_biometrics = False

# 2. CINEMATIC FOREST & GLASSMORPHISM CSS
st.markdown("""
    <style>
    /* Background Image: Lungs of the World (Trees) */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                    url('https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1920&q=80');
        background-attachment: fixed;
        background-size: cover;
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Frame for X-ray */
    .stImage {
        border: 8px solid rgba(255, 255, 255, 0.1);
        border-radius: 0px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.8);
        padding: 5px;
        background-color: #000;
    }
            
    /* 2. Target the Actual Image Inside */
    .stImage img {
        border-radius: 0px !important; /* Sharp corners for the X-ray itself */
    }

    /* 1. Content Container (Light background) */
    .main-glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px;
    }

    /* 2. Fix File Uploader Text & Button */
    .stFileUploader label, .stFileUploader p, .stFileUploader span {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Target the 'Browse files' button specifically */
    .stFileUploader button {
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: #00f2ff !important;
        border: 1px solid #00f2ff !important;
        transition: 0.3s !important;
    }

    .stFileUploader button:hover {
        background-color: rgba(0, 242, 255, 0.2) !important;
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.4) !important;
    }
    
    /* Redesigned Drag & Drop Area */
    .stFileUploader section {
        background-color: rgba(0, 0, 0, 0.4) !important;
        border: 2px dashed #00f2ff !important;
        border-radius: 20px !important;
        padding: 30px !important;
    }

    /* Elegant Result Typography */
    .status-text {
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 200;
        text-align: center;
    }

    /* Style the buttons to be sleek and semi-transparent */
    .stButton>button {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 50px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: #00f2ff;
    }
            
    /* Fix visibility for the uploaded file name and size text */
    [data-testid="stFileUploaderFileName"], 
    [data-testid="stFileUploaderFileData"] {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 400 !important;
    }

    /* Optional: Fix the 'X' (remove) icon color next to the filename */
    .stFileUploader [data-testid="stBaseButton-secondary"] {
        color: white !important;
    }

    /* 1. The Result Shield Container */
    .result-shield {
        padding: 30px;
        border-radius: 20px;
        margin-top: 30px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.5s ease;
    }

    /* 2. Detected Style (Soft Ruby) */
    .status-positive {
        background: rgba(230, 57, 70, 0.15); /* Soft ruby tint */
        border-bottom: 4px solid #E63946;
        color: #FFAAAA;
        box-shadow: 0 10px 30px rgba(230, 57, 70, 0.2);
    }

    /* 3. Normal Style (Soft Emerald) */
    .status-negative {
        background: rgba(0, 242, 255, 0.1); /* Soft cyan/emerald tint */
        border-bottom: 4px solid #00f2ff;
        color: #AAFFFF;
        box-shadow: 0 10px 30px rgba(0, 242, 255, 0.2);
    }

    /* 4. Confidence Typography */
    .conf-label {
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 10px;
    }      
    
    /* Centered Floating Orb Container */
    .orb-container {
        display: flex;
        flex-direction: column; /* Stack orb above text */
        align-items: center;
        justify-content: center;
        gap: 15px;
        background: rgba(255, 255, 255, 0.08);
        padding: 40px;
        border-radius: 30px;
        margin: 40px auto; /* Centers the box itself */
        border: 1px solid rgba(255, 255, 255, 0.15);
        max-width: 400px; /* Keeps the box from getting too wide */
    }

    .orb {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        filter: blur(10px);
        animation: breathe 3s infinite ease-in-out;
    }

    .result-title {
        margin: 0;
        color: white;
        font-weight: 200;
        letter-spacing: 2px;
        text-align: center;
    }

    .confidence-white {
        margin: 0;
        color: #FFFFFF !important; /* Pure white */
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        text-shadow: 0 0 15px rgba(255,255,255,0.3);
    }

    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
    }
            
    /* Sidebar Glassmorphism Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

   /* Sidebar Heading Styling - Pure White Fix */
    [data-testid="stSidebar"] .stMarkdown h3, 
    [data-testid="stSidebar"] .stMarkdown h3 span,
    [data-testid="stSidebar"] .stMarkdown h3 label {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important; /* Forces color on some browsers */
        font-weight: 300 !important;
        letter-spacing: 2px;
        margin-bottom: -10px;
    }
    
    /* Sidebar Body Text - High Visibility */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stMarkdown strong {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 300;
    }

    /* Make sidebar metrics white and clean */
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.5rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.6) !important;
    }
            
    /* 1. Main Titles (## SYSTEM METRICS) */
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #FFFFFF !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 300 !important;
    }

    /* 2. Sub-headers (### PERFORMANCE) */
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFFFFF !important;
        font-weight: 300 !important;
        letter-spacing: 1px;
    }

    /* 3. Standard Text (Model, Accuracy) */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #FFFFFF !important;
    }

    /* 4. Expander Header & Text (VIEW CONFUSION MATRIX) */
    [data-testid="stSidebar"] .st-expanderHeader,
    [data-testid="stSidebar"] summary p {
        color: #FFFFFF !important;
    }

    /* 5. Expander Arrow Icon */
    [data-testid="stSidebar"] .st-expanderIcon,
    [data-testid="stSidebar"] summary svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    /* Style for the Confusion Matrix Table */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Style for the button in the sidebar */
    div[data-testid="stSidebar"] button {
        width: 100%;
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Styling for the centered Exit Button to match the theme */
    div[data-testid="stColumn"] button {
        border-radius: 20px !important;
        border: 1px solid rgba(0, 242, 255, 0.4) !important;
        letter-spacing: 2px !important;
        font-weight: 300 !important;
    }

    # 7 Second biometric
    /* Center the table content and text */
    .stTable {
        margin-left: auto;
        margin-right: auto;
        background-color: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Styling for the centered Exit Button to match the theme */
    div[data-testid="stColumn"] button {
        border-radius: 20px !important;
        border: 1px solid rgba(0, 242, 255, 0.4) !important;
        letter-spacing: 2px !important;
        font-weight: 300 !important;
    }         

/* Fix Sidebar Metrics (Recall/Precision) to be White */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        color: rgba(255, 255, 255, 0.9) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

/* Update your CSS block with this line */
div[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 22px !important;
    padding: 10px;
    backdrop-filter: blur(5px);
    
    /* THIS REMOVES THE SIDE SCROLL */
    overflow: hidden !important; 
}

    </style>
    """, unsafe_allow_html=True)


# 3. CORE LOGIC (Essential parts preserved)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('pneumonia_model.keras')

model = load_my_model()

# 4. CENTRALIZED UI WRAPPER
# We use a container-like div to center the content
st.markdown("<h1 style='text-align: center; color: white; font-weight: 100; letter-spacing: 10px;'>PNEUMA AI</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style="width: 100%; display: flex; justify-content: center;">
        <p style='
            text-align: center; 
            color: #00f2ff; 
            font-size: 0.8rem; 
            font-family: "Inter", sans-serif;
            letter-spacing: 5px; 
            margin-bottom: 40px; 
            text-transform: uppercase;
            width: 100%;
        '>
            ADVANCED BIOMETRIC IMAGING
        </p>
    </div>
    """, unsafe_allow_html=True)


# 1. Initialize memory at the top of your script
if 'diagnosed' not in st.session_state:
    st.session_state.diagnosed = False
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

# 4. CENTRALIZED UI WRAPPER
with st.container():
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Check if the file has changed or was removed/re-added
        if uploaded_file.name != st.session_state.last_file:
            st.session_state.diagnosed = False      # Reset the "switch"
            st.session_state.last_file = uploaded_file.name # Update memory

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])

        with col_btn_2:
            # ONLY show the button if we are NOT in "diagnosed" mode
            if not st.session_state.diagnosed:
                if st.button("INITIATE DIAGNOSTIC SEQUENCE"):
                    st.session_state.diagnosed = True
                    st.rerun() 

            # ONLY show results if the button WAS pressed
            if st.session_state.diagnosed:
                # --- PREPROCESSING & PREDICTION (Keep your existing lines) ---
                size = (150, 150)
                img_processed = ImageOps.fit(image, size)
                img_array = np.asarray(img_processed).astype('float32') / 255.0
                img_reshape = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_reshape)
                confidence = prediction[0][0]

                orb_color = "#E63946" if confidence > 0.5 else "#00f2ff"
                result_text = "PNEUMONIA DETECTED" if confidence > 0.5 else "SYSTEM STATUS: NORMAL"
                conf_val = confidence*100 if confidence > 0.5 else (1-confidence)*100

                st.markdown(f"""
                    <div class="orb-container">
                        <div class="orb" style="background: {orb_color}; box-shadow: 0 0 30px {orb_color};"></div>
                        <div>
                            <p class="result-title">{result_text}</p>
                            <p class="confidence-white">{conf_val:.2f}%</p>
                            <p style="text-align:center; color:rgba(255,255,255,0.85); font-size:0.75rem; letter-spacing: 2px; margin:0; font-weight:300;">AI CONFIDENCE SCORE</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)   
    else:
        # If the user removes the photo, we reset everything
        st.session_state.diagnosed = False
        st.session_state.last_file = None


# 1. Initialize State (at the top with your other states)
if 'show_matrix' not in st.session_state:
    st.session_state.show_matrix = False               

# 6. SIDEBAR
with st.sidebar:
    st.markdown("## SYSTEM METRICS")
    st.markdown("---")
    st.markdown("### PERFORMANCE") 
    c1, c2 = st.columns(2)
    c1.metric("Recall", "96%")
    c2.metric("Precision", "80%")
    st.markdown("---")
    st.write("**Model:** CNN-ResNet")
    st.write("**Accuracy:** 72%")
    st.markdown("---")

    # Button 1: Confusion Matrix
    if st.button("CONFUSION MATRIX"):
        st.session_state.show_matrix = True
        st.session_state.show_biometrics = False # Close other view
        st.session_state.sidebar_state = "collapsed"
        st.rerun()

    # Button 2: Biometrics (Same Style)
    if st.button("BIOMETRICS MAP"):
        st.session_state.show_biometrics = True
        st.session_state.show_matrix = False # Close other view
        st.session_state.sidebar_state = "collapsed"
        st.rerun()

# 7. MAIN SCREEN DISPLAY
if st.session_state.show_matrix:
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: white; font-weight: 100; letter-spacing: 10px; text-transform: uppercase; font-size: 1.25rem;'>CONFUSION MATRIX</h2>", unsafe_allow_html=True)

    # DATA
    z = [[66, 168], [4, 386]]
    x_labels = ["NORMAL", "PNEUMONIA"]
    y_labels = ["NORMAL", "PNEUMONIA"]

    # 1. USE HEX COLORS (To avoid the rgba int() error)
    # We use a dark background hex to simulate the "light" look
    colorscale = [[0, '#0a1a1a'], [1, '#00f2ff']]

    fig = ff.create_annotated_heatmap(
        z, x=x_labels, y=y_labels, 
        annotation_text=[[str(val) for val in row] for row in z],
        colorscale=colorscale,
        showscale=False
    )

    # 2. APPLY TRANSPARENCY TO THE TRACE DIRECTLY
    # This bypasses the figure_factory limitation
    fig.data[0].opacity = 0.4  # Adjust this (0.1 to 0.9) for "lightness"
    fig.data[0].xgap = 3      # Adds the clean grid gaps
    fig.data[0].ygap = 3

    # 3. STYLE THE LAYOUT
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis=dict(title=dict(text="P R E D I C T E D", font=dict(size=10, color='#00f2ff')), side="bottom"),
        yaxis=dict(title=dict(text="A C T U A L", font=dict(size=10, color='#00f2ff')), autorange="reversed")
    )

    # Clean up annotation fonts (Unbolded)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 18
        fig.layout.annotations[i].font.color = 'white'
        fig.layout.annotations[i].font.family = "Inter"

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # --- CENTERED BUTTON ---
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

    # Automated Metrics from your MODEL_RESULTS
    categories = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
    
    # Mapping the values from your dictionary
    # (Note: Specificity is usually TN / (TN + FP))
    tn, fp, fn, tp = MODEL_RESULTS["matrix"][0][0], MODEL_RESULTS["matrix"][0][1], MODEL_RESULTS["matrix"][1][0], MODEL_RESULTS["matrix"][1][1]
    spec_val = tn / (tn + fp)

    pne_values = [
        MODEL_RESULTS["pneumonia"]["precision"], 
        MODEL_RESULTS["pneumonia"]["recall"], 
        MODEL_RESULTS["pneumonia"]["f1"],
        MODEL_RESULTS["accuracy"],
        spec_val
    ]

    # Create the Radar Chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=pne_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 242, 255, 0.2)',
        line=dict(color='#00f2ff', width=2),
        marker=dict(color='#00f2ff', size=8),
        name='Pneumonia Model'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            # Keep the radar web centered within the large box
            domain=dict(x=[0, 1], y=[0, 1]), 
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor='rgba(255, 255, 255, 0.1)',
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                # Your 15pt labels now have plenty of room
                tickfont=dict(color='white', size=15, family="Inter"),
            )
        ),
        
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # Large margins to account for the wider box width
        margin=dict(t=40, b=40, l=40, r=40), 
        # MATCHING HEIGHT
        height=500,
        # If you want to force a specific width in pixels:
        # width=900 
    )

    # Ensure use_container_width is True so it stretches to our CSS max-width
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # --- CENTERED EXIT BUTTON ---
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    with col_c2:
        if st.button("EXIT BIOMETRICS", use_container_width=True):
            st.session_state.show_biometrics = False
            st.session_state.sidebar_state = "expanded"
            st.rerun()