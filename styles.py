CSS_CODE = """
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
    """
HEADER_HTML = """
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
    """

def get_result_orb(orb_color, result_text, conf_val):
    """Generates the HTML for the diagnostic result orb."""
    return f"""
    <div class="orb-container">
        <div class="orb" style="background: {orb_color}; box-shadow: 0 0 30px {orb_color};"></div>
        <div>
            <p class="result-title">{result_text}</p>
            <p class="confidence-white">{conf_val:.2f}%</p>
            <p style="text-align:center; color:rgba(255,255,255,0.85); font-size:0.75rem; letter-spacing: 2px; margin:0; font-weight:300;">AI CONFIDENCE SCORE</p>
        </div>
    </div>
    """