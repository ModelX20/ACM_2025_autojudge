import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# Set page config MUST be the first Streamlit command
st.set_page_config(
    page_title="AutoJudge",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling - letting Streamlit default handle light/dark but adding polish */
    
    /* Custom Header */
    .main-header {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #212529; /* Stark dark for better contrast */
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Subtle shadow for depth/visibility */
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    /* Card-like containers - Light friendly */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Input area styling tweak */
    .stTextArea textarea {
        background-color: #F8F9FA;
        border: 1px solid #CED4DA;
        border-radius: 5px;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        background-color: #007BFF;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 12px rgba(0,123,255,0.3);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #333;
    }
    
    /* Sidebar styling - DARK THEME */
    section[data-testid="stSidebar"] {
        background-color: #262730; /* Dark sidebar */
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }
    
    /* Input Container Styling */
    .input-card {
        background-color: #FFFFFF;
        padding: 1.5rem; /* Reduced padding */
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #F8F9FA;
        border: 1px solid #CED4DA;
        border-radius: 8px;
        transition: border-color 0.3s ease;
        min-height: 100px;
    }
    
    /* Reduce vertical gaps */
    .stMarkdown, .stButton {
        margin-bottom: 0.5rem !important;
    }

    /* Hide Streamlit Toolbar, footer, and menu */
    [data-testid="stToolbar"], 
    [data-testid="stHeader"], 
    header, 
    footer, 
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
        height: 0px !important;
    }

    /* ... (rest of CSS) ... */
    
    /* Result Card Styling */
    .result-card {
        background-color: #FFFFFF;
        padding: 20px; 
        border-radius: 12px; 
        border-left: 6px solid; 
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* ... */
    /* Restore missing Layout CSS */
    
    /* Remove default top padding - AGGRESSIVE */
    .block-container {
        padding-top: 0rem !important;
        margin-top: -1rem !important; /* Balanced: not too high, not too low */
        padding-bottom: 1rem;
        max-width: 95% !important;
    }
    
    /* Hide sidebar close button */
    [data-testid="stSidebar"] > div > div:first-child button {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Sidebar top padding removal */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem;
    }

</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models_dir = "models"
    clf_path = os.path.join(models_dir, 'classifier_model.pkl')
    reg_path = os.path.join(models_dir, 'regressor_model.pkl')
    
    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        return None, None
        
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    return clf, reg

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚖️ AutoJudge")
        
        st.markdown("---")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Acc", "55%")
        with col_s2:
            st.metric("RMSE", "1.72")
            
        st.markdown("---")
        
        # Super Compact "How it works"
        st.markdown("### ⚙️ Under the hood")
        st.info("**Ensemble Model:** LinearSVC + LogReg + RandomForest\n\n**Features:** 20,000 TF-IDF Tri-grams")

    # Main Content
    st.markdown('<div class="main-header">AutoJudge</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header" style="margin-bottom: 2rem;">AI-Powered Programming Problem Difficulty Assessor</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'desc' not in st.session_state: st.session_state.desc = ""
    if 'inp' not in st.session_state: st.session_state.inp = ""
    if 'out' not in st.session_state: st.session_state.out = ""

    def clear_inputs():
        st.session_state.desc = ""
        st.session_state.inp = ""
        st.session_state.out = ""

    st.markdown("### 📝 Problem Details")
    
    with st.container():
        description = st.text_area("Problem Statement", height=150, placeholder="Paste problem text here...", key="desc")
        
        c1, c2 = st.columns(2)
        with c1:
            input_desc = st.text_area("Input Format", height=80, placeholder="Input format...", key="inp")
        with c2:
            output_desc = st.text_area("Output Format", height=80, placeholder="Output format...", key="out")
        
        # Buttons Row
        st.markdown("<br>", unsafe_allow_html=True) 
        b1, b2, b3 = st.columns([1, 0.4, 3]) 
        with b1:
            predict_btn = st.button("🚀 Analyze & Predict", type="primary", use_container_width=True)
        with b2:
            st.button("🗑️ Clear", on_click=clear_inputs, help="Reset form", use_container_width=True)

    # Results Section (Bottom)
    if predict_btn:
        st.markdown("---")
        st.markdown("### 🎯 Difficulty Analysis")
        
        if not description.strip():
            st.warning("⚠️ Please provide at least a problem statement.")
        else:
            clf, reg = load_models()
            
            if clf is None or reg is None:
                st.error("❌ Models not found. Please train backend models first.")
            else:
                combined_text = f"{description} {input_desc} {output_desc}"
                input_data = [combined_text]
                
                with st.spinner("🧠 Analyzing complexity..."):
                    pred_class = clf.predict(input_data)[0]
                    pred_score = reg.predict(input_data)[0]
                    
                    # Layout Results
                    col_r1, col_r2 = st.columns([1, 1])
                    
                    difficulty_color = {"easy": "green", "medium": "orange", "hard": "red"}
                    color = difficulty_color.get(pred_class.lower(), "blue")
                    
                    with col_r1:
                        st.markdown(f"""
                        <div class="result-card" style="border-left-color: {color}; text-align: center;">
                            <h2 style="margin:0; color: {color}; text-transform: uppercase;">{pred_class}</h2>
                            <p style="margin:0; opacity: 0.7; color: #555;">Predicted Class</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_r2:
                        norm_score = min(max(pred_score, 0), 10) / 10.0
                        st.metric("Difficulty Score", f"{pred_score:.2f} / 10")
                        st.progress(norm_score)
                        st.caption("Estimated cognitive load.")

if __name__ == "__main__":
    main()
