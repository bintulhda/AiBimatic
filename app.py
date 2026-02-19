
"""
HealthGuard AI - Smart Health Tracker
A comprehensive health prediction and analysis tool with Dark Neon Glassmorphism Design.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DARK NEON GLASSMORPHISM THEME
# ============================================================================

def apply_glassmorphism_theme():
    """Apply Dark Neon Glassmorphism CSS Theme"""
    css = """
    <style>
    /* ========== ROOT & PAGE STYLING ========== */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1c3a 50%, #0f3a4a 100%) !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        color: #ffffff !important;
    }
    
    .stApp {
        background: transparent !important;
    }
    
    /* ========== SIDEBAR STYLING ========== */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(0, 242, 255, 0.1) !important;
    }
    
    [data-testid="stSidebarNav"] {
        background: transparent !important;
    }
    
    /* ========== SIDEBAR TEXT ========== */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5) !important;
    }
    
    /* ========== MAIN CONTENT AREA ========== */
    [data-testid="stMainBlockContainer"] {
        padding: 40px 20px !important;
    }
    
    /* ========== HEADERS & TITLES ========== */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(0, 242, 255, 0.6) !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }
    
    .main-title {
        text-align: center;
        background: rgba(0, 242, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        padding: 40px 20px !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        margin-bottom: 30px !important;
        font-size: 3em !important;
    }
    
    /* ========== GLASS MORPHISM CARDS ========== */
    .glass-card {
        background: rgba(10, 25, 47, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0, 242, 255, 0.15) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        padding: 25px !important;
        color: #ffffff !important;
        transition: all 0.3s ease !important;
    }
    
    .glass-card:hover {
        background: rgba(10, 25, 47, 0.7) !important;
        border-color: rgba(0, 242, 255, 0.3) !important;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-5px) !important;
    }
    
    /* ========== STATUS BOXES ========== */
    .success-box {
        background: rgba(76, 175, 80, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-left: 4px solid #4caf50 !important;
        border-radius: 15px !important;
        padding: 20px !important;
        color: #81c784 !important;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.1) !important;
    }
    
    .warning-box {
        background: rgba(255, 152, 0, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 152, 0, 0.3) !important;
        border-left: 4px solid #ff9800 !important;
        border-radius: 15px !important;
        padding: 20px !important;
        color: #ffb74d !important;
        box-shadow: 0 8px 32px rgba(255, 152, 0, 0.1) !important;
    }
    
    .danger-box {
        background: rgba(244, 67, 54, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        border-left: 4px solid #f44336 !important;
        border-radius: 15px !important;
        padding: 20px !important;
        color: #ef5350 !important;
        box-shadow: 0 8px 32px rgba(244, 67, 54, 0.1) !important;
    }
    
    .info-box {
        background: rgba(0, 242, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 242, 255, 0.3) !important;
        border-left: 4px solid #00f2ff !important;
        border-radius: 15px !important;
        padding: 20px !important;
        color: #64b5f6 !important;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.1) !important;
    }
    
    .feature-card {
        background: rgba(10, 25, 47, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 242, 255, 0.15) !important;
        padding: 20px !important;
        color: #ffffff !important;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.05) !important;
    }
    
    /* ========== INPUT ELEMENTS ========== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider > div > div > div,
    input[type="text"],
    input[type="number"],
    select {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 10px 15px !important;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    input[type="text"]:focus,
    input[type="number"]:focus {
        border-color: #00f2ff !important;
        outline: none !important;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.1), 0 0 10px rgba(0, 242, 255, 0.3) !important;
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, #00f2ff 0%, #0088cc 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 242, 255, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 255, 0.5), 0 0 20px rgba(0, 242, 255, 0.3) !important;
        background: linear-gradient(135deg, #00f2ff 0%, #00ccff 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* ========== NEON ACCENT BUTTON ========== */
    .neon-button {
        background: linear-gradient(135deg, #ff0055 0%, #ff6699 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 0, 85, 0.5) !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(255, 0, 85, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .neon-button:hover {
        box-shadow: 0 6px 20px rgba(255, 0, 85, 0.6), 0 0 20px rgba(255, 0, 85, 0.4) !important;
        transform: translateY(-3px) !important;
    }
    
    /* ========== METRICS & STATS ========== */
    .stMetric {
        background: rgba(0, 242, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 242, 255, 0.15) !important;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 242, 255, 0.1) !important;
    }
    
    .stMetric > div > div > h3,
    .stMetric > div > label {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5) !important;
    }
    
    .stMetric > div > div > p {
        color: #00f2ff !important;
        font-weight: 600 !important;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(10, 25, 47, 0.3) !important;
        border-bottom: 1px solid rgba(0, 242, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #b0b0b0 !important;
        background: transparent !important;
        border-radius: 10px 10px 0 0 !important;
        border: none !important;
        padding: 15px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #00f2ff !important;
        background: rgba(0, 242, 255, 0.1) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #00f2ff !important;
        background: rgba(0, 242, 255, 0.15) !important;
        border-bottom: 3px solid #00f2ff !important;
        box-shadow: 0 -4px 15px rgba(0, 242, 255, 0.2) !important;
    }
    
    .stTabs [data-baseweb="tab-content"] {
        background: rgba(10, 25, 47, 0.3) !important;
        border-radius: 0 10px 10px 10px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* ========== RADIO & CHECKBOX ========== */
    .stRadio > div {
        background: rgba(10, 25, 47, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 242, 255, 0.15) !important;
        padding: 10px !important;
    }
    
    .stRadio > div > label {
        color: #ffffff !important;
        padding: 10px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stRadio > div > label:hover {
        background: rgba(0, 242, 255, 0.1) !important;
    }
    
    .stCheckbox > label {
        color: #ffffff !important;
    }
    
    /* ========== GRID LAYOUT ========== */
    .stats-grid {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)) !important;
        gap: 20px !important;
        margin: 20px 0 !important;
    }
    
    .stat-item {
        background: rgba(0, 242, 255, 0.08) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        text-align: center !important;
        color: #ffffff !important;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stat-item:hover {
        border-color: #00f2ff !important;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.2), 0 0 20px rgba(0, 242, 255, 0.15) !important;
        transform: translateY(-5px) !important;
    }
    
    /* ========== HERO SECTION ========== */
    .hero-section {
        background: rgba(10, 25, 47, 0.5) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 25px !important;
        border: 2px solid rgba(0, 242, 255, 0.2) !important;
        padding: 50px 40px !important;
        margin: 30px 0 !important;
        text-align: center !important;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    
    .hero-section h1 {
        font-size: 2.5em !important;
        margin: 10px 0 !important;
        background: linear-gradient(135deg, #00f2ff 0%, #ff0055 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    .hero-section p {
        font-size: 1.1em !important;
        color: #b0b0b0 !important;
        margin: 15px 0 !important;
    }
    
    /* ========== FEATURE CARD BUTTONS ========== */
    .feature-button {
        background: rgba(10, 25, 47, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        color: #ffffff !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 242, 255, 0.1) !important;
    }
    
    .feature-button:hover {
        background: rgba(0, 242, 255, 0.1) !important;
        border-color: #00f2ff !important;
        box-shadow: 0 8px 32px rgba(0, 242, 255, 0.2), 0 0 20px rgba(0, 242, 255, 0.15) !important;
        transform: translateY(-5px) !important;
    }
    
    /* ========== TEXT STYLING ========== */
    p, span, li {
        color: #ffffff !important;
    }
    
    a {
        color: #00f2ff !important;
        text-decoration: none !important;
        transition: all 0.3s ease !important;
    }
    
    a:hover {
        color: #ff0055 !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5) !important;
    }
    
    /* ========== DATAFRAME & TABLE ========== */
    .stDataFrame {
        background: rgba(10, 25, 47, 0.3) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    /* ========== MATPLOTLIB CHARTS ========== */
    .stPlotlyChart {
        background: rgba(10, 25, 47, 0.2) !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 10px !important;
        height: 10px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(10, 25, 47, 0.3) !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 242, 255, 0.5) !important;
        border-radius: 5px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 242, 255, 0.8) !important;
    }
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply the glassmorphism theme
apply_glassmorphism_theme()

# ============================================================================
# MODEL TRAINING & INITIALIZATION
# ============================================================================

@st.cache_resource
def load_and_train_model():
    """Load diabetes dataset and train Random Forest Classifier"""
    try:
        # Load diabetes dataset
        df = pd.read_csv("diabetes.csv")
        
        # Define features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Create scaler for input normalization
        scaler = StandardScaler()
        scaler.fit(X)
        
        return model, scaler, X.columns.tolist()
    except FileNotFoundError:
        st.error("❌ diabetes.csv not found! Please ensure it's in the same directory as app.py")
        return None, None, None

# ============================================================================
# FEATURE FUNCTIONS
# ============================================================================

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, dpf, age, model, scaler):
    """Predict diabetes using the trained ML model"""
    try:
        # Create input array
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                insulin, bmi, dpf, age]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None


def analyze_blood_pressure(systolic, diastolic):
    """Analyze blood pressure and provide category and advice"""
    if systolic < 120 and diastolic < 80:
        category = "🟢 Normal"
        color = "green"
        advice = "✅ Your blood pressure is normal. Maintain a healthy lifestyle!"
        detail = "Keep up with regular exercise and a balanced diet."
    elif systolic >= 120 and systolic < 130 and diastolic < 80:
        category = "🟡 Elevated"
        color = "orange"
        advice = "⚠️ Your blood pressure is elevated. Monitor regularly."
        detail = "Consider reducing sodium intake and increasing physical activity."
    elif (systolic >= 130 and systolic < 140) or (diastolic >= 80 and diastolic < 90):
        category = "🟠 High BP (Stage 1)"
        color = "orange"
        advice = "⚠️ Stage 1 Hypertension detected. Consult your doctor."
        detail = "Lifestyle changes and possible medication may be needed. See a cardiologist."
    elif systolic >= 140 or diastolic >= 90:
        category = "🔴 High BP (Stage 2)"
        color = "red"
        advice = "❌ Stage 2 Hypertension detected. Seek medical attention immediately."
        detail = "URGENT: Consult a cardiologist or medical professional immediately."
    else:
        category = "⚪ Unknown"
        color = "gray"
        advice = "Unable to categorize"
        detail = "Please enter valid blood pressure values"
    
    return category, color, advice, detail


def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI and categorize"""
    try:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        if bmi < 18.5:
            category = "⚠️ Underweight"
            color = "blue"
            recommendation = "Consider consulting a nutritionist for weight gain."
        elif 18.5 <= bmi < 25:
            category = "✅ Normal Weight"
            color = "green"
            recommendation = "Great! Maintain your current weight and lifestyle."
        elif 25 <= bmi < 30:
            category = "⚠️ Overweight"
            color = "orange"
            recommendation = "Consider increasing physical activity and adjusting diet."
        else:
            category = "🔴 Obese"
            color = "red"
            recommendation = "Consult a healthcare professional for a personalized plan."
        
        return bmi, category, color, recommendation
    except Exception as e:
        st.error(f"Error calculating BMI: {str(e)}")
        return None, None, None, None


def create_health_comparison_chart(user_value, healthy_avg, metric_name):
    """Create a comparison chart between user value and healthy average"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Your Value', 'Healthy Average']
    values = [user_value, healthy_avg]
    colors = ['#ff6b6b', '#51cf66']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    return fig


def generate_report(user_data, predictions):
    """Generate a downloadable report"""
    report = f"""
{'='*60}
SMART HEALTH TRACKER - PERSONAL HEALTH REPORT
{'='*60}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
HEALTH METRICS ANALYSIS
{'='*60}

{user_data}

{predictions}

{'='*60}
DISCLAIMER
{'='*60}
⚠️ This application is for informational purposes only and does 
NOT constitute medical advice. Please consult a qualified healthcare 
professional for accurate diagnosis and treatment. Always seek 
professional medical advice before making any health-related decisions.

{'='*60}
"""
    return report


def track_water_intake(current_intake, goal=8):
    """Track daily water intake and provide recommendations"""
    percentage = (current_intake / goal) * 100
    status = "✅ Goal Reached!" if current_intake >= goal else f"🚰 {goal - current_intake} glasses left"
    return percentage, status


def track_activity(minutes_exercised, goal=150):
    """Track weekly exercise and provide recommendations"""
    percentage = (minutes_exercised / goal) * 100
    status = "✅ Weekly Goal Reached!" if minutes_exercised >= goal else f"⏱️ {goal - minutes_exercised} min left"
    return percentage, status


def track_sleep(hours_slept, goal=8):
    """Track sleep duration and provide recommendations"""
    percentage = (hours_slept / goal) * 100
    quality = ""
    if hours_slept < 5:
        quality = "Poor - Get more rest!"
    elif hours_slept < 7:
        quality = "Fair - Try to get more sleep"
    elif hours_slept <= 9:
        quality = "Good - Keep it up!"
    else:
        quality = "Excellent - Perfect sleep!"
    return percentage, quality


def calculate_calorie_needs(weight, height, age, gender, activity_level):
    """Calculate daily calorie needs using Mifflin-St Jeor equation"""
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extremely Active": 1.9
    }
    
    daily_calories = bmr * activity_multipliers.get(activity_level, 1.55)
    return bmr, daily_calories


# ============================================================================
# SIDEBAR NAVIGATION & BRANDING
# ============================================================================

# HealthGuard AI Sidebar Header
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px 0; margin-bottom: 20px;'>
    <h1 style='font-size: 2em; margin: 0; background: linear-gradient(135deg, #00f2ff 0%, #ff0055 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               background-clip: text; text-shadow: 0 0 20px rgba(0, 242, 255, 0.5);'>
        🧠 HealthGuard AI
    </h1>
    <p style='margin: 5px 0; color: #00f2ff; font-size: 0.85em; text-shadow: 0 0 10px rgba(0, 242, 255, 0.3);'>
        Your AI Health Companion
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation
st.sidebar.markdown('<p style="color: #00f2ff; font-weight: 600; text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);">📍 NAVIGATION</p>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select a feature:",
    ["🏠 Home", "🩺 Diabetes Predictor", "❤️ Blood Pressure Analysis", 
     "⚖️ BMI Calculator", "📊 Dashboard", "🏃 Activity Tracker", 
     "💧 Water Intake", "😴 Sleep Tracker", "🔥 Calorie Calculator"],
    label_visibility="collapsed"
)

# Handle page override from card clicks
if "page_override" in st.session_state:
    page = st.session_state.page_override
    del st.session_state.page_override

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(0, 242, 255, 0.05); backdrop-filter: blur(10px); 
            border: 1px solid rgba(0, 242, 255, 0.15); border-radius: 10px; 
            padding: 15px; margin: 15px 0;'>
    <h4 style='color: #00f2ff; text-shadow: 0 0 10px rgba(0, 242, 255, 0.5); margin-top: 0;'>🎯 Today's Goals</h4>
    <ul style='margin: 10px 0; padding-left: 20px; font-size: 0.9em;'>
        <li>💧 Hydration: 8 glasses</li>
        <li>🏃 Exercise: 150 minutes</li>
        <li>😴 Sleep: 8 hours</li>
        <li>❤️ Monitor: BP & Glucose</li>
    </ul>
</div>

<div style='background: rgba(76, 175, 80, 0.05); backdrop-filter: blur(10px); 
            border: 1px solid rgba(76, 175, 80, 0.15); border-radius: 10px; 
            padding: 15px; margin: 15px 0;'>
    <h4 style='color: #4caf50; margin-top: 0;'>✨ Features</h4>
    <ul style='margin: 10px 0; padding-left: 20px; font-size: 0.85em;'>
        <li>ML Diabetes Prediction</li>
        <li>Real-time BP Analysis</li>
        <li>Activity Tracking</li>
        <li>Health Reports</li>
    </ul>
</div>

<p style='color: #b0b0b0; font-size: 0.8em; text-align: center; margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(0, 242, 255, 0.1);'>
⚠️ <b>Disclaimer:</b> This is NOT medical advice.<br>Always consult a healthcare professional.
</p>
""", unsafe_allow_html=True)

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class='hero-section'>
        <h1>Welcome back, User 👋</h1>
        <p>Your AI health companion is ready to help you achieve optimal wellness</p>
        <hr style='border: 1px solid rgba(0, 242, 255, 0.2); margin: 20px 0;'>
        <p style='color: #00f2ff; font-size: 0.95em;'>
            🧠 Powered by Advanced ML | 📊 Real-time Analysis | 🔒 Your Privacy Protected
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Cards Grid
    st.markdown('<h2 style="text-align: center; color: #ffffff; margin: 30px 0; text-shadow: 0 0 15px rgba(0, 242, 255, 0.6);">✨ Core Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <div style='font-size: 3em; margin-bottom: 15px;'>🩺</div>
            <h3 style='color: #00f2ff; margin-top: 0; text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);'>
                Diabetes Predictor
            </h3>
            <p style='color: #b0b0b0; font-size: 0.95em;'>
                AI-powered risk assessment using machine learning
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🩺 Analyze Now", key="diabetes_btn", use_container_width=True):
            st.session_state.page_override = "🩺 Diabetes Predictor"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <div style='font-size: 3em; margin-bottom: 15px;'>❤️</div>
            <h3 style='color: #ff0055; margin-top: 0; text-shadow: 0 0 10px rgba(255, 0, 85, 0.5);'>
                Blood Pressure
            </h3>
            <p style='color: #b0b0b0; font-size: 0.95em;'>
                Real-time BP monitoring & instant categorization
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("❤️ Check Now", key="bp_btn", use_container_width=True):
            st.session_state.page_override = "❤️ Blood Pressure Analysis"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <div style='font-size: 3em; margin-bottom: 15px;'>⚖️</div>
            <h3 style='color: #00f2ff; margin-top: 0; text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);'>
                BMI Calculator
            </h3>
            <p style='color: #b0b0b0; font-size: 0.95em;'>
                Calculate & track your body mass index
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚖️ Calculate", key="bmi_btn", use_container_width=True):
            st.session_state.page_override = "⚖️ BMI Calculator"
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Additional Features
    st.markdown('<h2 style="text-align: center; color: #ffffff; margin: 30px 0; text-shadow: 0 0 15px rgba(0, 242, 255, 0.6);">🚀 Additional Tools</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div class='feature-card' style='text-align: center;'>
            <div style='font-size: 2em; margin-bottom: 10px;'>🏃</div>
            <h4 style='color: #00f2ff; margin: 5px 0;'>Activity Tracker</h4>
            <p style='color: #b0b0b0; font-size: 0.9em; margin: 0;'>150 min/week goal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card' style='text-align: center;'>
            <div style='font-size: 2em; margin-bottom: 10px;'>💧</div>
            <h4 style='color: #00f2ff; margin: 5px 0;'>Water Intake</h4>
            <p style='color: #b0b0b0; font-size: 0.9em; margin: 0;'>Stay hydrated</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card' style='text-align: center;'>
            <div style='font-size: 2em; margin-bottom: 10px;'>😴</div>
            <h4 style='color: #00f2ff; margin: 5px 0;'>Sleep Tracker</h4>
            <p style='color: #b0b0b0; font-size: 0.9em; margin: 0;'>7-9 hours recommended</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Info Boxes
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <h4 style='margin-top: 0;'>🎯 How It Works</h4>
            <ol style='padding-left: 20px; margin: 10px 0;'>
                <li>Input your health metrics</li>
                <li>AI analyzes your data instantly</li>
                <li>Get personalized insights</li>
                <li>Download your health report</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='success-box'>
            <h4 style='margin-top: 0;'>✅ Key Features</h4>
            <ul style='padding-left: 20px; margin: 10px 0;'>
                <li>Machine Learning Predictions</li>
                <li>Real-time Health Analysis</li>
                <li>Downloadable Reports</li>
                <li>Data Visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Warning
    st.markdown("""
    <div class='danger-box'>
        <h4 style='margin-top: 0;'>⚠️ Medical Disclaimer</h4>
        <p style='margin: 10px 0;'>
            This application is for informational purposes only and does <b>NOT</b> constitute medical advice.
            The predictions and analyses provided are not professional medical diagnoses.
            <b>Always consult a qualified healthcare professional</b> for accurate medical guidance.
            In case of medical emergency, seek immediate medical attention.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# DIABETES PREDICTOR PAGE
# ============================================================================

elif page == "🩺 Diabetes Predictor":
    st.markdown("<h1 class='main-title'>🩺 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_and_train_model()
    
    if model is None:
        st.error("Cannot load model. Please check diabetes.csv file.")
    else:
        st.info("💡 Enter your health metrics below to predict your diabetes risk")
        
        # Create input columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
        
        with col2:
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=80)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
        with col3:
            insulin = st.number_input("Insulin (µU/mL)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0)
        
        with col4:
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("Age (years)", min_value=1, max_value=150, value=30)
        
        # Prediction button
        if st.button("🔍 Predict Diabetes Risk", key="predict_button", use_container_width=True):
            prediction, probability = predict_diabetes(
                pregnancies, glucose, blood_pressure, skin_thickness, 
                insulin, bmi, dpf, age, model, scaler
            )
            
            if prediction is not None:
                st.markdown("---")
                
                # Display prediction result
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown("""<div class='danger-box'>
                        <h3>⚠️ Diabetes Risk Detected</h3>
                        Based on your metrics, you may have a higher risk of diabetes.
                        </div>""", unsafe_allow_html=True)
                        confidence = probability[1] * 100
                        risk_level = "High Risk"
                    else:
                        st.markdown("""<div class='success-box'>
                        <h3>✅ Low Diabetes Risk</h3>
                        Based on your metrics, your diabetes risk appears low.
                        </div>""", unsafe_allow_html=True)
                        confidence = probability[0] * 100
                        risk_level = "Low Risk"
                    
                    st.metric("Confidence Score", f"{confidence:.1f}%")
                
                with col2:
                    # Create prediction gauge
                    fig, ax = plt.subplots(figsize=(6, 4))
                    risk_percentage = probability[1] * 100
                    colors_gauge = ['#51cf66', '#ffa500', '#ff6b6b']
                    
                    if risk_percentage < 30:
                        color_idx = 0
                    elif risk_percentage < 70:
                        color_idx = 1
                    else:
                        color_idx = 2
                    
                    ax.barh(['Risk Level'], [risk_percentage], color=colors_gauge[color_idx], height=0.5)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Risk Percentage (%)', fontweight='bold')
                    ax.text(risk_percentage + 2, 0, f'{risk_percentage:.1f}%', 
                           va='center', fontweight='bold', fontsize=12)
                    
                    st.pyplot(fig, use_container_width=True)
                
                # Health comparison chart
                st.markdown("#### 📊 Your Glucose Level vs. Healthy Average")
                healthy_glucose_avg = 100
                fig = create_health_comparison_chart(glucose, healthy_glucose_avg, "Glucose (mg/dL)")
                st.pyplot(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.markdown("#### 📋 Recommendations")
                
                if prediction == 1:
                    st.markdown(f"""
                    🔴 **High Risk Warning**
                    - **Consult a Doctor:** Schedule an appointment with an endocrinologist
                    - **Lifestyle Changes:** Increase physical activity to at least 150 min/week
                    - **Diet:** Reduce sugar and refined carbohydrates
                    - **Monitoring:** Check blood glucose regularly
                    - **Testing:** Get an HbA1c test done (diabetes screening)
                    """)
                else:
                    st.markdown(f"""
                    🟢 **Low Risk - Stay Healthy**
                    - **Maintain Habits:** Continue your current healthy lifestyle
                    - **Regular Check-ups:** Get health screenings every 1-2 years
                    - **Exercise:** Aim for 150 minutes of moderate activity weekly
                    - **Balanced Diet:** Include plenty of fruits, vegetables, and whole grains
                    - **Monitor:** Keep track of your glucose levels
                    """)
                
                # Prepare data for report download
                user_data = f"""
USER HEALTH METRICS:
- Pregnancies: {pregnancies}
- Glucose: {glucose} mg/dL
- Blood Pressure: {blood_pressure} mmHg
- Skin Thickness: {skin_thickness} mm
- Insulin: {insulin} µU/mL
- BMI: {bmi} kg/m²
- Diabetes Pedigree Function: {dpf}
- Age: {age} years
"""
                
                predictions_text = f"""
DIABETES PREDICTION RESULTS:
- Risk Level: {risk_level}
- Confidence Score: {confidence:.2f}%
- Prediction: {'DIABETIC RISK DETECTED' if prediction == 1 else 'LOW DIABETES RISK'}
"""
                
                report = generate_report(user_data, predictions_text)
                
                # Download button
                st.download_button(
                    label="📥 Download Health Report",
                    data=report,
                    file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ============================================================================
# BLOOD PRESSURE ANALYSIS PAGE
# ============================================================================

elif page == "❤️ Blood Pressure Analysis":
    st.markdown("<h1 class='main-title'>❤️ Blood Pressure Analysis</h1>", unsafe_allow_html=True)
    
    st.info("💡 Enter your blood pressure readings (in mmHg) below")
    
    col1, col2 = st.columns(2)
    
    with col1:
        systolic = st.number_input("Systolic (Upper) Pressure (mmHg)", min_value=0, max_value=300, value=120)
    
    with col2:
        diastolic = st.number_input("Diastolic (Lower) Pressure (mmHg)", min_value=0, max_value=200, value=80)
    
    if st.button("📊 Analyze Blood Pressure", key="bp_button", use_container_width=True):
        category, color, advice, detail = analyze_blood_pressure(systolic, diastolic)
        
        st.markdown("---")
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"### {category}")
            
            if "Normal" in category:
                st.markdown("""<div class='success-box'>
                Your blood pressure is in the normal range.
                </div>""", unsafe_allow_html=True)
            elif "Elevated" in category or "Stage 1" in category:
                st.markdown("""<div class='warning-box'>
                Your blood pressure is elevated. Monitor regularly.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='danger-box'>
                Your blood pressure is dangerously high. Seek medical attention.
                </div>""", unsafe_allow_html=True)
        
        with col2:
            # Display readings as metrics
            st.metric("Systolic", f"{systolic} mmHg")
            st.metric("Diastolic", f"{diastolic} mmHg")
        
        # Display advice
        st.markdown("---")
        st.markdown("#### 📋 Health Advice")
        st.write(advice)
        st.write(detail)
        
        # BP Categories reference
        st.markdown("---")
        st.markdown("#### 📚 Blood Pressure Categories")
        
        bp_categories = {
            "Category": ["Normal", "Elevated", "High (Stage 1)", "High (Stage 2)"],
            "Systolic": ["< 120", "120-129", "130-139", "≥ 140"],
            "Diastolic": ["< 80", "< 80", "80-89", "≥ 90"]
        }
        
        bp_df = pd.DataFrame(bp_categories)
        st.table(bp_df)
        
        # Download report
        user_data = f"""
BLOOD PRESSURE READINGS:
- Systolic: {systolic} mmHg
- Diastolic: {diastolic} mmHg
"""
        
        predictions_text = f"""
BLOOD PRESSURE ANALYSIS:
- Category: {category}
- Status: {advice}
- Recommendation: {detail}
"""
        
        report = generate_report(user_data, predictions_text)
        
        st.download_button(
            label="📥 Download BP Report",
            data=report,
            file_name=f"bp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ============================================================================
# BMI CALCULATOR PAGE
# ============================================================================

elif page == "⚖️ BMI Calculator":
    st.markdown("<h1 class='main-title'>⚖️ BMI Calculator</h1>", unsafe_allow_html=True)
    
    st.info("💡 Calculate your Body Mass Index and get weight category insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
    
    with col2:
        height_cm = st.number_input("Height (cm)", min_value=1.0, max_value=250.0, value=175.0)
    
    if st.button("🔢 Calculate BMI", key="bmi_button", use_container_width=True):
        bmi, category, color, recommendation = calculate_bmi(weight_kg, height_cm)
        
        if bmi is not None:
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display BMI metric
                st.metric("Your BMI", f"{bmi:.1f}")
                st.markdown(f"### {category}")
            
            with col2:
                # BMI Scale visualization
                fig, ax = plt.subplots(figsize=(10, 3))
                
                bmi_ranges = [0, 18.5, 25, 30, 50]
                colors_bmi = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
                labels_bmi = ['Underweight\n< 18.5', 'Normal\n18.5-24.9', 'Overweight\n25-29.9', 'Obese\n≥ 30']
                
                for i, (start, end) in enumerate(zip(bmi_ranges[:-1], bmi_ranges[1:])):
                    ax.barh(0, end - start, left=start, height=0.5, color=colors_bmi[i], 
                           edgecolor='black', linewidth=2)
                
                # Mark user's BMI
                ax.axvline(bmi, color='red', linestyle='--', linewidth=3, label=f'Your BMI: {bmi:.1f}')
                
                ax.set_xlim(0, 50)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel('BMI Value', fontweight='bold')
                ax.set_xticks([18.5, 25, 30])
                ax.set_yticks([])
                ax.legend(loc='upper right')
                
                st.pyplot(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.markdown("#### 📋 Recommendations")
            st.write(recommendation)
            
            # BMI Categories reference
            st.markdown("---")
            st.markdown("#### 📚 BMI Categories")
            
            bmi_info = {
                "Category": ["Underweight", "Normal Weight", "Overweight", "Obese"],
                "BMI Range": ["< 18.5", "18.5 - 24.9", "25.0 - 29.9", "≥ 30.0"]
            }
            
            bmi_df = pd.DataFrame(bmi_info)
            st.table(bmi_df)
            
            # Download report
            user_data = f"""
BODY METRICS:
- Weight: {weight_kg} kg
- Height: {height_cm} cm
"""
            
            predictions_text = f"""
BMI CALCULATION RESULTS:
- BMI Value: {bmi:.2f}
- Category: {category}
- Recommendation: {recommendation}
"""
            
            report = generate_report(user_data, predictions_text)
            
            st.download_button(
                label="📥 Download BMI Report",
                data=report,
                file_name=f"bmi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

elif page == "📊 Dashboard":
    st.markdown("<h1 class='main-title'>📊 Health Dashboard</h1>", unsafe_allow_html=True)
    
    st.info("📈 Comprehensive health metrics summary and insights")
    
    # Load model for reference stats
    model, scaler, feature_names = load_and_train_model()
    
    if model is not None:
        try:
            df = pd.read_csv("diabetes.csv")
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🩺 Health Insights", 
                                               "📈 Correlation Matrix", "📋 Health Tips"])
            
            with tab1:
                st.markdown("#### Dataset Statistics")
                st.write(f"**Total Records:** {len(df)}")
                st.write(f"**Features:** {len(df.columns)}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    diabetic_count = len(df[df['Outcome'] == 1])
                    st.metric("Diabetic Cases", diabetic_count)
                with col2:
                    healthy_count = len(df[df['Outcome'] == 0])
                    st.metric("Healthy Cases", healthy_count)
                with col3:
                    diabetes_rate = (diabetic_count / len(df)) * 100
                    st.metric("Diabetes Rate", f"{diabetes_rate:.1f}%")
                with col4:
                    avg_age = df['Age'].mean()
                    st.metric("Avg Age", f"{avg_age:.0f} years")
                
                # Feature statistics
                st.markdown("#### Key Metrics Summary")
                st.dataframe(df.describe().round(2), use_container_width=True)
            
            with tab2:
                st.markdown("#### Health Insights from Dataset")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Outcome distribution
                    fig, ax = plt.subplots(figsize=(8, 5))
                    outcome_counts = df['Outcome'].value_counts()
                    colors_pie = ['#51cf66', '#ff6b6b']
                    labels_pie = ['Healthy', 'Diabetic']
                    ax.pie(outcome_counts, labels=labels_pie, autopct='%1.1f%%', 
                          colors=colors_pie, startangle=90, textprops={'fontsize': 12})
                    ax.set_title('Health Status Distribution', fontweight='bold', fontsize=14)
                    st.pyplot(fig, use_container_width=True)
                
                with col2:
                    # Average metrics comparison
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    healthy_avg = df[df['Outcome'] == 0]['Glucose'].mean()
                    diabetic_avg = df[df['Outcome'] == 1]['Glucose'].mean()
                    
                    categories = ['Healthy\nAverage', 'Diabetic\nAverage']
                    values = [healthy_avg, diabetic_avg]
                    colors_bar = ['#51cf66', '#ff6b6b']
                    
                    ax.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
                    ax.set_ylabel('Glucose Level (mg/dL)', fontweight='bold')
                    ax.set_title('Average Glucose Levels', fontweight='bold', fontsize=14)
                    
                    for i, v in enumerate(values):
                        ax.text(i, v + 2, f'{v:.1f}', ha='center', fontweight='bold')
                    
                    st.pyplot(fig, use_container_width=True)
            
            with tab3:
                st.markdown("#### Feature Correlation Heatmap")
                
                # Select numerical columns
                numeric_df = df.select_dtypes(include=[np.number])
                
                if len(numeric_df.columns) > 0:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr_matrix = numeric_df.corr()
                    
                    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    
                    ax.set_xticks(range(len(corr_matrix.columns)))
                    ax.set_yticks(range(len(corr_matrix.columns)))
                    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                    ax.set_yticklabels(corr_matrix.columns)
                    
                    # Add correlation values
                    for i in range(len(corr_matrix.columns)):
                        for j in range(len(corr_matrix.columns)):
                            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontsize=9)
                    
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig, use_container_width=True)
            
            with tab4:
                st.markdown("#### 💡 Personalized Health Tips")
                
                st.markdown("""
                ### Diabetes Prevention Tips:
                
                1. **Physical Activity**
                   - Aim for at least 150 minutes of moderate exercise per week
                   - Include strength training 2-3 times per week
                   
                2. **Healthy Diet**
                   - Reduce refined sugars and carbohydrates
                   - Increase fiber intake (fruits, vegetables, whole grains)
                   - Control portion sizes
                
                3. **Weight Management**
                   - Maintain a healthy BMI (18.5 - 24.9)
                   - Lose 5-10% of body weight if overweight
                
                4. **Blood Pressure Control**
                   - Keep BP below 130/80 mmHg
                   - Reduce sodium intake
                   - Manage stress effectively
                
                5. **Regular Monitoring**
                   - Get yearly health check-ups
                   - Monitor blood glucose if at risk
                   - Track weight and BP regularly
                
                6. **Lifestyle Changes**
                   - Quit smoking if applicable
                   - Limit alcohol consumption
                   - Get 7-9 hours of quality sleep
                   - Manage stress through meditation or yoga
                
                ### When to See a Doctor:
                - If you have persistent high glucose levels
                - If your BMI is in the obese range
                - If you have consistently high blood pressure
                - If you have a family history of diabetes
                - For annual health screenings
                """)
        
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
    else:
        st.error("Cannot load dashboard. Please check diabetes.csv file.")

# ============================================================================
# ACTIVITY TRACKER PAGE
# ============================================================================

elif page == "🏃 Activity Tracker":
    st.markdown("<h1 class='main-title'>🏃 Weekly Activity Tracker</h1>", unsafe_allow_html=True)
    
    st.info("📊 Track your exercise progress towards your weekly goal of 150 minutes")
    
    minutes_exercised = st.slider("Minutes of exercise this week", 0, 300, 75, step=15)
    
    percentage, status = track_activity(minutes_exercised, goal=150)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Weekly Progress", f"{minutes_exercised}/150 min", f"{percentage:.0f}%")
    
    with col2:
        st.markdown(f"<div class='feature-card'><h3>{status}</h3></div>", unsafe_allow_html=True)
    
    # Progress bar using HTML
    st.markdown(f"""
    <div style='width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 20px 0;'>
        <div style='width: {min(percentage, 100)}%; height: 30px; background: linear-gradient(90deg, #4caf50 0%, #81c784 100%); 
                    display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
            {percentage:.0f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Activity recommendations
    st.markdown("---")
    st.markdown("#### 📋 Exercise Recommendations")
    
    if minutes_exercised < 75:
        st.markdown("""<div class='danger-box'>
        <h4>⚠️ Low Activity</h4>
        Try to increase your activity level! Start with:
        - 30 min walks 5 days a week
        - Light strength training 2 days a week
        </div>""", unsafe_allow_html=True)
    elif minutes_exercised < 150:
        st.markdown("""<div class='warning-box'>
        <h4>📈 Getting There!</h4>
        You're on the right track! Continue with:
        - Consistent daily movement
        - Mix cardio and strength training
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='success-box'>
        <h4>✅ Goal Achieved!</h4>
        Great job maintaining your fitness! 
        - Keep up the excellent work
        - Consider increasing intensity for more benefits
        </div>""", unsafe_allow_html=True)

# ============================================================================
# WATER INTAKE TRACKER PAGE
# ============================================================================

elif page == "💧 Water Intake":
    st.markdown("<h1 class='main-title'>💧 Daily Water Intake Tracker</h1>", unsafe_allow_html=True)
    
    st.info("💡 Stay hydrated! Aim for 8 glasses of water per day")
    
    glasses_consumed = st.slider("Glasses of water consumed today", 0, 16, 4, step=1)
    
    percentage, status = track_water_intake(glasses_consumed, goal=8)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Daily Progress", f"{glasses_consumed}/8", f"{min(percentage, 100):.0f}%")
    
    with col2:
        remaining = max(0, 8 - glasses_consumed)
        st.metric("Water Needed", f"{remaining} glasses")
    
    with col3:
        st.markdown(f"<div class='feature-card' style='text-align: center;'><h3>{status}</h3></div>", unsafe_allow_html=True)
    
    # Visual water tracker
    st.markdown("---")
    st.markdown("#### 🥤 Hydration Level")
    
    hydration_html = '<div class="stats-grid">'
    for i in range(1, 9):
        if i <= glasses_consumed:
            hydration_html += '<div class="stat-item" style="background: linear-gradient(135deg, #1e88e5 0%, #42a5f5 100%);"><h3>💧</h3></div>'
        else:
            hydration_html += '<div class="stat-item" style="background: #e0e0e0; color: #999;"><h3>💧</h3></div>'
    hydration_html += '</div>'
    
    st.markdown(hydration_html, unsafe_allow_html=True)
    
    # Health benefits
    st.markdown("---")
    st.markdown("#### 🏥 Benefits of Staying Hydrated")
    st.markdown("""
    - ✅ Improves energy and focus
    - ✅ Aids digestion
    - ✅ Regulates body temperature
    - ✅ Supports healthy skin
    - ✅ Enhances athletic performance
    """)

# ============================================================================
# SLEEP TRACKER PAGE
# ============================================================================

elif page == "😴 Sleep Tracker":
    st.markdown("<h1 class='main-title'>😴 Sleep Tracker</h1>", unsafe_allow_html=True)
    
    st.info("💤 Track your sleep and ensure you get the recommended 7-9 hours per night")
    
    hours_slept = st.slider("Hours of sleep last night", 0.0, 12.0, 7.0, step=0.5)
    
    percentage, quality = track_sleep(hours_slept, goal=8)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sleep Duration", f"{hours_slept:.1f} hours")
    
    with col2:
        st.metric("Goal Progress", f"{percentage:.0f}%")
    
    with col3:
        st.markdown(f"<div class='feature-card' style='text-align: center;'><h3>{quality}</h3></div>", unsafe_allow_html=True)
    
    # Sleep quality gauge
    st.markdown("---")
    st.markdown("#### 📊 Sleep Quality")
    
    if hours_slept < 5:
        color, emoji = "#f44336", "😴"
    elif hours_slept < 7:
        color, emoji = "#ff9800", "😕"
    elif hours_slept <= 9:
        color, emoji = "#4caf50", "😊"
    else:
        color, emoji = "#2196f3", "😴"
    
    st.markdown(f"""
    <div style='width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 20px 0;'>
        <div style='width: {min((hours_slept/9)*100, 100)}%; height: 40px; background-color: {color}; 
                    display: flex; align-items: center; justify-content: center; font-size: 24px;'>
            {emoji}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sleep recommendations
    st.markdown("---")
    st.markdown("#### 💡 Sleep Tips")
    st.markdown("""
    **Recommended Sleep:** 7-9 hours per night
    
    **Better Sleep Habits:**
    - 🕐 Keep a consistent sleep schedule
    - 📵 Avoid screens 1 hour before bed
    - 🌙 Keep your bedroom cool and dark
    - ☕ Limit caffeine after 2 PM
    - 🧘 Try relaxation techniques
    - 🏃 Exercise regularly (but not before bed)
    """)

# ============================================================================
# CALORIE CALCULATOR PAGE
# ============================================================================

elif page == "🔥 Calorie Calculator":
    st.markdown("<h1 class='main-title'>🔥 Daily Calorie Needs Calculator</h1>", unsafe_allow_html=True)
    
    st.info("📊 Calculate your daily calorie needs based on your profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
        height_cm = st.number_input("Height (cm)", min_value=1.0, max_value=250.0, value=175.0)
    
    with col2:
        age = st.number_input("Age (years)", min_value=1, max_value=150, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    activity_level = st.selectbox(
        "Activity Level",
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"]
    )
    
    if st.button("🔢 Calculate Calories", use_container_width=True):
        bmr, daily_calories = calculate_calorie_needs(weight_kg, height_cm, age, gender, activity_level)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='feature-card'>
            <h3>🔥 Basal Metabolic Rate (BMR)</h3>
            <h2 style='color: #ff6b6b;'>{bmr:.0f} calories</h2>
            <p>Calories burned at rest per day</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='feature-card'>
            <h3>⚡ Daily Calorie Needs</h3>
            <h2 style='color: #4caf50;'>{daily_calories:.0f} calories</h2>
            <p>With {activity_level} activity level</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Goal breakdown
        st.markdown("---")
        st.markdown("#### 🎯 Calorie Goals")
        
        goals_data = {
            "Goal": ["Weight Loss", "Maintain Weight", "Weight Gain"],
            "Daily Calories": [f"{daily_calories * 0.85:.0f}", f"{daily_calories:.0f}", f"{daily_calories * 1.15:.0f}"],
            "Weekly Change": ["-0.5 kg", "Stable", "+0.5 kg"]
        }
        
        goals_df = pd.DataFrame(goals_data)
        st.table(goals_df)
        
        # Macro breakdown
        st.markdown("---")
        st.markdown("#### 🥗 Macro Nutrient Breakdown (Balanced Diet)")
        
        protein_cals = daily_calories * 0.30
        carbs_cals = daily_calories * 0.45
        fats_cals = daily_calories * 0.25
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='feature-card' style='background: linear-gradient(135deg, #ff6b6b 0%, #ef5350 100%);'>
            <h3>🍗 Protein</h3>
            <h2>{protein_cals:.0f}</h2>
            <p>{protein_cals/4:.0f}g per day</p>
            <p style='font-size: 12px;'>30% of calories</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='feature-card' style='background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);'>
            <h3>🍞 Carbs</h3>
            <h2>{carbs_cals:.0f}</h2>
            <p>{carbs_cals/4:.0f}g per day</p>
            <p style='font-size: 12px;'>45% of calories</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='feature-card' style='background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);'>
            <h3>🥑 Fats</h3>
            <h2>{fats_cals:.0f}</h2>
            <p>{fats_cals/9:.0f}g per day</p>
            <p style='font-size: 12px;'>25% of calories</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; border-radius: 10px;'>
<p style='font-size: 12px; color: #666;'>
⚠️ <b>DISCLAIMER:</b> This application is for informational purposes only and does NOT 
constitute medical advice. The predictions and analyses are not professional medical diagnoses. 
Always consult a qualified healthcare professional for accurate medical guidance. 
In case of medical emergency, seek immediate medical attention.
</p>
<p style='font-size: 11px; color: #999; margin-top: 15px;'>
<b>Features:</b> 🩺 Diabetes Predictor | ❤️ BP Analysis | ⚖️ BMI Calculator | 🏃 Activity Tracking | 💧 Water Intake | 😴 Sleep Tracker | 🔥 Calorie Calculator | 🌙 Dark Mode
</p>
<p style='font-size: 12px; color: #666;'>
<i>Smart Health Tracker v2.0 | Enhanced with Dark Mode & Activity Tracking | © 2026 | Built with ❤️ using Streamlit</i>
</p>
</div>
""", unsafe_allow_html=True)
