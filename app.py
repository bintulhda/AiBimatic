
"""
Smart Health Tracker - Hackathon Project
A comprehensive health prediction and analysis tool using ML and rule-based logic.
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
    page_title="🏥 Smart Health Tracker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DARK MODE INITIALIZATION
# ============================================================================

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Dynamic CSS for light and dark modes
def get_css_theme():
    if st.session_state.dark_mode:
        return """
        <style>
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --accent: #00d4ff;
            --success: #4caf50;
            --warning: #ff9800;
            --danger: #f44336;
            --info: #2196f3;
        }
        
        .main-title {
            text-align: center;
            color: #00d4ff;
            margin-bottom: 2rem;
            font-size: 2.5em;
            font-weight: bold;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #2d2d2d 0%, #1f1f1f 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #404040;
            color: #ffffff;
        }
        
        .warning-box {
            background-color: rgba(255, 152, 0, 0.1);
            padding: 15px;
            border-left: 4px solid #ff9800;
            border-radius: 5px;
            margin: 10px 0;
            color: #ffb74d;
        }
        
        .success-box {
            background-color: rgba(76, 175, 80, 0.1);
            padding: 15px;
            border-left: 4px solid #4caf50;
            border-radius: 5px;
            margin: 10px 0;
            color: #81c784;
        }
        
        .danger-box {
            background-color: rgba(244, 67, 54, 0.1);
            padding: 15px;
            border-left: 4px solid #f44336;
            border-radius: 5px;
            margin: 10px 0;
            color: #ef5350;
        }
        
        .info-box {
            background-color: rgba(33, 150, 243, 0.1);
            padding: 15px;
            border-left: 4px solid #2196f3;
            border-radius: 5px;
            margin: 10px 0;
            color: #64b5f6;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #2d2d2d 0%, #253238 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 1px solid #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
            color: #ffffff;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-item {
            background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        </style>
        """
    else:
        return """
        <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f5;
            --text-primary: #000000;
            --text-secondary: #666666;
            --accent: #1f77b4;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
            --info: #17a2b8;
        }
        
        .main-title {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
            font-size: 2.5em;
            font-weight: bold;
        }
        
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #e0e0e0;
        }
        
        .warning-box {
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            border-radius: 5px;
            margin: 10px 0;
            color: #856404;
        }
        
        .success-box {
            background-color: #d4edda;
            padding: 15px;
            border-left: 4px solid #28a745;
            border-radius: 5px;
            margin: 10px 0;
            color: #155724;
        }
        
        .danger-box {
            background-color: #f8d7da;
            padding: 15px;
            border-left: 4px solid #dc3545;
            border-radius: 5px;
            margin: 10px 0;
            color: #721c24;
        }
        
        .info-box {
            background-color: #d1ecf1;
            padding: 15px;
            border-left: 4px solid #17a2b8;
            border-radius: 5px;
            margin: 10px 0;
            color: #0c5460;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 2px solid #1f77b4;
            box-shadow: 0 4px 15px rgba(31, 119, 180, 0.1);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-item {
            background: linear-gradient(135deg, #1f77b4 0%, #1464a8 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        </style>
        """

st.markdown(get_css_theme(), unsafe_allow_html=True)

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
# SIDEBAR NAVIGATION & DARK MODE TOGGLE
# ============================================================================

st.sidebar.markdown("## 🏥 Navigation")

# Dark mode toggle
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    st.sidebar.markdown("**Dark Mode**")
with col2:
    if st.sidebar.button("🌙" if not st.session_state.dark_mode else "☀️", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a feature:",
    ["🏠 Home", "🩺 Diabetes Predictor", "❤️ Blood Pressure Analysis", 
     "⚖️ BMI Calculator", "📊 Dashboard", "🏃 Activity Tracker", 
     "💧 Water Intake", "😴 Sleep Tracker", "🔥 Calorie Calculator"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 Quick Stats
Track your daily health metrics:
- **💧 Water Intake**: Stay hydrated!
- **🏃 Exercise**: 150 min/week goal
- **😴 Sleep**: 7-9 hours recommended
- **🔥 Calories**: Personalized needs

### About This App
This Smart Health Tracker helps you monitor your health using:
- **Machine Learning** for diabetes prediction
- **Rule-based Logic** for BP analysis
- **Data Visualization** for insights
- **Activity Tracking** for wellness

⚠️ **Disclaimer:** This is NOT medical advice. Always consult a healthcare professional.
""")

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>🏥 Smart Health Tracker</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("👨‍⚕️ **Diabetes Predictor**\nML-based prediction using patient data")
    
    with col2:
        st.info("❤️ **BP Analysis**\nBlood pressure categorization & advice")
    
    with col3:
        st.info("⚖️ **BMI Calculator**\nWeight classification & recommendations")
    
    st.markdown("---")
    
    st.markdown("""
    ## Welcome to Smart Health Tracker! 👋
    
    This application is designed to help you monitor and understand your health metrics better.
    
    ### What You Can Do:
    
    1. **Predict Diabetes Risk** 🩺
       - Input your health metrics
       - Get an AI-powered diabetes prediction with confidence scores
       - Compare your glucose levels with healthy averages
    
    2. **Analyze Blood Pressure** ❤️
       - Enter your systolic and diastolic readings
       - Get instant categorization (Normal, Elevated, Stage 1, Stage 2)
       - Receive personalized health advice
    
    3. **Calculate BMI** ⚖️
       - Find your Body Mass Index
       - Get weight category classification
       - Receive lifestyle recommendations
    
    4. **Download Reports** 📋
       - Generate personalized health reports
       - Download as text files for your records
    
    ### Quick Tips:
    - ✅ Keep your readings handy when using the predictors
    - ✅ Take multiple readings for accuracy
    - ✅ Consult a doctor for serious concerns
    - ✅ Use this tool for general awareness, not diagnosis
    
    ---
    """)
    
    st.warning("""
    ⚠️ **Important Disclaimer**
    
    This application is for educational and informational purposes only. 
    The predictions and analyses provided are NOT medical diagnoses. 
    Please consult a qualified healthcare professional for accurate medical advice.
    """)

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
