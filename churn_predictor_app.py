#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from collections import OrderedDict

MODEL_FILE = 'xgboost_classifier_churn_model.joblib'
# NOTE: If you saved your StandardScaler object, uncomment and update this line:
# SCALER_FILE = 'standard_scaler.joblib' 

# CRITICAL FIX: The FEATURE_COLUMNS list MUST exactly match the order the model expects.
# This list is derived directly from the 'Model Expected Features' in your ValueError traceback.
FEATURE_COLUMNS = [
    'Tenure_Months', 
    'Monthly_Charges_ETB', 
    'Support_Calls_3Months', 
    'Network_Outage_Score_0_5', 
    'Total_Charges_ETB',
    # --- CORRECTED ORDER ---
    'Region_Regional City (Mid Density)', 
    'Region_Rural Area (Low Density)', 
    'Contract_Type_24-Month', 
    'Contract_Type_6-Month', 
    'Contract_Type_Month-to-month', 
    'Service_Plan_Data/Internet + Voice', 
    'Service_Plan_Premium Bundle (Data, telebirr, VAS)',
    'Network_Technology_3G', 
    'Network_Technology_4G/LTE', 
    'Network_Technology_5G'
]

@st.cache_resource # Caches the model so it loads only once
def load_assets():
    """Loads the trained model and scaler."""
    try:
        model = joblib.load(MODEL_FILE)
        # If you saved a scaler, load it here:
        # scaler = joblib.load(SCALER_FILE)
        return model #, scaler 
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model (and scaler if applicable)
model = load_assets() #, scaler

# --- Streamlit Application Setup ---

st.set_page_config(
    page_title="Ethio Telecom Churn Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #00A651; /* Ethio Telecom Green */
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .risk-high {
        background-color: #FFC0CB; /* Light Pink for High Risk */
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #E60000; /* Red Border */
        font-size: 18px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #E6FFE6; /* Light Green for Low Risk */
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00CC00; /* Green Border */
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🇪🇹 Ethio Telecom Churn Risk Analyzer")
st.markdown("---")

# --- Sidebar for Inputs (REORDERED BY IMPORTANCE) ---
with st.sidebar:
    st.header("Customer Profile Input")
    st.markdown("Adjust the variables to assess the churn risk.")
    st.markdown("---")

    # 1. CRITICAL SERVICE QUALITY (Ranks #1 & #4)
    st.subheader("1. Service & Support (Highest Impact)")

    # Network Outage Score (Rank #1)
    outage_score = st.slider(
        "Network Outage/Reliability Score (0-5)", 
        0, 5, 1, 
        help="Feature Importance Rank #1. 5 indicates severe, frequent outages."
    )

    # Support Calls (Rank #4)
    support_calls = st.slider("Support Calls (Last 3 Months)", 0, 10, 1, help="Feature Importance Rank #4. High call volume indicates frustration.")

    st.markdown("---")

    # 2. INFRASTRUCTURE & LOCATION (Ranks #2, #3, #5)
    st.subheader("2. Infrastructure & Location (High Impact)")

    # Region (Rank #2 & #3)
    region = st.selectbox(
        "Customer Region", 
        ['Addis Ababa (Base)', 'Regional City (Mid Density)', 'Rural Area (Low Density)'],
        index=0,
        help="Feature Importance Rank #2 & #3. Regional/Rural areas are significantly higher risk."
    )

    # Technology (Rank #5)
    network_tech = st.selectbox(
        "Network Technology Used",
        ['5G/4G (Base)', '3G', '2G Only'],
        index=0,
        help="Older technology (3G/2G) correlates with higher churn."
    )

    st.markdown("---")

    # 3. COMMERCIAL & CONTRACT (Medium Impact)
    st.subheader("3. Commercial & Contract Details")

    # Contract Type (Rank #7-9)
    contract = st.selectbox(
        "Contract Type",
        ['12-Month (Base)', '24-Month', '6-Month', 'Month-to-month'],
        index=0,
        help="6-Month and Month-to-month contracts carry more risk."
    )

    # Monthly Charges & Tenure
    monthly_charges = st.number_input("Monthly Charges (ETB)", 100.0, 3000.0, 550.0, step=50.0)
    tenure = st.slider("Tenure (Months)", 1, 84, 12)

    # Service Plan
    service_plan = st.selectbox(
        "Service Plan",
        ['Basic Mobile Voice (Base)', 'Data/Internet + Voice', 'Premium Bundle (Data, telebirr, VAS)'],
        index=0
    )


# --- Main Content Area for Prediction ---

st.header("1. Churn Prediction Result")

if st.button("Analyze Risk"):

    # 1. Prepare Feature Vector (Use OrderedDict to ensure keys are processed consistently)
    input_features = OrderedDict({col: 0 for col in FEATURE_COLUMNS})

    # Populate numerical/ordinal features
    input_features['Tenure_Months'] = tenure
    input_features['Monthly_Charges_ETB'] = monthly_charges
    input_features['Support_Calls_3Months'] = support_calls
    input_features['Network_Outage_Score_0_5'] = outage_score
    input_features['Total_Charges_ETB'] = monthly_charges * tenure

    # Populate categorical/OHE features
    # Region
    if region == 'Regional City (Mid Density)':
        input_features['Region_Regional City (Mid Density)'] = 1
    elif region == 'Rural Area (Low Density)':
        input_features['Region_Rural Area (Low Density)'] = 1

    # Contract
    if contract == 'Month-to-month':
        input_features['Contract_Type_Month-to-month'] = 1
    elif contract == '6-Month':
        input_features['Contract_Type_6-Month'] = 1
    elif contract == '24-Month':
        input_features['Contract_Type_24-Month'] = 1

    # Network Technology
    if network_tech == '3G':
        input_features['Network_Technology_3G'] = 1
    elif network_tech == '4G/LTE':
        input_features['Network_Technology_4G/LTE'] = 1
    elif network_tech == '5G/4G (Base)': # If 4G/5G is selected, set 5G to 1 as the primary modern tech.
         input_features['Network_Technology_5G'] = 1

    # Service Plan
    if service_plan == 'Data/Internet + Voice':
        input_features['Service_Plan_Data/Internet + Voice'] = 1
    elif service_plan == 'Premium Bundle (Data, telebirr, VAS)':
        input_features['Service_Plan_Premium Bundle (Data, telebirr, VAS)'] = 1

    # Convert to DataFrame in the exact order enforced by FEATURE_COLUMNS list
    input_df = pd.DataFrame([input_features])
    input_df = input_df[FEATURE_COLUMNS] 

    # 2. Scaling (Uncomment if you saved and loaded a scaler)
    # input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # 3. Predict
    prediction = model.predict(input_df)[0] 
    probability = model.predict_proba(input_df)[:, 1][0] 

    # 4. Display Results
    if prediction == 1:
        st.markdown(f"""
        <div class="risk-high">
            🚨 HIGH CHURN RISK
            <p style='font-size:16px; margin-top:5px;'>Predicted Churn Probability: <b>{probability:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.error("ACTION REQUIRED: This customer is flagged by the XGBoost model as high risk.")
    else:
        st.markdown(f"""
        <div class="risk-low">
            ✅ LOW CHURN RISK
            <p style='font-size:16px; margin-top:5px;'>Predicted Churn Probability: <b>{probability:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.success("Monitoring recommended, but immediate retention effort is not necessary.")

    st.markdown("---")

    # --- Actionable Insights based on Prediction ---
    st.header("2. Targeted Retention Strategy")
    st.markdown("**Intervention Plan based on High-Impact Features:**")

    if prediction == 1:
        if outage_score >= 3:
            st.warning("⚠️ **Service Quality Trigger (Rank #1):** High Network Outage Score is the primary driver. **Action:** Proactive credit/data bonus to acknowledge and compensate for service disruption.")

        if region != 'Addis Ababa (Base)':
            st.warning("🏘️ **Geographical Trigger (Rank #2/3):** Location contributes significantly to risk.")
            st.markdown("- **Action:** Offer **subsidized 4G device/data package** tied to a 12-month contract to address regional infrastructure concerns.")

        if support_calls >= 3:
            st.warning("📞 **Support Frustration Trigger (Rank #4):** High call volume indicates unresolved issues.")
            st.markdown("- **Action:** Escalate to the **senior retention team** for personal follow-up and definitive resolution.")

        if contract in ['6-Month', 'Month-to-month']:
            st.info("🤝 **Contract Trigger:** Short-term contract increases risk.")
            st.markdown("- **Action:** Offer a strong value incentive to upgrade to a stable 12-Month or 24-Month contract.")
    else:
        st.info("No immediate retention action required. Continue monitoring. Focus resources on high-risk customers.")

# --- Footer: Project and Model Information ---
st.sidebar.markdown("---")
st.sidebar.caption("Project: Ethio Telecom Churn Analysis")
st.sidebar.caption(f"Model: XGBoost Classifier (Loaded from {MODEL_FILE})")


# In[ ]:




