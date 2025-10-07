# Ethio Telecom Churn Prediction: From Model to Action

This repository houses an end-to-end Machine Learning solution designed to combat customer churn for Ethio Telecom. The project moves beyond prediction to deliver **actionable, data-backed retention strategies** derived from deep analysis of service quality and infrastructure data.

## 🔗 Live Application & Overview

The Streamlit web application allows retention teams to input customer profiles and receive real-time churn risk predictions and customized intervention tactics.

**Live Streamlit Dashboard:** [INSERT YOUR STREAMLIT APP LINK HERE]

---

## 📊 Model Evaluation and Business Efficiency

The project focused on selecting a model that prioritizes the efficient use of the retention budget. The **XGBoost Classifier** was chosen for its superior performance on the crucial business metric: Precision.

| Metric | Score (Churn Class) | Business Implication |
| :--- | :--- | :--- |
| **Precision** | **58%** | When the model flags a customer as high-risk, it is correct **58% of the time**, ensuring that retention marketing and budget are targeted efficiently. |
| **F1-Score** | **48%** | This balanced score indicates strong overall model performance in identifying true churners while minimizing false positives. |

### Conclusion
The XGBoost Classifier is confirmed as the production model due to its high precision, which directly translates to **more effective retention spending** and a lower cost-per-retained-customer.

---

## 📈 CORE CHURN DRIVERS & FEATURE IMPORTANCE

The analysis reveals that **operational and infrastructure factors** are overwhelmingly more predictive of churn than traditional commercial factors like price or contract length.

| Rank | Feature | Importance Score | Business Insight |
| :--- | :--- | :--- | :--- |
| **#1** | **Network\_Outage\_Score\_0\_5** | **19.91%** | **Service Quality is the #1 Driver.** Frequent network outages are the most significant operational factor pushing customers to churn. |
| **#2** | **Region (Regional City & Rural)** | **~26%** | **Geographical Disparity.** Churn risk is highly concentrated outside of the main high-density area (Addis Ababa), indicating a critical regional service gap. |
| **#3** | **Network\_Technology (3G/4G/LTE)** | **~16%** | **Infrastructure Gap.** The technology gap (especially customers still on 3G) strongly influences the decision to leave, confirming network quality goes beyond simple outages. |
| **#4** | **Support\_Calls\_3Months** | **10.00%** | **Customer Experience Breakdown.** High volume of support calls (3+) suggests customers are frustrated by unresolved issues, making this a critical secondary driver. |
| **#5** | **Contract\_Type\_6-Month** | **4.16%** | **Conversion Priority.** Customers on 6-Month contracts are the highest-risk group among the commitment tiers, marking them as the primary focus for contract conversion efforts. |

---

## 💡 Strategic & Actionable Recommendations

These recommendations translate the model's insights into direct business action items.

### 1. Proactive "Network Health" Retention

* **Engineering Priority:** Dedicate **50% of the retention budget** to network stability projects in flagged high-risk regions.
* **Targeting Logic:** Automatically flag any customer with an `Outage Score ≥ 3` **AND** `Support Calls ≥ 3` in the last 3 months.
* **Retention Tactic (Automated):** Do **not** call them. Instead, issue an automatic, personalized "We fixed your recent network issue" notification with a **free data bonus** to soothe the frustration and re-establish goodwill immediately.

### 2. Accelerate Regional Digital Inclusion

* **Infrastructure Strategy:** Prioritize 4G/LTE expansion (and limited 5G rollout) specifically in **Regional Cities and Rural Areas** to directly address the geographical churn disparity.
* **Commercial Strategy:** Launch a **4G device and data subsidy program** exclusively for customers in the high-risk regions currently on 3G (or 2G). Tie the subsidy to a mandatory **12-month contract commitment** to convert unstable infrastructure into stable revenue.

### 3. Incentivize Long-Term Commitment

* **Pricing Tactic:** Focus retention efforts on converting **6-Month contract customers** (the highest risk contract group).
* **Offer:** Present a compelling price break or data upgrade to move these users immediately to a stable 12-Month or 24-Month fixed contract. The low importance of the 24-Month contract confirms that long-term contracts foster customer stability.

---

## 💻 Technical Setup and Project Structure

Follow these steps to set up and run the Streamlit application locally.

### Prerequisites
* Python 3.8+
* Git

### Installation
1.  **Clone the repository:** `git clone <repo-url>`
2.  **Navigate to the directory:** `cd ethio-telecom-churn-analyzer`
3.  **Create and activate a virtual environment** (Recommended).
4.  **Install dependencies:** `pip install -r requirements.txt`

### Running the App
Execute the following command to start the web application:
```bash
streamlit run churn_predictor_app.py
