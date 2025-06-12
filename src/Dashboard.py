import streamlit as st
import pandas as pd
import xgboost as xgb
import time
from sdv.single_table import CTGANSynthesizer
from collections import deque

# Load GAN model
model = CTGANSynthesizer.load("../Models/ctgan_model_CPU.pkl")

# Load XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("../Models/xgboost_model.json")

# Expected one-hot encoded columns
EXPECTED_ONEHOT_COLS = [
    "transaction_type_CASH_OUT",
    "transaction_type_PAYMENT",
    "transaction_type_TRANSFER",
    "transaction_type_CASH_IN",
    "transaction_type_DEBIT"
]

# Preprocessing function (must match training pipeline)
def preprocess_sample(sample):
    df = pd.DataFrame([sample])
    df = df.drop(["newbalanceOrig", "newbalanceDest", "isFlaggedFraud"], axis=1, errors="ignore")
    df["isMerchantTransOrig"] = df["nameOrig"].str.startswith('M').astype(int)
    df["isMerchantTransDest"] = df["nameDest"].str.startswith('M').astype(int)
    df["isMerchantTrans"] = df["isMerchantTransOrig"] | df["isMerchantTransDest"]
    df = df.drop(['nameOrig', 'nameDest'], axis=1, errors="ignore")
    df = pd.get_dummies(df, prefix='transaction_type', columns=['type'])

    for col in EXPECTED_ONEHOT_COLS:
        if col not in df.columns:
            df[col] = 0

    ordered_columns = ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest',
                       'isMerchantTransOrig', 'isMerchantTransDest', 'isMerchantTrans'] + EXPECTED_ONEHOT_COLS
    df = df[ordered_columns]

    # Rename columns to match XGBoost training
    df.columns = [str(i) for i in range(1, len(df.columns) + 1)]
    return df

# Stream generator
def data_stream(model, batch_size=1, sleep_time=1):
    while True:
        synthetic_data = model.sample(batch_size)
        yield synthetic_data
        time.sleep(sleep_time)

# Streamlit UI
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Real-Time Fraud Detection Dashboard")

# Sidebar indicators
st.sidebar.title("📊 Live Metrics")
total_tx_placeholder = st.sidebar.metric("Total Transactions", "0")
fraud_count_placeholder = st.sidebar.metric("Fraudulent", "0")
nonfraud_count_placeholder = st.sidebar.metric("Legitimate", "0")
fraud_rate_placeholder = st.sidebar.metric("Fraud Rate", "0.00%")

# Main UI placeholders
placeholder_info = st.empty()
placeholder_table = st.empty()

# State tracking
history = deque(maxlen=10)
fraud_total = 0
nonfraud_total = 0
amounts = []

stream = data_stream(model, batch_size=1, sleep_time=1)

for synthetic_batch in stream:
    sample_dict = synthetic_batch.to_dict(orient="records")[0]
    preprocessed = preprocess_sample(sample_dict)
    dmatrix = xgb.DMatrix(preprocessed)
    prediction = xgb_model.predict(dmatrix)[0]
    is_fraud = prediction > 0.5

    # Update counters
    amounts.append(sample_dict['amount'])
    if is_fraud:
        fraud_total += 1
    else:
        nonfraud_total += 1
    total_transactions = fraud_total + nonfraud_total
    
    fraud_rate = (fraud_total / total_transactions) * 100 if total_transactions > 0 else 0

    # Update sidebar metrics
    total_tx_placeholder.metric("Total Transactions", total_transactions)
    fraud_count_placeholder.metric("Fraudulent", fraud_total)
    nonfraud_count_placeholder.metric("Legitimate", nonfraud_total)
    fraud_rate_placeholder.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    # Show transaction info
    placeholder_info.markdown(f"""
    ### 🔄 New Transaction
    **Amount:** {sample_dict['amount']:.2f}  
    **Type:** {sample_dict['type']}  
    **From:** {sample_dict['nameOrig']}  
    **To:** {sample_dict['nameDest']}  
    **Predicted Fraud Probability:** `{prediction:.4f}`  
    **Is Fraudulent?** {'✅ YES' if is_fraud else '❌ NO'}
    """)

    # Add to history for table display
    sample_display = sample_dict.copy()
    sample_display["fraud_probability"] = round(prediction, 4)
    sample_display["is_fraud"] = "YES" if is_fraud else "NO"
    history.append(sample_display)

    # Show only latest table data
    placeholder_table.dataframe(pd.DataFrame(list(history)))
