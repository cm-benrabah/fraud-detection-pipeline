from sdv.single_table import CTGANSynthesizer  # or TVAESynthesizer, CopulaGANSynthesizer
import pandas as pd
import time
import pickle  # if you used pickle to save/load

# Example if you saved using .save()
model = CTGANSynthesizer.load("../Models/ctgan_model_CPU.pkl")

# OR if you saved using pickle
# with open("ctgan_model.pkl", "rb") as f:
#     model = pickle.load(f)


def data_stream(model, batch_size=1, sleep_time=1):
    while True:
        synthetic_data = model.sample(batch_size)
        yield synthetic_data
        time.sleep(sleep_time)


stream = data_stream(model, batch_size=1, sleep_time=1)

for synthetic_batch in stream:
    print(synthetic_batch.to_dict(orient="records")[0])  # simulate one event per second




import pandas as pd

# The set of expected one-hot columns from training 
EXPECTED_ONEHOT_COLS = [ 
    "transaction_type_CASH_OUT",
    "transaction_type_PAYMENT",
    "transaction_type_TRANSFER",
    "transaction_type_CASH_IN",       # Include all types seen during training
    "transaction_type_DEBIT"
]

def preprocess_sample(sample):
    df = pd.DataFrame([sample])

    # Drop columns
    df = df.drop(["newbalanceOrig", "newbalanceDest", "isFlaggedFraud"], axis=1, errors="ignore")

    # Add isMerchantTrans columns
    df["isMerchantTransOrig"] = df["nameOrig"].str.startswith('M').astype(int)
    df["isMerchantTransDest"] = df["nameDest"].str.startswith('M').astype(int)
    df["isMerchantTrans"] = df["isMerchantTransOrig"] | df["isMerchantTransDest"]

    # Drop names
    df = df.drop(['nameOrig', 'nameDest'], axis=1, errors="ignore")

    # One-hot encode transaction type
    df = pd.get_dummies(df, prefix='transaction_type', columns=['type'])

    # Add missing one-hot columns
    for col in EXPECTED_ONEHOT_COLS:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order matches what was used during training
    # Drop target column if present (it shouldn't be during inference)
    if 'isFraud' in df.columns:
        df = df.drop("isFraud", axis=1)

    # Order columns to match training order
    ordered_columns = ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest',
                       'isMerchantTransOrig', 'isMerchantTransDest', 'isMerchantTrans'] + EXPECTED_ONEHOT_COLS
    df = df[ordered_columns]
    
    return df