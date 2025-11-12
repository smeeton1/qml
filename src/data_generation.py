import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)


def data_generation_10(n_rows, fraud_ratio, printToFile = False):


# Generate features
    transactionAmount = np.random.exponential(scale=100, size=n_rows).round(2)
    customerAge = np.random.randint(18, 80, size=n_rows)
    transactionType = np.random.choice(['purchase', 'withdrawal', 'transfer'], size=n_rows)
    deviceType = np.random.choice(['mobile', 'desktop', 'ATM'], size=n_rows)
    accountBalance = np.random.normal(2000, 1000, size=n_rows).clip(min=0).round(2)
    numPrevTransactions = np.random.poisson(5, size=n_rows)
    country = np.random.choice(['US', 'UK', 'DE', 'IN', 'NG', 'CN'], size=n_rows)
    isForeignTransaction = np.random.choice(['Y', 'N'], size=n_rows, p=[0.1, 0.9])
    merchantRiskScore = np.random.uniform(0, 1, size=n_rows).round(2)
    timeOfDay = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=n_rows)

# Create DataFrame
    dataFraud = pd.DataFrame({
        'transaction_amount': transactionAmount,
        'customer_age': customerAge,
        'transaction_type': transactionType,
        'device_type': deviceType,
        'account_balance': accountBalance,
        'num_prev_transactions': numPrevTransactions,
        'country': country,
        'is_foreign_transaction': isForeignTransaction,
        'merchant_risk_score': merchantRiskScore,
        'time_of_day': timeOfDay
    })

# Generate fraud probabilities
    prob_fraud = (
        0.001 * dataFraud['transaction_amount'] +
        0.3 * (dataFraud['is_foreign_transaction'] == 'Y').astype(int) +
        0.5 * dataFraud['merchant_risk_score'] +
        np.random.normal(0, 0.05, size=n_rows)
    )

# Normalize and apply threshold for fraud label
    prob_fraud = (prob_fraud - prob_fraud.min()) / (prob_fraud.max() - prob_fraud.min())
    threshold = np.quantile(prob_fraud, 1 - fraud_ratio)
    dataFraud['is_fraud'] = (prob_fraud > threshold).astype(int)

# Save to CSV
    if printToFile:
        dataFraud.to_csv("fraud_detection_data.csv", index=False)
    return  dataFraud




def data_generation_5(n_rows, fraud_ratio, printToFile = False):


# Generate features
    transactionAmount = np.random.exponential(scale=100, size=n_rows).round(2)
    transactionType = np.random.choice(['purchase', 'withdrawal', 'transfer'], size=n_rows)
    accountBalance = np.random.normal(2000, 1000, size=n_rows).clip(min=0).round(2)
    merchantRiskScore = np.random.uniform(0, 1, size=n_rows).round(2)
    timeOfDay = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=n_rows)

# Create DataFrame
    dataFraud = pd.DataFrame({
        'transaction_amount': transactionAmount,
        'transaction_type': transactionType,
        'account_balance': accountBalance,
        'merchant_risk_score': merchantRiskScore,
        'time_of_day': timeOfDay
    })

# Generate fraud probabilities
    prob_fraud = (
        0.001 * dataFraud['transaction_amount'] +
        0.5 * dataFraud['merchant_risk_score'] +
        np.random.normal(0, 0.05, size=n_rows)
    )

# Normalize and apply threshold for fraud label
    prob_fraud = (prob_fraud - prob_fraud.min()) / (prob_fraud.max() - prob_fraud.min())
    threshold = np.quantile(prob_fraud, 1 - fraud_ratio)
    dataFraud['is_fraud'] = (prob_fraud > threshold).astype(int)

# Save to CSV
    if printToFile:
        dataFraud.to_csv("fraud_detection_data.csv", index=False)
    return  dataFraud