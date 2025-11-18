import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)


def data_generation_10(nRows, fraudRatio, printToFile = False):
    '''Generates data with 10 features.
    
    nRows: number of data row to generate.
    fraudRatio: the anount of the data to be generated to have a positive fraud result.
    printToFile: if True prints data to file fraud_detection_data.csv
    
    returns generated data'''

# Generate features
    transactionAmount = np.random.exponential(scale=100, size=nRows).round(2)
    customerAge = np.random.randint(18, 80, size=nRows)
    transactionType = np.random.choice(['purchase', 'withdrawal', 'transfer'], size=nRows)
    deviceType = np.random.choice(['mobile', 'desktop', 'ATM'], size=nRows)
    accountBalance = np.random.normal(2000, 1000, size=nRows).clip(min=0).round(2)
    numPrevTransactions = np.random.poisson(5, size=nRows)
    country = np.random.choice(['US', 'UK', 'DE', 'IN', 'NG', 'CN'], size=nRows)
    isForeignTransaction = np.random.choice(['Y', 'N'], size=nRows, p=[0.1, 0.9])
    merchantRiskScore = np.random.uniform(0, 1, size=nRows).round(2)
    timeOfDay = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=nRows)

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
        np.random.normal(0, 0.05, size=nRows)
    )

# Normalize and apply threshold for fraud label
    prob_fraud = (prob_fraud - prob_fraud.min()) / (prob_fraud.max() - prob_fraud.min())
    threshold = np.quantile(prob_fraud, 1 - fraudRatio)
    dataFraud['is_fraud'] = (prob_fraud > threshold).astype(int)

# Save to CSV
    if printToFile:
        dataFraud.to_csv("fraud_detection_data.csv", index=False)
    return  dataFraud




def data_generation_5(nRows, fraudRatio, printToFile = False):
    '''Generates data with 5 features.
    
    nRows: number of data row to generate.
    fraudRatio: the anount of the data to be generated to have a positive fraud result.
    printToFile: if True prints data to file fraud_detection_data.csv
    
    returns generated data'''


# Generate features
    transactionAmount = np.random.exponential(scale=100, size=nRows).round(2)
    transactionType = np.random.choice(['purchase', 'withdrawal', 'transfer'], size=nRows)
    accountBalance = np.random.normal(2000, 1000, size=nRows).clip(min=0).round(2)
    merchantRiskScore = np.random.uniform(0, 1, size=nRows).round(2)
    timeOfDay = np.random.choice(['morning', 'afternoon', 'evening', 'night'], size=nRows)

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
        np.random.normal(0, 0.05, size=nRows)
    )

# Normalize and apply threshold for fraud label
    prob_fraud = (prob_fraud - prob_fraud.min()) / (prob_fraud.max() - prob_fraud.min())
    threshold = np.quantile(prob_fraud, 1 - fraudRatio)
    dataFraud['is_fraud'] = (prob_fraud > threshold).astype(int)

# Save to CSV
    if printToFile:
        dataFraud.to_csv("fraud_detection_data.csv", index=False)
    return  dataFraud