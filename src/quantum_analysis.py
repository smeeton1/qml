import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.optimizers import  ADAM
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.connectors import TorchConnector
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize


# Read in the data and formats for qml.
def qml_read_prepare_data(fileName, targetName):
    data = pd.read_csv(fileName)
    labelEncoders = {}


    for column in data.columns[1:]:
        le = LabelEncoder()
        data[column + '_encoded'] = le.fit_transform(data[column])
        labelEncoders[column] = le  
    targetEncoded =targetName + '_encoded' 
    encodedColumns = [col for col in data.columns if col.endswith('_encoded') and col != targetEncoded ]
    paraparamiter = data[encodedColumns].values
    target = data[targetEncoded ].values
    scaler = StandardScaler()
    paramiterScaled = scaler.fit_transform(paraparamiter)
    paramiterTrain, paramiterTest, targetTrain, targetTest = train_test_split(paramiterScaled, target)
    return paramiterTrain, paramiterTest, targetTrain, targetTest


# Make a vqc model
def create_vqc(features, rep, toler):
    featureMap = ZZFeatureMap(feature_dimension=features, reps=rep)
    ansatz = RealAmplitudes(num_qubits=features, reps=rep)

    vqc = VQC(
        optimizer=ADAM(lr=0.01, tol=toler),
        feature_map=featureMap,
        ansatz=ansatz,
        sampler = Sampler(options={"shots": 200}) 
    )

    return vqc


# Runs test for vqc method.
def test_vqc(vqc, paramiterTest, targetTest):
    targetPred = vqc.predict(paramiterTest)
    accuracy = accuracy_score(targetTest, targetPred)
    precision = precision_score(targetTest, targetPred)
    recall = recall_score(targetTest, targetPred)
    f1 = f1_score(targetTest, targetPred)

    try:
        probabilities = vqc.predict_proba(paramiterTest)[:, 1]  
        auc = roc_auc_score(targetTest, probabilities)
    except AttributeError:

        auc = roc_auc_score(targetTest, targetPred)

    return accuracy, precision, recall, f1, auc


# Creates a qnn circuit.
def create_qnn(features, rep):
    featureMap = ZZFeatureMap(feature_dimension=features, reps=rep, entanglement='linear')
    ansatz = RealAmplitudes(num_qubits=features, reps=rep)
    qc = QuantumCircuit(features)
    qc.compose(featureMap, inplace=True)
    qc.compose(ansatz, inplace=True)
    estimator = Estimator()
    qnn = EstimatorQNN(estimator=estimator,
                    circuit=qc, 
                    input_params=featureMap.parameters,
                    weight_params=ansatz.parameters)
    return qnn


# A hybrid quantm classical method. Used pyTorch and qnn
class HybridModel(nn.Module):
    def __init__(self, qnn_module):
        super().__init__()
        self.qnn = qnn_module

    def forward(self, x):
        out = (self.qnn(x) + 1) / 2
        return out


# Creates the hybrid model for a qnn 
def creat_pytorch_qnn(qnn):
    torch_qnn = TorchConnector(qnn)
    model = HybridModel(torch_qnn)
    return model


# Trains the hybrid model.
def train_qnn(model, paramiterTrain, targetTrain, epochs, tol):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=tol)
    paramiterTrainNorm = np.pi * paramiterTrain / np.max(paramiterTrain)
    paramiterTrainTor = torch.tensor(paramiterTrainNorm, dtype=torch.float32)
    targetTrainTor = torch.tensor(targetTrain, dtype=torch.float32).view(-1, 1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(paramiterTrainTor )
        loss = criterion(outputs, targetTrainTor)
        loss.backward()
        optimizer.step()
        #print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")
    
    return model



# Test the hybrid model.
def test_qnn(model, paramiterTest, targetTest):
    paramiterTestNorm = np.pi * paramiterTest / np.max(paramiterTest)
    paramiterTestTor = torch.tensor(paramiterTestNorm, dtype=torch.float32)
    targetTestTor = torch.tensor(targetTest, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        targetProb = model(paramiterTestTor).numpy().flatten()
        targetPred = np.round(targetProb)

    accuracy = accuracy_score(targetTestTor, targetPred )
    precision = precision_score(targetTestTor, targetPred , zero_division=0)
    recall = recall_score(targetTestTor, targetPred, zero_division=0)
    f1 = f1_score(targetTestTor, targetPred, zero_division=0)
    try:
        auc = roc_auc_score(targetTestTor, targetProb)
    except ValueError:
        auc = float("nan")

    return accuracy, precision, recall, f1, auc