import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import COBYLA


def prepare_data(data):
    '''Function to prepare CSV data for QNN clustering
    
    data: the data to be prepared. Needs to be in pandas data frame format
    feature: list of string containing the names of the features for the data
    
    returns a pytorch tensor.'''
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    data = pd.get_dummies(data, columns=categorical_cols)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = (y > 0).astype(int)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

def make_qnn(X, rep):
    '''Function to make a qnn circuit.
    
    X prepared data
    rep number of time the qnn circuit is to be repeated
    
    returns the classifier'''
    num_qubits = X.shape[1]

    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=rep)
    ansatz = RealAmplitudes(num_qubits, reps=rep)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )

    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=100)
    )
    return classifier

def train_qnn(classifier, X, y):
    '''Train and test accuracy of the classifier
    
    classifier to be trained
    x training data
    y training data
    
    returns trained classifier and accuracy'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")
    return classifier, accuracy

def plot_data(X, y):
    '''function to plot data
    
    X data 
    y target'''
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Dataset (after PCA)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_classifier(classifier, X, y):
    '''plots the trained classifier.
    
    classifer to be plotted.
    X, y prepared data.'''
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on grid
    Z = classifier.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("QNN Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()