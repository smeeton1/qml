import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


# Class for a simple nn model with two internal layers.
class MediNN(nn.Module):
    '''Class to set up a pytorch model with two hidden layers.
        
        input_dim: number of parameters for the data.
    '''
    def __init__(self, input_dim):
        super(MediNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  
        self.fc2 = nn.Linear(64, 32) 
        self.fc3 = nn.Linear(32, 16)        
        self.fc4 = nn.Linear(16, 2)          

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        x = torch.relu(self.fc3(x))  
        x = self.fc4(x)             
        return x


# Read in data file and prepare the data for training. Returns Training set and Testing set.
def read_prepare_data(fileName, para, targetName):
    '''Prepares data for training and testing.
    
    fileName: the file containing the data.
    para: the names of the features.
    targetName: name of the target column.
    
    returns the training data, the test data, the traing data with out the target, and test target data.
    '''
    data = pd.read_csv(fileName)
    encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    encodedFeatures = encoder.fit_transform(data[para])
    encodedData = pd.DataFrame(encodedFeatures, columns=encoder.get_feature_names_out(para))
    encodedData[targetName] = data[targetName].map({'Yes': 1, 'No': 0})
    paramiter = encodedData.drop(columns=targetName).values
    target = encodedData[targetName].values
    paramiterTensor = torch.tensor(paramiter, dtype=torch.float32)
    targetTensor = torch.tensor(target, dtype=torch.long)
    paramiterTrain, paramiterTest, targetTrain, targetTest = train_test_split(paramiterTensor , targetTensor)
    trainData = TensorDataset(paramiterTrain, targetTrain)
    testData = TensorDataset(paramiterTest, targetTest)
    trainLoader = DataLoader(trainData, batch_size=2, shuffle=True)
    testLoader = DataLoader(testData, batch_size=2, shuffle=False)
    return trainLoader, testLoader, paramiterTrain, targetTest


# Train classical model. epochs is the number of training epochs. Returns the trained model
def train_model(model, trainLoader, epochs, tol):
    '''Train a pytorch ai model.
    
    model:       model to be trained.
    trainLoader: the data to train the model with.
    epochs: the number of training runs to preforms.
    tol: that amount to of the previous gradiant to keep.
    
    returns a trained model.
    '''
    
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=tol)

    for epoch in range(epochs):
        model.train()
        runningLoss = 0.0

        for inputs, targets in trainLoader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()

        #if (epoch + 1) % 10 == 0:
        #    print(f"Epoch [{epoch+1}/{epochs}], Loss: {runningLoss/len(trainLoader):.4f}")

    return model


# Test the model. Returns the accuracy.
def test_model(model, testLoader, testTarget):
    '''Test a trained model.

model:      model to test
testLoader: Data to use for testing
testTarget: The results for the test data.

returns the accuracy, precision, recall, f1, and auc
    '''


    model.eval()  
    targetPred = []
    targetTrue = []

    with torch.no_grad():
        for inputs, targets in testLoader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  
            targetPred.extend(predicted.cpu().numpy())
            targetTrue.extend(targets.cpu().numpy()) 

    
    targetTrue = testTarget.numpy()

    accuracy = accuracy_score(targetTrue, targetPred)
    precision = precision_score(targetTrue, targetPred)
    recall = recall_score(targetTrue, targetPred)
    f1 = f1_score(targetTrue, targetPred)
    auc = roc_auc_score(targetTrue, targetPred)

    return accuracy, precision, recall, f1, auc