import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def prepare_data_cluster_cl(data, features):
    '''Prepares data for classiacl Ai clustering.

    data: the data to be prepared. Needs to be in pandas data frame format
    feature: list of string containing the names of the features for the data
    
    returns a pytorch tensor.'''
    encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    encodedFeatures = encoder.fit_transform(data[features])
    # Convert to DataFrame
    encodedData = pd.DataFrame(encodedFeatures, columns=encoder.get_feature_names_out(features))
    # Standardizing data (important for model performance)
    scaler = StandardScaler()
    dataScaled = scaler.fit_transform(encodedData)
    # Convert to PyTorch tensor
    dataTensor = torch.tensor(dataScaled, dtype=torch.float32)
    return dataTensor


class Autoencoder(nn.Module):
    '''Model to auto cluster data
    
    imputDim: the number of features for the data
    latentDim: number of classifications'''
    def __init__(self, inputDim=6, latentDim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputDim, 4),
            nn.ReLU(),
            nn.Linear(4, latentDim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latentDim, 4),
            nn.ReLU(),
            nn.Linear(4, inputDim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

  
def train_auto_coder(model, dataTensor, epochs):
    '''Function to encode the data in to an Ai model.
    
    model: the model to be trained.
    dataTensor: the data to be analysised
    epochs: number of epochs to preform
    
    returns the trained model.'''
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(dataTensor)):
            x = dataTensor[i].unsqueeze(0)  # Add batch dimension
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #if (epoch+1) % 10 == 0:
            #print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_tensor):.4f}")
    return model


def pattern_discover(model, dataTensor):
    '''Function to find patterns in model. Returns a plot of the clustering.
    
    model: trained model
    dataTensor: data to be classified
    
    returns a plot and vizulization data.'''
    with torch.no_grad():
        latent_features = model.encoder(dataTensor).numpy()

    # Visualize the latent space (2D)
    dfViz = pd.DataFrame(latent_features, columns=["z1", "z2"])

    # Plot the latent features
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=dfViz, x="z1", y="z2", s=60)
    plt.title("Latent Space Exploration (Autoencoder)")
    plt.savefig("Latent Space Exploration (Autoencoder)")
    plt.show()
    plt.close()

    return dfViz


def anomaly_detection(model, dataTensor, dfViz):
    '''Calculates the error of the model.
     
    model: trained model.
    dataTemsor: data being analysised.
    
    returns a plot and the outliers.'''
    with torch.no_grad():
        reconstructions = model(dataTensor)
        reconstruction_error = torch.mean((reconstructions - datatensor) ** 2, dim=1).numpy()
    # Define a threshold for anomaly (e.g., top 10% as outliers)
    threshold = np.percentile(reconstruction_error, 90)
    # Find anomalies (outliers) based on reconstruction error
    outliers = np.where(reconstruction_error > threshold)[0]
    dfViz["Reconstruction Error"] = reconstruction_error
    dfViz["Anomaly"] = dfViz["Reconstruction Error"] > threshold

    # Plot anomalies
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=dfViz, x="z1", y="z2", hue="Anomaly", palette={True: 'red', False: 'blue'}, s=60)
    plt.title("Latent Space with Anomalies Highlighted")
    plt.savefig("Latent Space with Anomalies Highlighted")
    plt.show()
    plt.close()
    return outliers