'''
This file contains three functions that is used in the primary file, model_pipeline.ipynb. 
These functions defines:
    train_data_loader : the dataloader used for training
    training : the training function that is distributed with TorchDistributor
    inference : the inference function that is distributed with TorchDistributor
This file cannot be run on itself but is rather imported in the notebook.
'''
from pyspark.ml.torch.distributor import TorchDistributor 
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
import time as time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import warnings
from datetime import datetime


def train_data_loader(split_index, datasets, batch_size):
    """
    This function creates the dataloader for each node.

    Parameters
    ----------
    split_index : (int)
        The number of the assigned node (0,1,...)
    datasets : (list)
        A list of datasets
    batch_size : (int)
        Size of the minibatch

    Returns
    -------
    A dataloader for the dataset at index split_index
    """

    train_dataset_split_index = TensorDataset(datasets[split_index]) 
    train_dataloader_split_index = DataLoader(train_dataset_split_index, batch_size=batch_size, shuffle=True)
    
    return(train_dataloader_split_index)

def training(models, train_datasets, criterion, num_epochs, batch_size, learning_rate, print_epoch, directory):
    """
    This function is used by the TorchDistributor.

    Parameters
    ----------
    models : [models]
        A list of models to be trained
    train_datasets : (list)
        A list of datasets to be used for training
    criterion : (function)
        Loss function
    num_epochs : (int)
        Number of epochs to be run
    batch_size : (int)
        Size of the minibatch
    learning_rate : (float)
        The learning rate for the optimizer
    print_epoch : (int)
        Determines if we print a loss and how often, set to 0 to omit print
    directory : (file path)
        directory to store models
    """
    # Local rank of this process (used to access the corresponding model)
    local_rank = int(os.environ["LOCAL_RANK"]) 

    # Access the model
    model = models[local_rank] 

    # Load the data for this process
    dataloader = train_data_loader(local_rank, train_datasets, batch_size) 
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1,num_epochs+1): 
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if print_epoch != 0:
            if epoch % print_epoch == 0:
                print(f"Model {local_rank} Epoch [{epoch}/{num_epochs}], Loss: {total_loss/len(dataloader):.6f}")

    ## Save the model weights
    torch.save(model.state_dict(), directory + os.sep + "ensemble_" + str(local_rank) + ".pth") # save this model

    return 



def inference(models, test_datasets, train_datasets, num_splits, criterion, directory):
    """
    Computes anomaly scores for a specific chunk of a new dataset. To do this the function 
    first computes the ECDF on its corresponding training data. The second part consist of 
    letting the test data stay in each node, but letting the nodes loop over the different
    models such that in the end every model has infered the whole test data. This minimizes
    the data transfer as the test data is only once sent to the corresponding machine and we 
    only need to transfer the models. Finally the scores are saved.

    Parameters
    ----------
    models : (list)
       List of trained models
    test_datasets : (list)
        List of chunks of new data
    train_datasets : (list)
        List of chunks of old data
    num_splits : (int)
        Number of models.
    criterion: (function)
        Metric used to compute reconstruction error.
    directory: (file path)
        directory to store results
    """
    # Local rank of this process (used to access the corresponding model)
    local_rank = int(os.environ["LOCAL_RANK"]) 

    ## The first part - Compute reconstruction losses on training data
    # Load model
    model = models[local_rank] 
    # Set to evaluation mode
    model.eval() 
    normal_loss = []
    with torch.no_grad():
        for j in range(train_datasets[local_rank].shape[0]): 
            # Compute reconstruction loss for each data points in the training data
            normal_loss.append(criterion(model(train_datasets[local_rank][j,:]), train_datasets[local_rank][j,:]).numpy())

    # Computed the ECDF 
    normal_ecdf = stats.ecdf(normal_loss)
    
    ## Second part - evaluate the test data on the ECDF
    # Store the anomaly scores (ECDF - values) in a matrix 
    result_all = np.zeros((test_datasets[local_rank].shape[0], num_splits))

    # Load the test data at the current machine, fixed 
    local_test_data = test_datasets[local_rank]
    
    
    with torch.no_grad():
        # We loop over the different models while keep the test data fixed
        for i in range(num_splits): 
            # Load corresponding model
            model = models[i] 
            
            # Set to evaluation mode
            model.eval() 

            # Evaluate the corresponding model on the local test data
            reconstructed_new_patients = model(local_test_data)
            
            # Store the reconstruction loss
            new_patients_loss = []

            # Compute reconstruction loss for each time series in the test data set
            for j in range(test_datasets[local_rank].shape[0]): 
                new_patients_loss.append(criterion(reconstructed_new_patients[j,:], local_test_data[j,:]).numpy())

            # Evaluate the ECDF on the reconstruction losses of the test data
            result = normal_ecdf.cdf.evaluate(new_patients_loss) 
            
            result_all[:,i] = result
            

    # Save the results in directory
    np.save(directory + os.sep + "inference_" + str(local_rank) + ".npy", result_all)