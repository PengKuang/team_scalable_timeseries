####################
### Import libraries
####################

from pyspark.ml.torch.distributor import TorchDistributor 
from pyspark.context import SparkContext
from pyspark import TaskContext
from pyspark.sql.session import SparkSession
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
import time as time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs('.' + os.sep + 'test_data', exist_ok=True) 
data_folder_name = '.' + os.sep + 'test_data' + os.sep + 'out_' +  datetime.now().strftime("%d%m%y_%H%M%S")
os.makedirs(data_folder_name, exist_ok=True) 


########################
### Set the SparkContext
########################

# sc = SparkContext('spark://:7077')
# sc = SparkContext('local')
sc = SparkContext('spark://localhost:7077')
# sc = SparkContext('spark://172.18.139.208:7077')
spark = SparkSession(sc)


################
### Load dataset
################

df_raw_data = pd.read_csv('ECG5000/ecg.csv', header=None)
df_raw_data.columns = [f't{i+1}' for i in range(df_raw_data.shape[1] - 1)] + ['label'] 
torch_data_normal = torch.tensor(df_raw_data[df_raw_data['label'] == True].values[:,:-1], dtype=torch.float32)
torch_data_anomalous = torch.tensor(df_raw_data[df_raw_data['label'] == False].values[:,:-1], dtype=torch.float32)
hold_out_idx = torch.tensor(random.sample(range(torch_data_normal.shape[0]), 500))
all_indices = torch.arange(torch_data_normal.shape[0])
remaining_indices = all_indices[~torch.isin(all_indices, hold_out_idx)]
hold_out_normal = torch_data_normal[hold_out_idx,:]
torch_data_normal = torch_data_normal[remaining_indices,:]
torch.save(hold_out_normal, data_folder_name + os.sep + 'hold_out_normal.pt')
torch.save(torch_data_anomalous, data_folder_name + os.sep + 'torch_data_anomalous.pt')

input_dim = torch_data_normal.shape[1]
encoding_dim = 10

# # Uncomment to increment the size of the training data to see if the notebook is scalable
# torch_data_normal = torch.concat((torch_data_normal,torch_data_normal,torch_data_normal,torch_data_normal,torch_data_normal,torch_data_normal,
#               torch_data_normal,torch_data_normal,torch_data_normal,torch_data_normal,torch_data_normal,
#               torch_data_normal,torch_data_normal,torch_data_normal,torch_data_normal,torch_data_normal))

print(f'Training data (normal): {torch_data_normal.shape[0]}')
print(f'Anomalous data: {torch_data_anomalous.shape[0]}')
print(f'Hold-out data (normal): {hold_out_normal.shape[0]}')
print(f'Total data: {hold_out_normal.shape[0]+torch_data_anomalous.shape[0]+torch_data_normal.shape[0]}')


#################
### Split dataset
#################

train_datasets = []

num_splits = 3  
split_sizes = [torch_data_normal.shape[0] // num_splits] * (num_splits - 1) + [torch_data_normal.shape[0] - (torch_data_normal.shape[0] // num_splits) * (num_splits - 1)]
train_datasets = torch.split(torch_data_normal, split_sizes)
for tt_idx, tt in enumerate(train_datasets):
    print(f"{tt.shape[0]} data in chunk number {tt_idx+1}")


#####################
### Define the models 
#####################

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(TimeSeriesAutoencoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
        )
        
        self.decode = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
models = []
for i in range(num_splits):
    models.append(TimeSeriesAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim))


#############################
### Define training functions
#############################

def train_data_loader(split_index, datasets, batch_size):
    """This function creates the dataloader for each node.
    """
    train_dataset_split_index = TensorDataset(datasets[split_index])
    train_dataloader_split_index = DataLoader(train_dataset_split_index, batch_size=batch_size, shuffle=True)
    return train_dataloader_split_index 

def training(models, train_datasets, criterion, num_epochs, batch_size, learning_rate, print_epoch):
    # local_rank = int(os.environ.get("LOCAL_RANK")) 
    local_rank = TaskContext.get().partitionId()
    print(f"Worker {local_rank} started. Processing {len(train_datasets[local_rank])} samples.")
    
    model = models[local_rank]
    dataloader = train_data_loader(local_rank, train_datasets, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs+1):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if print_epoch != 0 and epoch % print_epoch == 0:
            print(f"Worker {local_rank} Model Epoch [{epoch}/{num_epochs}], Loss: {total_loss/len(dataloader):.6f}")
    
    current_directory = '/home/michele/aiuto'

    model_path = os.path.join(current_directory, f"ensemble_{local_rank}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Worker {local_rank} saved model to {model_path}")
    
    data_path = os.path.join(current_directory, f"train_data_{local_rank}.pt")
    torch.save(train_datasets[local_rank], data_path)
    print(f"Worker {local_rank} saved data to {data_path}")

    return 


if __name__ == "__main__":
    ###################
    ### Launch training
    ###################

    criterion = nn.MSELoss()

    num_epochs = 100
    batch_size = 32
    learning_rate = 0.01
    print_epoch = 0

    start_time = time.time()

    TorchDistributor(num_processes=num_splits, local_mode=False, use_gpu=False).run(training,
                    models, train_datasets, criterion, num_epochs, batch_size, learning_rate, print_epoch)

    print('It took ', time.time()-start_time, 's')
