import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import GNN_Data_cleanup
from Graph_preprocessing_functions import convert_to_data
import HyperParameters
import Utils as U
from GNN_Model import GNN
import time
import math

device = HyperParameters.device
print(device)

# Load or preprocess data
try:
    # Load the preprocessed data stored in .pt files
    training_data = torch.load((U.CLEAN_DATA_FOLDER / 'Processed_Training_Graphs.pt').resolve())
    testing_data = torch.load((U.CLEAN_DATA_FOLDER / 'Processed_Testing_Graphs.pt').resolve())

except:
    # If the data hasn't been preprocessed, clean it, preprocess it, and save it
    print("data not found")
    GNN_Data_cleanup.clean_data()
    training_data = torch.load((U.CLEAN_DATA_FOLDER / 'Processed_Training_Graphs.pt').resolve())
    testing_data = torch.load((U.CLEAN_DATA_FOLDER / 'Processed_Testing_Graphs.pt').resolve())

#LABELS
###Finish loading data###

### HYPER PARAMETERS ###
BATCH_SIZE = HyperParameters.BATCH_SIZE
LEARNING_RATE = HyperParameters.LEARNING_RATE
EPOCHS = HyperParameters.EPOCHS

'''print(training_data[0])
print(training_data[0].x[0])
print(training_data[0].edge_attr)
print(training_data[0].edge_attr[0])
print(training_data[0].edge_attr[1])'''

#Load the data into training batches.
training_batches = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
testing_batches = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

#DECLARE MODEL INSTANCE WITH INPUT DIMENSION
# Before the model call
Model_0 = GNN() 
Model_0.to(device)
#Define loss function and optimizer
loss_fn = nn.MSELoss(reduction='none')  # Compute loss per element MSELoss for continuous prediction
optimizer = torch.optim.Adam(Model_0.parameters(), lr=LEARNING_RATE)

#Accuracy function
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

'''
Training Loop
#1. Forward Pass
#2. Calculate the loss on the model's predictions
#3. Optimizer
#4. Back Propagation using loss
#5. Optimizer step
'''

# Lists to store loss values
train_losses = []
val_losses = []

best_val_loss = float('inf')  # Initialize best validation loss as infinity
patience = HyperParameters.PATIENCE  
epochs_no_improve = 0

print(f"Num Batches: {len(training_batches)}") 
update_interval = math.ceil(len(training_batches)*.25)
test_update_interval = math.ceil(len(testing_batches)*.25)
print(update_interval)

for epoch in range(EPOCHS):
    print(f"\nEpoch: {epoch}\n---------")
    Model_0.train()
    training_loss = 0
    start_time = time.time()
    
    for batch_index, batch_graphs in enumerate(training_batches):
        i = 1
        batch_graphs = batch_graphs.to(device)

        # 1. Forward Pass:
        y = batch_graphs.edge_attr[:, 0]  # Collect all the ratings as the y values.
        y_clone = y.clone()
        # Run the model
        y_preds, random_mask = Model_0(batch_graphs.x, batch_graphs.edge_index, batch_graphs.edge_attr, y, batch_graphs.batch)
        # Squeeze y_pred to ensure it's 1D
        y_preds = y_preds.squeeze()
        # Initialize an empty list to collect the values
        actual_y = []

        # Iterate through y and random_mask simultaneously
        for orig_y, mask in zip(y_clone, random_mask):
            if mask:  # If the mask is True, append the value from y
                actual_y.append(orig_y.item())

        # Convert the list back to a tensor
        actual_y = torch.tensor(actual_y, device=y.device, dtype=y.dtype)
        '''print(f"original y 2: {y_clone[:10]}; Shape: {y_clone.shape}")
        print(f"Random Mask: {random_mask[:10]}; Shape: {random_mask.shape}")
        print(f"Masked preds: {y_preds[:10]}; Shape: {y_preds.shape}")
        print(f"y ratings: {actual_y[:10]}; Shape: {actual_y.shape}")'''

        # 2. Calculate loss
        loss = loss_fn(y_preds, actual_y)  # Calculate per-element loss
        loss = loss.mean()  # Reduce the loss to a scalar value

        training_loss += loss.item()  # Accumulate the scalar value for logging

        # 3. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()  # Frees unused memory

        if math.floor((batch_index + 1) % update_interval) == 0:
            print(f"Training loop is {int((batch_index + 1) / len(training_batches) * 100)}% finished.")

    #Post training loop
    training_loss /= len(training_batches)
    print(f"Training loss: {training_loss}")
    train_losses.append(training_loss)

    elapsed = time.time() - start_time
    print(f"Loop took {elapsed:.2f} seconds.")

    print("Testing the model...")
    testing_loss, test_acc = 0, 0
    Model_0.eval()
    with torch.inference_mode():
        for batch_index, batch_graphs in enumerate(testing_batches):
            i = 1
            batch_graphs = batch_graphs.to(device)

            # 1. Forward Pass:
            y = batch_graphs.edge_attr[:, 0]  # Collect all the ratings as the y values.
            y_clone = y.clone()
            # Run the model
            y_preds, random_mask = Model_0(batch_graphs.x, batch_graphs.edge_index, batch_graphs.edge_attr, y, batch_graphs.batch)
            # Squeeze y_pred to ensure it's 1D
            y_preds = y_preds.squeeze()
            # Initialize an empty list to collect the values
            actual_y = []

            # Iterate through y and random_mask simultaneously
            for orig_y, mask in zip(y_clone, random_mask):
                if mask:  # If the mask is True, append the value from y
                    actual_y.append(orig_y.item())

            # Convert the list back to a tensor
            actual_y = torch.tensor(actual_y, device=y.device, dtype=y.dtype)

            # 2. Calculate loss
            loss = loss_fn(y_preds, actual_y)  # Calculate per-element loss
            loss = loss.mean()  # Reduce the loss to a scalar value

            testing_loss += loss.item()  # Accumulate the scalar value for logging
            if math.floor((batch_index + 1) % test_update_interval) == 0:
                print(f"Testing loop is {int((batch_index + 1) / len(testing_batches) * 100)}% finished.")
    testing_loss /= len(testing_batches)
    print(f"Testing loss: {testing_loss}")


# Plotting the loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
