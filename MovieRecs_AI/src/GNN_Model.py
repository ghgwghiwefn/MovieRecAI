import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, NNConv, SAGEConv, CGConv, global_mean_pool
import HyperParameters

### HYPER PARAMETERS ###
HIDDEN_UNITS = HyperParameters.HIDDEN_UNITS
DROPOUT_RATE = HyperParameters.DROPOUT_RATE
INPUT_DIM = HyperParameters.input_dim
EDGE_DIM = HyperParameters.edge_dim
NUM_HEADS = HyperParameters.NUM_HEADS
device = HyperParameters.device

class GNN(torch.nn.Module):
    def __init__(self, 
                 input_dim=INPUT_DIM, 
                 edge_dim=EDGE_DIM,
                 hidden_dim=HIDDEN_UNITS,
                 num_heads=NUM_HEADS,
                 dropout_rate=DROPOUT_RATE,
                 post_head_dim=NUM_HEADS*HIDDEN_UNITS):
        super(GNN, self).__init__()
        # Define the neural network for NNConv
        '''self.nn_conv_nn = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),  # Edge feature transformation
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )'''
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.gatconv1 = GATv2Conv(input_dim, hidden_dim, edge_dim=edge_dim, heads=num_heads, dropout=dropout_rate)
        self.gatconv2 = GATv2Conv(post_head_dim, hidden_dim, edge_dim=edge_dim, heads=num_heads, dropout=dropout_rate)
        '''self.nnConv1 = NNConv(post_head_dim, 
                              hidden_dim,
                              nn=self.nn_conv_nn,
                              aggr='mean')'''
        self.sageconv1 = SAGEConv(2 * post_head_dim, hidden_dim)
        self.sageconv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, edge_attr, y, batch):
        valid_ratings_mask = y != -1 
        random_mask = (torch.rand_like(valid_ratings_mask, dtype=torch.float) < 0.2) & valid_ratings_mask
        y[random_mask] = -2  # Set the selected masked values to -2
        '''print(f"Len y: {len(y)}")
        print(f"Len random mask: {len(random_mask)}")
        print(f"Len x: {len(x)}")'''
        edge_attr[:, 0] = y

        # Ensure features for edges, not just nodes
        x = F.relu(self.gatconv1(x, edge_index, edge_attr))
        x = F.relu(self.gatconv2(x, edge_index, edge_attr))
        row, col = edge_index  # Get the two nodes that each edge connects
        edge_feature = torch.cat([x[row], x[col]], dim=-1)  # Concatenate the features of the two nodes for each edge
        edge_feature = self.dropout(edge_feature)

        #print(f"Len edge_feature after gatconv layers: {len(edge_feature)}")
        edge_feature = self.sageconv1(edge_feature, edge_index) 
        edge_feature = self.dropout(edge_feature) 
        edge_feature = self.sageconv2(edge_feature, edge_index)
        # Continue as you would normally for per-edge prediction
        edge_feature = self.fc1(edge_feature)  # Apply linear layer
        y_pred = self.fc2(edge_feature)  # Final prediction layer: [num_edges, 1]

        # Apply an activation function (e.g., sigmoid or tanh)
        y_pred = torch.sigmoid(y_pred)

        # Rescale to [1, 5]
        y_pred = y_pred * 4 + 1  # This applies for sigmoid (output range [0, 1])
        # Extract the predicted ratings for the masked (missing) values
        y_preds = y_pred[random_mask]  # Get the predictions for masked ratings

        return y_preds, random_mask