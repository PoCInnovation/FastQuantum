import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json
import numpy as np
import time


class QAOAPredictorGAT(nn.Module):
    """
    GNN using Graph Attention Networks (GAT) to predict optimal QAOA parameters
    GAT learns which nodes/edges are important for determining optimal parameters
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, p_layers=1, 
                 attention_heads=8, dropout=0.3):
        super(QAOAPredictorGAT, self).__init__()
        
        self.num_layers = num_layers
        self.p_layers = p_layers
        self.attention_heads = attention_heads
        
        # GAT layers with multi-head attention
        self.convs = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.convs.append(GATConv(
            input_dim, 
            hidden_dim // attention_heads,  # Each head outputs hidden_dim//heads
            heads=attention_heads,
            dropout=dropout,
            concat=True  # Concatenate heads
        ))
        
        # Intermediate layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_dim,
                hidden_dim // attention_heads,
                heads=attention_heads,
                dropout=dropout,
                concat=True
            ))
        
        # Last layer: hidden_dim -> hidden_dim (average heads instead of concat)
        self.convs.append(GATConv(
            hidden_dim,
            hidden_dim,
            heads=attention_heads,
            dropout=dropout,
            concat=False  # Average heads for final layer
        ))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Graph-level pooling (combine mean and max for richer representation)
        self.use_dual_pooling = True
        
        # MLP head for parameter prediction
        if self.use_dual_pooling:
            self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 because of mean+max pooling
        else:
            self.fc1 = nn.Linear(hidden_dim, 128)
            
        self.fc2 = nn.Linear(128, 64)
        self.fc_gamma = nn.Linear(64, p_layers)
        self.fc_beta = nn.Linear(64, p_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT layers with residual connections
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            
            # Batch normalization
            x_new = self.batch_norms[i](x_new)
            x_new = F.elu(x_new)  # ELU works well with GAT
            x_new = self.dropout(x_new)
            
            # Residual connection (skip connection)
            if i > 0 and x.shape[1] == x_new.shape[1]:
                x = x + x_new  # Residual
            else:
                x = x_new
        
        # Graph-level pooling (dual pooling: mean + max)
        if self.use_dual_pooling:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x = global_mean_pool(x, batch)
        
        # MLP head
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        
        # Predict gamma and beta
        gamma = self.fc_gamma(x)
        beta = self.fc_beta(x)
        
        # Concatenate predictions
        output = torch.cat([gamma, beta], dim=1)
        
        return output
    
    def get_attention_weights(self, data):
        # Get the device of the model
        device = next(self.parameters()).device
        
        # Move data to the same device as the model
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device) if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        
        attention_weights = []
        
        with torch.no_grad():
            for i, conv in enumerate(self.convs):
                # Get attention weights from GAT layer
                x_new, (edge_idx, alpha) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx.cpu(), alpha.cpu()))
                
                x_new = self.batch_norms[i](x_new)
                x_new = F.elu(x_new)
                
                if i > 0 and x.shape[1] == x_new.shape[1]:
                    x = x + x_new
                else:
                    x = x_new
        
        return attention_weights


class QAOAPredictorGCN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, p_layers=1, dropout=0.3):
        super(QAOAPredictorGCN, self).__init__()
        
        self.num_layers = num_layers
        self.p_layers = p_layers
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # MLP head
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 for dual pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc_gamma = nn.Linear(64, p_layers)
        self.fc_beta = nn.Linear(64, p_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            if i > 0 and x.shape[1] == x_new.shape[1]:
                x = x + x_new
            else:
                x = x_new
        
        # Dual pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # MLP head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        gamma = self.fc_gamma(x)
        beta = self.fc_beta(x)
        
        output = torch.cat([gamma, beta], dim=1)
        return output


class DatasetLoader:
    """
    Load and prepare dataset generated by QAOADataGenerator for GNN training
    """
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.dataset = json.load(f)
        
        if len(self.dataset) > 0:
            self.p_layers = len(self.dataset[0]['optimal_gamma'])
        else:
            self.p_layers = 1
    
    def to_pyg_data(self, sample_dict):
        """
        Convert sample dictionary to PyTorch Geometric Data object
        """
        x = torch.tensor(sample_dict['node_features'], dtype=torch.float)
        
        adj_matrix = np.array(sample_dict['adjacency_matrix'])
        edge_index = torch.tensor(np.array(np.where(adj_matrix > 0)), dtype=torch.long)
        
        gamma = sample_dict['optimal_gamma']
        beta = sample_dict['optimal_beta']
        
        if not isinstance(gamma, list):
            gamma = [gamma]
        if not isinstance(beta, list):
            beta = [beta]
            
        y = torch.tensor(gamma + beta, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.n_nodes = sample_dict['n_nodes']
        data.n_edges = sample_dict['n_edges']
        data.graph_type = sample_dict['graph_type']
        
        return data
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        """
        Create PyTorch Geometric DataLoader
        """
        pyg_data_list = [self.to_pyg_data(sample) for sample in self.dataset]
        return DataLoader(pyg_data_list, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        batch_size = out.shape[0]
        target = batch.y.view(batch_size, -1)
        
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    """
    Evaluate model
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            
            batch_size = out.shape[0]
            target = batch.y.view(batch_size, -1)
            
            loss = criterion(out, target)
            
            total_loss += loss.item()
            n_batches += 1
            
            predictions.append(out.cpu())
            targets.append(target.cpu())
    
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return total_loss / n_batches, predictions, targets


def compute_parameter_errors(predictions, targets, p_layers):
    """
    Compute separate errors for gamma and beta parameters
    """
    pred_gamma = predictions[:, :p_layers]
    pred_beta = predictions[:, p_layers:]
    target_gamma = targets[:, :p_layers]
    target_beta = targets[:, p_layers:]
    
    gamma_mae = torch.abs(pred_gamma - target_gamma).mean().item()
    beta_mae = torch.abs(pred_beta - target_beta).mean().item()
    
    return gamma_mae, beta_mae


def count_parameters(model):
    """
    Count trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Training pipeline comparing GAT vs GCN
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load datasets
    train_path = "Dataset/qaoa_train_dataset.json"
    val_path = "Dataset/qaoa_val_dataset.json"
    
    print("Loading datasets...")
    train_loader_obj = DatasetLoader(train_path)
    val_loader_obj = DatasetLoader(val_path)
    
    p_layers = train_loader_obj.p_layers
    print(f"QAOA depth (p): {p_layers}")
    print(f"Training samples: {len(train_loader_obj.dataset)}")
    print(f"Validation samples: {len(val_loader_obj.dataset)}\n")
    
    # Create dataloaders
    train_loader = train_loader_obj.get_dataloader(batch_size=16, shuffle=True)
    val_loader = val_loader_obj.get_dataloader(batch_size=16, shuffle=False)
    
    # model architecture
    model_type = 'GAT'
    
    if model_type == 'GAT':
        model = QAOAPredictorGAT(
            input_dim=7,
            hidden_dim=64,
            num_layers=3,
            p_layers=p_layers,
            attention_heads=8,
            dropout=0.3
        ).to(device)
        print("🧠 Using GAT (Graph Attention Network) with 7 node features")
    else:
        model = QAOAPredictorGCN(
            input_dim=7,
            hidden_dim=64,
            num_layers=3,
            p_layers=p_layers,
            dropout=0.3
        ).to(device)
        print("🧠 Using GCN (Graph Convolutional Network) with 7 node features")
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}\n")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    
    # Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 50
    
    start_time = time.time()
    
    for epoch in range(300):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        gamma_mae, beta_mae = compute_parameter_errors(val_preds, val_targets, p_layers)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'p_layers': p_layers,
                'model_type': model_type,
                'gamma_mae': gamma_mae,
                'beta_mae': beta_mae
            }, f'best_qaoa_{model_type.lower()}_model.pt')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"γ MAE: {gamma_mae:.6f} | "
                  f"β MAE: {beta_mae:.6f}")
        
        if patience_counter >= max_patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ Training completed in {training_time/60:.2f} minutes!")
    print(f"📊 Best validation loss: {best_val_loss:.6f}")
    print(f"🎯 Best γ MAE: {gamma_mae:.6f}")
    print(f"🎯 Best β MAE: {beta_mae:.6f}")
    print(f"💾 Model saved: best_qaoa_{model_type.lower()}_model.pt")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()