import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from GnnmodelGat import QAOAPredictorGAT
import os
from tqdm import tqdm

class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data_list):
        # Compute global stats for z-score norm
        all_features = torch.cat([data.x for data in data_list], dim=0)
        self.mean = all_features.mean(dim=0)
        self.std = all_features.std(dim=0)
        self.std[self.std == 0] = 1.0 # Avoid div by zero
        
    def transform(self, data_list):
        for data in data_list:
            data.x = (data.x - self.mean) / self.std
        return data_list
    
    def save_stats(self, path):
        torch.save({'mean': self.mean, 'std': self.std}, path)

def load_data(pt_path):
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{pt_path} not found")
        
    print(f"Loading {pt_path}...")
    raw = torch.load(pt_path)
    
    data_list = []
    for item in raw:
        x = torch.tensor(item['node_features'], dtype=torch.float)
        adj = np.array(item['adjacency_matrix'])
        edge_index = torch.tensor(np.where(adj > 0), dtype=torch.long)
        
        # Target: gamma + beta
        y = torch.tensor(item['optimal_gamma'] + item['optimal_beta'], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = x.shape[0]
        data_list.append(data)
        
    return data_list

def train_step(model, loader, optimizer, criterion, device, scaler):
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Mixed precision context
        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
            out = model(batch)
            target = batch.y.view(out.shape)
            loss = criterion(out, target)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
            
        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            target = batch.y.view(out.shape)
            total_loss += criterion(out, target).item()
    return total_loss / len(loader)

def get_metrics(model, loader, device, p_layers):
    model.eval()
    gamma_err = 0
    beta_err = 0
    n_graphs = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            target = batch.y.view(out.shape)
            
            pred_g = out[:, :p_layers]
            pred_b = out[:, p_layers:]
            targ_g = target[:, :p_layers]
            targ_b = target[:, p_layers:]
            
            gamma_err += torch.abs(pred_g - targ_g).sum().item()
            beta_err += torch.abs(pred_b - targ_b).sum().item()
            n_graphs += batch.num_graphs
            
    return gamma_err / n_graphs, beta_err / n_graphs

def run_finetuning(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Data setup
    train_set = load_data(args['train'])
    val_set = load_data(args['val'])
    
    # Normalization
    scaler = FeatureScaler()
    scaler.fit(train_set)
    train_set = scaler.transform(train_set)
    val_set = scaler.transform(val_set)
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if 'cuda' in str(device) else {}
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, **kwargs)

    # Model setup
    ckpt = torch.load(args['model'], map_location=device)
    p_layers = ckpt.get('p_layers', 1)
    
    model = QAOAPredictorGAT(
        input_dim=7,
        hidden_dim=64,
        num_layers=3,
        p_layers=p_layers,
        attention_heads=8,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    
    # Optimizer setup - lower LR for GAT layers
    gat_params = [p for n, p in model.named_parameters() if 'convs' in n]
    mlp_params = [p for n, p in model.named_parameters() if 'convs' not in n]
    
    opt = torch.optim.AdamW([
        {'params': gat_params, 'lr': args['lr'] * 0.2},
        {'params': mlp_params, 'lr': args['lr']}
    ], weight_decay=1e-4)
    
    grad_scaler = torch.amp.GradScaler()
    crit = nn.SmoothL1Loss()
    
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # Training loop
    best_loss = float('inf')
    patience = 25
    counter = 0
    
    print(f"Starting training (p={p_layers})...")

    for epoch in range(args['epochs']):
        train_loss = train_step(model, train_loader, opt, crit, device, grad_scaler)
        val_loss = evaluate(model, val_loader, crit, device)
        g_mae, b_mae = get_metrics(model, val_loader, device, p_layers)
        
        sched.step(val_loss)
        lr = opt.param_groups[1]['lr']
        
        saved = ""
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state': opt.state_dict(),
                'feature_mean': scaler.mean,
                'feature_std': scaler.std,
                'p_layers': p_layers
            }, args['save_path'])
            saved = "*"
        else:
            counter += 1
            
        print(f"Ep {epoch+1:03d} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"MAE(g/b): {g_mae:.4f}/{b_mae:.4f} | LR: {lr:.1e} {saved}")
        
        if counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune GAT model for QAOA")
    parser.add_argument('--model', type=str, default="best_qaoa_gat_model.pt", help="Path to pretrained model")
    parser.add_argument('--train', type=str, default="Dataset/finetune/maxcut_train_massive.pt", help="Training dataset path")
    parser.add_argument('--val', type=str, default="Dataset/finetune/maxcut_val_massive.pt", help="Validation dataset path")
    parser.add_argument('--save', type=str, default="finetuned_maxcut_model.pt", help="Path to save fine-tuned model")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    
    # Fallback logic
    if not os.path.exists(args.train):
        print(f"Dataset {args.train} not found, trying standard datasets...")
        args.train = "Dataset/finetune/maxcut_train.pt"
        args.val = "Dataset/finetune/maxcut_val.pt"
        
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found.")
        exit(1)

    config = {
        'model': args.model,
        'train': args.train,
        'val': args.val,
        'save_path': args.save,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size
    }

    run_finetuning(config)
