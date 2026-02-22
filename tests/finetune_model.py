import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import json
import numpy as np
import time
from pathlib import Path
import argparse

# Import model architecture and dataset loader
from GnnmodelGat import QAOAPredictorGAT, DatasetLoader, train_epoch, evaluate, compute_parameter_errors
from generate_dataset import QAOADataGeneratorImproved


class FineTuner:
    """
    Fine-tune a pre-trained QAOA GNN model on a specific problem instance
    """
    def __init__(self, pretrained_model_path='best_qaoa_gat_model.pt'):
        """
        Initialize the fine-tuner
        
        Args:
            pretrained_model_path: Path to the pre-trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}\n")
        
        # Load pre-trained model
        print(f"📥 Loading pre-trained model from {pretrained_model_path}...")
        checkpoint = torch.load(pretrained_model_path, map_location=self.device)
        
        self.p_layers = checkpoint['p_layers']
        self.model_type = checkpoint.get('model_type', 'GAT')
        
        # Initialize model with same architecture
        self.model = QAOAPredictorGAT(
            input_dim=7,
            hidden_dim=64,
            num_layers=3,
            p_layers=self.p_layers,
            attention_heads=8,
            dropout=0.3
        ).to(self.device)
        
        # Load pre-trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Model type: {self.model_type}")
        print(f"   - QAOA depth (p): {self.p_layers}")
        print(f"   - Pre-trained validation loss: {checkpoint.get('val_loss', 'N/A')}")
        if 'gamma_mae' in checkpoint:
            print(f"   - Pre-trained γ MAE: {checkpoint['gamma_mae']:.6f}")
            print(f"   - Pre-trained β MAE: {checkpoint['beta_mae']:.6f}")
        print()
    
    def generate_finetuning_dataset(
        self, 
        problem_type='maxcut',
        graph_type='erdos_renyi',
        n_nodes_range=(10, 30),
        n_train_samples=500,
        n_val_samples=100,
        edge_prob_range=(0.4, 0.7),
        save_dir='Dataset/finetune',
        seed=42
    ):
        """
        Generate a specialized dataset for fine-tuning
        
        Args:
            problem_type: Type of problem ('maxcut', 'vertex_cover', etc.)
            graph_type: Type of graphs to generate
            n_nodes_range: Range of nodes in graphs
            n_train_samples: Number of training samples
            n_val_samples: Number of validation samples
            edge_prob_range: Range of edge probabilities
            save_dir: Directory to save datasets
            seed: Random seed
        
        Returns:
            train_path, val_path: Paths to generated datasets
        """
        print(f"🔄 Generating fine-tuning dataset...")
        print(f"   - Problem: {problem_type}")
        print(f"   - Graph type: {graph_type}")
        print(f"   - Node range: {n_nodes_range}")
        print(f"   - Training samples: {n_train_samples}")
        print(f"   - Validation samples: {n_val_samples}\n")
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize generator with specific configuration
        generator = QAOADataGeneratorImproved(seed=seed)
        
        # Generate training dataset
        train_path = f"{save_dir}/{problem_type}_{graph_type}_train.json"
        print("Generating training dataset...")
        train_dataset = generator.generate_dataset(
            n_samples=n_train_samples,
            n_nodes_range=n_nodes_range,
            graph_type=graph_type,
            p=self.p_layers,
            save_path=train_path
        )
        
        # Generate validation dataset with different seed
        val_path = f"{save_dir}/{problem_type}_{graph_type}_val.json"
        print("\nGenerating validation dataset...")
        generator.seed = seed + 10000
        val_dataset = generator.generate_dataset(
            n_samples=n_val_samples,
            n_nodes_range=n_nodes_range,
            graph_type=graph_type,
            p=self.p_layers,
            save_path=val_path
        )
        
        # Display statistics
        print("\n📊 Dataset Statistics:")
        train_stats = generator.dataset_statistics(train_dataset)
        print("Training:")
        for key, value in train_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        val_stats = generator.dataset_statistics(val_dataset)
        print("\nValidation:")
        for key, value in val_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print()
        return train_path, val_path
    
    def finetune(
        self,
        train_path,
        val_path,
        batch_size=16,
        learning_rate=0.0001,  # Lower LR for fine-tuning
        num_epochs=100,
        patience=30,
        save_path='finetuned_qaoa_model.pt',
        freeze_early_layers=False
    ):
        """
        Fine-tune the model on specific dataset
        
        Args:
            train_path: Path to training dataset
            val_path: Path to validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate (should be lower than pre-training)
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save fine-tuned model
            freeze_early_layers: If True, freeze first GNN layers (only train head)
        
        Returns:
            best_val_loss: Best validation loss achieved
        """
        print(f"🎯 Starting fine-tuning...")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Max epochs: {num_epochs}")
        print(f"   - Patience: {patience}")
        print(f"   - Freeze early layers: {freeze_early_layers}\n")
        
        # Load datasets
        train_loader_obj = DatasetLoader(train_path)
        val_loader_obj = DatasetLoader(val_path)
        
        train_loader = train_loader_obj.get_dataloader(batch_size=batch_size, shuffle=True)
        val_loader = val_loader_obj.get_dataloader(batch_size=batch_size, shuffle=False)
        
        # Optionally freeze early layers
        if freeze_early_layers:
            print("❄️  Freezing early GNN layers...")
            for i, conv in enumerate(self.model.convs[:-1]):  # Freeze all but last layer
                for param in conv.parameters():
                    param.requires_grad = False
            print(f"   - Frozen {len(self.model.convs) - 1} layers\n")
        
        # Setup optimizer with lower learning rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_gamma_mae = float('inf')
        best_beta_mae = float('inf')
        
        start_time = time.time()
        
        print("Starting training loop...\n")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'γ MAE':<12} {'β MAE':<12} {'Status':<15}")
        print("-" * 80)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = train_epoch(self.model, train_loader, optimizer, criterion, self.device)
            
            # Evaluate
            val_loss, val_preds, val_targets = evaluate(self.model, val_loader, criterion, self.device)
            
            # Compute parameter-specific errors
            gamma_mae, beta_mae = compute_parameter_errors(val_preds, val_targets, self.p_layers)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Check for improvement
            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_gamma_mae = gamma_mae
                best_beta_mae = beta_mae
                patience_counter = 0
                status = "✅ Best!"
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'p_layers': self.p_layers,
                    'model_type': self.model_type,
                    'gamma_mae': gamma_mae,
                    'beta_mae': beta_mae,
                    'finetuned': True,
                    'base_model': 'best_qaoa_gat_model.pt'
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    status = "⏹️  Early stop"
                
            # Print progress
            if epoch % 5 == 0 or status:
                print(f"{epoch:<6} {train_loss:<12.6f} {val_loss:<12.6f} {gamma_mae:<12.6f} {beta_mae:<12.6f} {status:<15}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        
        # Final report
        print("\n" + "=" * 80)
        print("🎉 Fine-tuning completed!")
        print("=" * 80)
        print(f"⏱️  Training time: {training_time/60:.2f} minutes")
        print(f"📊 Best validation loss: {best_val_loss:.6f}")
        print(f"🎯 Best γ MAE: {best_gamma_mae:.6f}")
        print(f"🎯 Best β MAE: {best_beta_mae:.6f}")
        print(f"💾 Fine-tuned model saved: {save_path}")
        print("=" * 80)
        
        return best_val_loss


def main():
    """
    Main function with CLI arguments
    """
    parser = argparse.ArgumentParser(description='Fine-tune QAOA GNN model')
    
    # Model parameters
    parser.add_argument('--pretrained-model', type=str, default='best_qaoa_gat_model.pt',
                        help='Path to pre-trained model')
    
    # Dataset parameters
    parser.add_argument('--problem-type', type=str, default='maxcut',
                        help='Problem type (maxcut, vertex_cover, etc.)')
    parser.add_argument('--graph-type', type=str, default='erdos_renyi',
                        choices=['erdos_renyi', 'regular', 'weighted'],
                        help='Type of graphs to generate')
    parser.add_argument('--n-nodes-min', type=int, default=10,
                        help='Minimum number of nodes')
    parser.add_argument('--n-nodes-max', type=int, default=30,
                        help='Maximum number of nodes')
    parser.add_argument('--n-train-samples', type=int, default=500,
                        help='Number of training samples')
    parser.add_argument('--n-val-samples', type=int, default=100,
                        help='Number of validation samples')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate (lower than pre-training)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--freeze-early-layers', action='store_true',
                        help='Freeze early GNN layers (only train head)')
    
    # Output parameters
    parser.add_argument('--save-path', type=str, default='finetuned_qaoa_model.pt',
                        help='Path to save fine-tuned model')
    parser.add_argument('--dataset-dir', type=str, default='Dataset/finetune',
                        help='Directory to save/load datasets')
    
    # Dataset generation control
    parser.add_argument('--use-existing-dataset', action='store_true',
                        help='Use existing dataset instead of generating new one')
    parser.add_argument('--train-dataset', type=str, default=None,
                        help='Path to existing training dataset')
    parser.add_argument('--val-dataset', type=str, default=None,
                        help='Path to existing validation dataset')
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    finetuner = FineTuner(pretrained_model_path=args.pretrained_model)
    
    # Get datasets
    if args.use_existing_dataset and args.train_dataset and args.val_dataset:
        print(f"📂 Using existing datasets:")
        print(f"   - Training: {args.train_dataset}")
        print(f"   - Validation: {args.val_dataset}\n")
        train_path = args.train_dataset
        val_path = args.val_dataset
    else:
        # Generate new dataset
        train_path, val_path = finetuner.generate_finetuning_dataset(
            problem_type=args.problem_type,
            graph_type=args.graph_type,
            n_nodes_range=(args.n_nodes_min, args.n_nodes_max),
            n_train_samples=args.n_train_samples,
            n_val_samples=args.n_val_samples,
            save_dir=args.dataset_dir
        )
    
    # Fine-tune model
    finetuner.finetune(
        train_path=train_path,
        val_path=val_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        save_path=args.save_path,
        freeze_early_layers=args.freeze_early_layers
    )


if __name__ == "__main__":
    # Example usage in script mode
    print("="*80)
    print("🚀 QAOA Model Fine-Tuning Script")
    print("="*80)
    print()
    
    main()
