import torch
import torch.nn as nn
from GnnmodelGat import QAOAPredictorGAT, DatasetLoader, evaluate, compute_accuracy_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model_path = "best_qaoa_gat_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    p_layers = checkpoint['p_layers']

    print(f"Loading GAT model (p={p_layers})...")
    print(f"Trained at epoch {checkpoint['epoch']}")
    print(f"Best validation loss during training: {checkpoint['val_loss']:.6f}\n")

    # Create model
    model = QAOAPredictorGAT(
        input_dim=7,
        hidden_dim=64,
        num_layers=3,
        p_layers=p_layers,
        attention_heads=8,
        dropout=0.3
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load datasets
    val_path = "Dataset/qaoa_val_dataset.json"
    train_path = "Dataset/qaoa_train_dataset.json"

    print("Loading validation dataset...")
    val_loader_obj = DatasetLoader(val_path)
    val_loader = val_loader_obj.get_dataloader(batch_size=16, shuffle=False)

    print("Loading training dataset...")
    train_loader_obj = DatasetLoader(train_path)
    train_loader = train_loader_obj.get_dataloader(batch_size=16, shuffle=False)

    criterion = nn.MSELoss()

    # Evaluate on validation set
    print(f"\n{'='*70}")
    print("📈 VALIDATION SET EVALUATION")
    print(f"{'='*70}\n")

    val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

    print(f"Number of samples: {len(val_preds)}")
    print(f"Validation Loss: {val_loss:.6f}\n")

    # Compute metrics with different tolerances
    tolerances = [0.01, 0.02, 0.05, 0.1]
    print("🎯 Accuracy at different tolerances:")
    print(f"{'Tolerance':<12} {'γ Accuracy':<15} {'β Accuracy':<15} {'Overall':<15}")
    print("-" * 60)

    for tol in tolerances:
        metrics_tol = compute_accuracy_metrics(val_preds, val_targets, p_layers, tolerance=tol)
        print(f"  ±{tol:<10.2f} {metrics_tol['accuracy']['gamma']:<14.2f}% "
              f"{metrics_tol['accuracy']['beta']:<14.2f}% "
              f"{metrics_tol['accuracy']['overall']:<14.2f}%")

    # Detailed metrics for tolerance = 0.05
    metrics = compute_accuracy_metrics(val_preds, val_targets, p_layers, tolerance=0.05)

    print(f"\n📊 Detailed Metrics (tolerance = ±0.05):\n")
    print(f"  MAE:")
    print(f"    γ (gamma):  {metrics['mae']['gamma']:.6f}")
    print(f"    β (beta):   {metrics['mae']['beta']:.6f}")
    print(f"    Overall:    {metrics['mae']['overall']:.6f}")

    print(f"\n  RMSE:")
    print(f"    γ (gamma):  {metrics['rmse']['gamma']:.6f}")
    print(f"    β (beta):   {metrics['rmse']['beta']:.6f}")
    print(f"    Overall:    {metrics['rmse']['overall']:.6f}")

    print(f"\n  MAPE (Mean Absolute Percentage Error):")
    print(f"    γ (gamma):  {metrics['mape']['gamma']:.2f}%")
    print(f"    β (beta):   {metrics['mape']['beta']:.2f}%")

    print(f"\n  R² Score:")
    print(f"    γ (gamma):  {metrics['r2']['gamma']:.6f}")
    print(f"    β (beta):   {metrics['r2']['beta']:.6f}")
    print(f"    Overall:    {metrics['r2']['overall']:.6f}")

    print(f"\n  Max Error:")
    print(f"    γ (gamma):  {metrics['max_error']['gamma']:.6f}")
    print(f"    β (beta):   {metrics['max_error']['beta']:.6f}")

    print(f"\n{'='*70}")

    # Evaluate on training set
    print(f"\n{'='*70}")
    print("📈 TRAINING SET EVALUATION")
    print(f"{'='*70}\n")

    train_loss, train_preds, train_targets = evaluate(model, train_loader, criterion, device)
    train_metrics = compute_accuracy_metrics(train_preds, train_targets, p_layers, tolerance=0.05)

    print(f"Number of samples: {len(train_preds)}")
    print(f"Train Loss:       {train_loss:.6f}")
    print(f"γ MAE:            {train_metrics['mae']['gamma']:.6f}")
    print(f"β MAE:            {train_metrics['mae']['beta']:.6f}")
    print(f"Overall R²:       {train_metrics['r2']['overall']:.6f}")
    print(f"Accuracy (±0.05): {train_metrics['accuracy']['overall']:.2f}%")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
