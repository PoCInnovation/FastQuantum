# QAOA Parameter Fine-Tuner

This toolkit helps you refine Graph Neural Networks (GNN) to predict optimal QAOA parameters ($\gamma, \beta$) for MaxCut problems. It's designed to take a general pre-trained model and specialize it using a larger or more specific dataset.

## Quick Start

### 1. Generate Data (if needed)
You need training data first. This script uses Qiskit (or classical optimization) to find ground truth parameters for random graphs.

```bash
# Generates .pt files in Dataset/finetune/
python generate_maxcut_dataset_ai.py
```

### 2. Fine-Tune the Model
Take your existing model (`best_qaoa_gat_model.pt`) and train it on the new data.

```bash
# Default usage (uses standard paths)
python fine_tuner.py

# Custom usage
python fine_tuner.py --model best_qaoa_gat_model.pt --save my_custom_model.pt --epochs 200 --lr 0.0002
```

**Note:** Don't set `--lr` to 0 unless you want the model to learn nothing. A good default is `2e-4` (0.0002).

### 3. Compare Results
Check if the new model is actually better than the old one using the validation set.

```bash
python compare_models.py best_qaoa_gat_model.pt my_custom_model.pt
```

---

## How it works

### `fine_tuner.py`
This is the training script. It doesn't just run a loop; it implements a few best practices for GNN training:
*   **Feature Normalization:** Automatically scales node features (degree, centrality) using Z-score normalization so the optimizer converges faster.
*   **Mixed Precision:** Uses `torch.amp` to speed up training on NVIDIA GPUs and reduce memory usage.
*   **Gradient Clipping:** Prevents exploding gradients, which is common in GNNs.
*   **Robust Loss:** Uses `SmoothL1Loss` instead of MSE to be less sensitive to outliers.

**Arguments:**
*   `--model`: Input model path.
*   `--train` / `--val`: Paths to datasets.
*   `--save`: Output path.
*   `--lr`: Learning rate (default: 2e-4).
*   `--epochs`: Max epochs (stops early if no improvement).

### `compare_models.py`
A simple benchmarking tool. It loads both models and runs them on the *same* validation dataset (one that neither model saw during training).
*   Calculates Mean Absolute Error (MAE) for both.
*   Handles normalization correctly (applies scaler to the fine-tuned model, but not the original if it didn't have one).
*   Generates `comparison_results.png` to visualize the improvement.
