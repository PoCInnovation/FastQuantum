import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from GnnmodelGat import QAOAPredictorGAT
import os
from tqdm import tqdm

# --- Classes Utilitaires (Copie allégée de fine_tuner.py) ---
class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def load_from_checkpoint(self, checkpoint):
        if 'feature_mean' in checkpoint:
            self.mean = checkpoint['feature_mean']
            self.std = checkpoint['feature_std']
            return True
        return False

    def transform(self, x):
        if self.mean is not None:
            return (x - self.mean) / self.std
        return x

def load_dataset_from_pt(pt_path):
    print(f"📂 Chargement du dataset de test: {pt_path}...")
    raw_data = torch.load(pt_path)
    data_list = []
    for sample in raw_data:
        x = torch.tensor(sample['node_features'], dtype=torch.float)
        adj = np.array(sample['adjacency_matrix'])
        edge_index = torch.tensor(np.array(np.where(adj > 0)), dtype=torch.long)
        y = torch.tensor(sample['optimal_gamma'] + sample['optimal_beta'], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

def load_model(path, device):
    print(f"🛠️  Chargement: {path}")
    checkpoint = torch.load(path, map_location=device)
    p_layers = checkpoint.get('p_layers', 1)
    
    model = QAOAPredictorGAT(
        input_dim=7,
        hidden_dim=64,
        num_layers=3,
        p_layers=p_layers,
        attention_heads=8,
        dropout=0.0 # Pas de dropout en inférence
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Gestion du Scaler si présent dans le checkpoint
    scaler = FeatureScaler()
    has_scaler = scaler.load_from_checkpoint(checkpoint)
    
    return model, scaler, p_layers, has_scaler

def compare(original_path, finetuned_path, test_data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Chargement Data
    dataset = load_dataset_from_pt(test_data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # Batch 1 pour analyse fine
    
    # 2. Chargement Modèles
    model_orig, scaler_orig, p_orig, _ = load_model(original_path, device)
    model_new, scaler_new, p_new, has_scaler_new = load_model(finetuned_path, device)
    
    assert p_orig == p_new, "Les modèles n'ont pas la même profondeur QAOA (p-layers) !"
    p_layers = p_orig

    # 3. Inférence
    errors_orig = []
    errors_new = []
    
    print("\n⚔️  DUEL DES MODÈLES EN COURS...")
    
    with torch.no_grad():
        for data in tqdm(loader, unit="graph"):
            target = data.y.to(device).view(1, -1)
            
            # --- Modèle Original ---
            # On suppose qu'il prend les données brutes (ou gère sa propre affaire)
            # Si le modèle original n'avait pas de scaler sauvegardé, on passe data.x brut
            data_orig = data.clone().to(device)
            pred_orig = model_orig(data_orig)
            err_orig = torch.abs(pred_orig - target).mean().item()
            errors_orig.append(err_orig)
            
            # --- Modèle Finetuned ---
            # Lui, il a besoin de SES stats de normalisation
            data_new = data.clone().to(device)
            if has_scaler_new:
                data_new.x = scaler_new.transform(data_new.x)
                
            pred_new = model_new(data_new)
            err_new = torch.abs(pred_new - target).mean().item()
            errors_new.append(err_new)

    # 4. Statistiques
    mean_err_orig = np.mean(errors_orig)
    mean_err_new = np.mean(errors_new)
    improvement = ((mean_err_orig - mean_err_new) / mean_err_orig) * 100
    
    print("\n" + "="*60)
    print(f"🏆 RÉSULTATS DU DUEL")
    print("="*60)
    print(f"Moyenne Erreur Absolue (MAE) - ORIGINAL  : {mean_err_orig:.6f}")
    print(f"Moyenne Erreur Absolue (MAE) - FINETUNED : {mean_err_new:.6f}")
    print("-" * 60)
    
    if improvement > 0:
        print(f"🚀 GAIN DE PERFORMANCE : +{improvement:.2f}%")
        print("✅ Le modèle finetuned est SUPÉRIEUR.")
    else:
        print(f"❌ PERTE DE PERFORMANCE : {improvement:.2f}%")
        print("⚠️ Le modèle original reste meilleur (vérifiez l'overfitting).")

    # 5. Visualisation
    try:
        plt.figure(figsize=(12, 5))
        
        # Histogramme
        plt.subplot(1, 2, 1)
        plt.hist(errors_orig, bins=50, alpha=0.5, label='Original', color='gray')
        plt.hist(errors_new, bins=50, alpha=0.7, label='Finetuned', color='blue')
        plt.title("Distribution des Erreurs (MAE)")
        plt.xlabel("Erreur Absolue")
        plt.ylabel("Nombre de Graphes")
        plt.legend()
        
        # Scatter Plot (Comparaison directe)
        plt.subplot(1, 2, 2)
        plt.scatter(errors_orig, errors_new, alpha=0.5, c='purple', s=10)
        
        # Ligne y=x
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        global_min = min(x_min, y_min)
        global_max = max(x_max, y_max)
        
        lims = [global_min, global_max]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Pas de changement")
        plt.title("Comparaison Directe (Points sous la ligne = Gain)")
        plt.xlabel("Erreur Original")
        plt.ylabel("Erreur Finetuned")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('comparison_results.png')
        print(f"\n📊 Graphique sauvegardé sous : comparison_results.png")
        
    except Exception as e:
        print(f"Erreur graphique: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare two QAOA models")
    parser.add_argument('original', type=str, help="Path to original model")
    parser.add_argument('finetuned', type=str, help="Path to fine-tuned model")
    parser.add_argument('--data', type=str, default="Dataset/finetune/maxcut_val_massive.pt", help="Validation dataset path")
    
    args = parser.parse_args()
    
    # Fallback logic for data
    if not os.path.exists(args.data):
        fallback = "Dataset/finetune/maxcut_val.pt"
        if os.path.exists(fallback):
            print(f"Dataset {args.data} not found, using {fallback}")
            args.data = fallback
        else:
            print(f"Error: No validation dataset found at {args.data} or {fallback}")
            exit(1)
            
    if not os.path.exists(args.original):
        print(f"Error: Model {args.original} not found.")
        exit(1)
        
    if not os.path.exists(args.finetuned):
        print(f"Error: Model {args.finetuned} not found.")
        exit(1)
    
    compare(args.original, args.finetuned, args.data)
