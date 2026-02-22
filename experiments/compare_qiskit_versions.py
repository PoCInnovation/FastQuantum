import json
import numpy as np
import matplotlib.pyplot as plt
import os

def get_stats(path, name):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    if len(data) > 0 and 'problem' in data[0]:
        data = [s for s in data if s['problem'] == 'MAXCUT']
    
    if not data:
        print(f"No MAXCUT data in {name}")
        return None

    gammas = [s['gamma'][0] if 'gamma' in s else s['optimal_gamma'][0] for s in data]
    betas = [s['beta'][0] if 'beta' in s else s['optimal_beta'][0] for s in data]
    ratios = [s['ratio'] if 'ratio' in s else s['approximation_ratio'] for s in data]
    n_nodes = [s['n_nodes'] for s in data]
    
    print(f"\n--- {name} ({len(data)} samples) ---")
    print(f"  Nodes: Avg={np.mean(n_nodes):.1f}")
    print(f"  Ratio: Avg={np.mean(ratios):.4f}, Std={np.std(ratios):.4f}")
    print(f"  Gamma: Avg={np.mean(gammas):.4f}, Std={np.std(gammas):.4f}")
    print(f"  Beta : Avg={np.mean(betas):.4f}, Std={np.std(betas):.4f}")
    
    return {
        'name': name,
        'gammas': gammas,
        'betas': betas,
        'ratios': ratios
    }

def main():
    old_path = "Dataset/train_maxcut.json"
    new_path = "Dataset/train_maxcut_sota.json"
    
    old_stats = get_stats(old_path, "Old Qiskit RWPE")
    new_stats = get_stats(new_path, "New SOTA Qiskit")
    
    if not old_stats or not new_stats:
        return

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(old_stats['gammas'], bins=30, alpha=0.5, label=old_stats['name'], density=True)
    plt.hist(new_stats['gammas'], bins=30, alpha=0.5, label=new_stats['name'], density=True)
    plt.title("Gamma Distribution")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.hist(old_stats['betas'], bins=30, alpha=0.5, label=old_stats['name'], density=True)
    plt.hist(new_stats['betas'], bins=30, alpha=0.5, label=new_stats['name'], density=True)
    plt.title("Beta Distribution")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.hist(old_stats['ratios'], bins=30, alpha=0.5, label=old_stats['name'], density=True)
    plt.hist(new_stats['ratios'], bins=30, alpha=0.5, label=new_stats['name'], density=True)
    plt.title("Approximation Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig('compare_qiskit_versions.png')
    print("\n✅ Comparison plot saved to 'compare_qiskit_versions.png'")

if __name__ == "__main__":
    main()