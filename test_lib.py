import networkx as nx
import torch
import sys

# Add local path to test before installing
sys.path.insert(0, ".")

import fastquantum as fq

def test_inference():
    print("🧪 Starting FastQuantum Inference Test")
    
    # 1. Create a dummy graph
    print("-> Creating a random MaxCut graph (15 nodes)...")
    G = nx.erdos_renyi_graph(15, 0.4)
    
    # 2. Check if a model exists
    model_path = "best_qaoa_gat_model.pt"
    import os
    if not os.path.exists(model_path):
        print(f"⚠️  No model found at {model_path}.")
        print("Please train a model first or provide a valid checkpoint.")
        return
        
    # 3. Initialize Predictor
    print(f"-> Loading FastQuantumPredictor from {model_path}...")
    try:
        predictor = fq.FastQuantumPredictor(model_checkpoint=model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
        
    # 4. Predict
    print("-> Running prediction...")
    try:
        gamma, beta = predictor.predict(G, problem="MAXCUT")
        print("\n✅ Prediction successful!")
        print(f"Optimal Gamma: {gamma}")
        print(f"Optimal Beta:  {beta}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    test_inference()
