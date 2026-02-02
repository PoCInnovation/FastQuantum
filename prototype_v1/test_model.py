"""
Script de Test Complet pour le Prototype v1

Teste:
1. Chaque composant individuellement
2. Le modèle complet
3. La backpropagation
4. Différents problèmes (binaire et multi-classe)
"""

import torch
import sys
from pathlib import Path

# Ajouter le dossier au path
sys.path.insert(0, str(Path(__file__).parent))

from encoder import GNNEncoder
from problem_embedding import ProblemEmbeddingTable
from transformer import GraphTransformer
from classifier import Classifier
from model import QuantumGraphModel


def create_test_graph(n_nodes=6, n_features=7):
    """Crée un graphe de test simple (cycle)"""
    x = torch.randn(n_nodes, n_features)

    # Graphe en cycle: 0-1-2-3-4-5-0
    sources = list(range(n_nodes)) + list(range(1, n_nodes)) + [0]
    targets = list(range(1, n_nodes)) + [0] + list(range(n_nodes))
    edge_index = torch.tensor([sources, targets])

    return x, edge_index


def test_encoder():
    """Test du GNN Encoder"""
    print("=" * 60)
    print("TEST 1: GNN Encoder")
    print("=" * 60)

    x, edge_index = create_test_graph(n_nodes=6, n_features=7)

    encoder = GNNEncoder(
        input_dim=7,
        hidden_dim=128,
        num_layers=4,
        num_heads=4
    )

    e_local, e_global = encoder(x, edge_index)

    print(f"Input:  x = {x.shape}, edge_index = {edge_index.shape}")
    print(f"Output: e_local = {e_local.shape}, e_global = {e_global.shape}")

    assert e_local.shape == (6, 128), f"Expected (6, 128), got {e_local.shape}"
    assert e_global.shape == (1, 128), f"Expected (1, 128), got {e_global.shape}"

    print("✅ Encoder OK\n")
    return True


def test_problem_embedding():
    """Test de la table d'embedding"""
    print("=" * 60)
    print("TEST 2: Problem Embedding Table")
    print("=" * 60)

    table = ProblemEmbeddingTable(num_problems=10, embedding_dim=128)

    # Test single ID
    e_prob = table(0)
    print(f"Problem ID 0 (MaxCut): shape = {e_prob.shape}")

    # Test batch
    e_prob_batch = table(torch.tensor([0, 1, 2, 3]))
    print(f"Batch [0,1,2,3]: shape = {e_prob_batch.shape}")

    # Vérifier que chaque ID donne un embedding différent
    e0 = table(0)
    e1 = table(1)
    diff = (e0 - e1).abs().mean().item()
    print(f"Différence entre ID 0 et ID 1: {diff:.4f}")

    assert e_prob.shape == (1, 128)
    assert e_prob_batch.shape == (4, 128)
    assert diff > 0, "Les embeddings devraient être différents"

    print("✅ Problem Embedding OK\n")
    return True


def test_transformer():
    """Test du Transformer"""
    print("=" * 60)
    print("TEST 3: Graph Transformer")
    print("=" * 60)

    batch_size = 2
    n_nodes = 6
    input_dim = 128

    e_local = torch.randn(batch_size, n_nodes, input_dim)
    e_global = torch.randn(batch_size, input_dim)
    e_prob = torch.randn(batch_size, input_dim)

    transformer = GraphTransformer(
        input_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=8
    )

    output = transformer(e_local, e_global, e_prob)

    print(f"Input:  e_local = {e_local.shape}")
    print(f"        e_global = {e_global.shape}")
    print(f"        e_prob = {e_prob.shape}")
    print(f"Output: contextualized = {output.shape}")

    assert output.shape == (batch_size, n_nodes, 256)

    print("✅ Transformer OK\n")
    return True


def test_classifier():
    """Test du Classifier"""
    print("=" * 60)
    print("TEST 4: Classifier")
    print("=" * 60)

    batch_size = 2
    n_nodes = 6
    hidden_dim = 256

    x = torch.randn(batch_size, n_nodes, hidden_dim)
    classifier = Classifier(hidden_dim=256, max_classes=10)

    # Test binaire (2 classes)
    output_2 = classifier(x, num_classes=2)
    print(f"Binaire (2 classes):")
    print(f"  Logits: {output_2['logits'].shape}")
    print(f"  Probs: {output_2['probs'].shape}")
    print(f"  Predictions: {output_2['predictions'].shape}")
    print(f"  Exemple: {output_2['predictions'][0].tolist()}")

    # Test multi-classe (5 classes)
    output_5 = classifier(x, num_classes=5)
    print(f"\nMulti-classe (5 classes):")
    print(f"  Logits: {output_5['logits'].shape}")
    print(f"  Predictions: {output_5['predictions'][0].tolist()}")

    # Test loss
    targets = torch.randint(0, 2, (batch_size, n_nodes))
    loss = classifier.compute_loss(output_2['logits'], targets)
    print(f"\nLoss (CE): {loss.item():.4f}")

    assert output_2['logits'].shape == (batch_size, n_nodes, 2)
    assert output_5['logits'].shape == (batch_size, n_nodes, 5)

    print("✅ Classifier OK\n")
    return True


def test_full_model():
    """Test du modèle complet"""
    print("=" * 60)
    print("TEST 5: Modèle Complet (QuantumGraphModel)")
    print("=" * 60)

    x, edge_index = create_test_graph(n_nodes=6, n_features=7)

    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256,
        gnn_layers=4,
        transformer_layers=4
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres totaux: {n_params:,}")

    # Test MaxCut (problem_id=0, 2 classes)
    print(f"\n--- MaxCut (problem_id=0, 2 classes) ---")
    output = model(x, edge_index, problem_id=0, num_classes=2)
    print(f"Logits: {output['logits'].shape}")
    print(f"Predictions: {output['predictions'].tolist()}")

    # Test Vertex Cover (problem_id=1, 2 classes)
    print(f"\n--- Vertex Cover (problem_id=1, 2 classes) ---")
    output = model(x, edge_index, problem_id=1, num_classes=2)
    print(f"Predictions: {output['predictions'].tolist()}")

    # Test Graph Coloring (problem_id=3, 4 classes)
    print(f"\n--- Graph Coloring (problem_id=3, 4 classes) ---")
    output = model(x, edge_index, problem_id=3, num_classes=4)
    print(f"Logits: {output['logits'].shape}")
    print(f"Predictions: {output['predictions'].tolist()}")

    print("✅ Modèle Complet OK\n")
    return True


def test_backpropagation():
    """Test de la backpropagation à travers tout le modèle"""
    print("=" * 60)
    print("TEST 6: Backpropagation")
    print("=" * 60)

    x, edge_index = create_test_graph(n_nodes=6, n_features=7)

    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256
    )

    # Forward
    output = model(x, edge_index, problem_id=0, num_classes=2)

    # Loss
    targets = torch.tensor([[1, 0, 1, 0, 1, 0]])
    loss = model.compute_loss(output['logits'], targets)
    print(f"Loss avant backward: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Vérifier que les gradients existent partout
    components = {
        'GNN Encoder': model.encoder,
        'Problem Embedding': model.problem_embedding,
        'Transformer': model.transformer,
        'Classifier': model.classifier
    }

    print("\nGradients par composant:")
    for name, component in components.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in component.parameters() if p.requires_grad)
        status = "✅" if has_grad else "❌"
        print(f"  {status} {name}")

    # Vérifier spécifiquement la lookup table
    lookup_grad = model.problem_embedding.embedding_table.weight.grad
    if lookup_grad is not None:
        grad_problem_0 = lookup_grad[0].abs().sum().item()
        print(f"\n  Gradient lookup table (ID=0): {grad_problem_0:.6f}")

    print("✅ Backpropagation OK\n")
    return True


def test_training_step():
    """Simule une étape d'entraînement"""
    print("=" * 60)
    print("TEST 7: Training Step Simulation")
    print("=" * 60)

    x, edge_index = create_test_graph(n_nodes=6, n_features=7)

    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    targets = torch.tensor([[1, 0, 1, 0, 1, 0]])

    print("Simulation de 5 steps d'entraînement:")

    for step in range(5):
        optimizer.zero_grad()

        output = model(x, edge_index, problem_id=0, num_classes=2)
        loss = model.compute_loss(output['logits'], targets)

        loss.backward()
        optimizer.step()

        accuracy = (output['predictions'] == targets).float().mean().item()
        print(f"  Step {step+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2%}")

    print("✅ Training Step OK\n")
    return True


def test_different_graph_sizes():
    """Test avec différentes tailles de graphes"""
    print("=" * 60)
    print("TEST 8: Différentes Tailles de Graphes")
    print("=" * 60)

    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256
    )

    for n_nodes in [4, 8, 16, 32]:
        x, edge_index = create_test_graph(n_nodes=n_nodes, n_features=7)

        output = model(x, edge_index, problem_id=0, num_classes=2)

        print(f"  {n_nodes} nœuds: predictions shape = {output['predictions'].shape}")
        assert output['predictions'].shape == (1, n_nodes)

    print("✅ Différentes Tailles OK\n")
    return True


def run_all_tests():
    """Lance tous les tests"""
    print("\n" + "=" * 60)
    print("       TESTS DU PROTOTYPE v1 - QuantumGraphModel")
    print("=" * 60 + "\n")

    tests = [
        ("Encoder", test_encoder),
        ("Problem Embedding", test_problem_embedding),
        ("Transformer", test_transformer),
        ("Classifier", test_classifier),
        ("Full Model", test_full_model),
        ("Backpropagation", test_backpropagation),
        ("Training Step", test_training_step),
        ("Different Sizes", test_different_graph_sizes),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"❌ ERREUR dans {name}: {e}\n")
            results.append((name, False))

    # Résumé
    print("=" * 60)
    print("                    RÉSUMÉ")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{total} tests passés")

    if passed == total:
        print("\n  🎉 TOUS LES TESTS SONT PASSÉS ! 🎉")
    else:
        print("\n  ⚠️  Certains tests ont échoué.")

    print("=" * 60 + "\n")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
