"""
Script de Test Complet pour le Prototype v1

Teste:
1. Chaque composant individuellement
2. Le modèle complet (binaire uniquement)
3. La backpropagation
4. La Symmetric BCE Loss (symétrie des solutions)
5. La métrique de similarité
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

    print("OK Encoder\n")
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

    print("OK Problem Embedding\n")
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

    print("OK Transformer\n")
    return True


def test_classifier_binary():
    """Test du Classifier binaire"""
    print("=" * 60)
    print("TEST 4: Classifier (Binaire + Sigmoid)")
    print("=" * 60)

    batch_size = 2
    n_nodes = 6
    hidden_dim = 256

    x = torch.randn(batch_size, n_nodes, hidden_dim)
    classifier = Classifier(hidden_dim=256)

    output = classifier(x)
    print(f"Logits: {output['logits'].shape}")
    print(f"Probs: {output['probs'].shape}")
    print(f"Predictions: {output['predictions'].shape}")
    print(f"Exemple probs: {[f'{p:.2f}' for p in output['probs'][0].tolist()]}")
    print(f"Exemple preds: {output['predictions'][0].tolist()}")

    # Vérifier les shapes (binaire = pas de dimension de classes)
    assert output['logits'].shape == (batch_size, n_nodes)
    assert output['probs'].shape == (batch_size, n_nodes)
    assert output['predictions'].shape == (batch_size, n_nodes)

    # Vérifier que les probs sont entre 0 et 1
    assert output['probs'].min() >= 0 and output['probs'].max() <= 1

    print("OK Classifier Binaire\n")
    return True


def test_symmetric_loss():
    """Test de la Symmetric BCE Loss"""
    print("=" * 60)
    print("TEST 5: Symmetric BCE Loss")
    print("=" * 60)

    classifier = Classifier(hidden_dim=256)

    # Logits fixes pour les tests
    logits = torch.tensor([[2.0, -2.0, 2.0, -2.0, 2.0, -2.0]])

    # Target et son inverse
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    target_inv = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])

    loss_normal = classifier.compute_loss(logits, target)
    loss_inverse = classifier.compute_loss(logits, target_inv)

    print(f"Target:         {target[0].tolist()}")
    print(f"Target inversé: {target_inv[0].tolist()}")
    print(f"Loss (normal):  {loss_normal.item():.4f}")
    print(f"Loss (inversé): {loss_inverse.item():.4f}")
    print(f"Différence:     {abs(loss_normal.item() - loss_inverse.item()):.6f}")

    # Les deux loss doivent être identiques
    assert abs(loss_normal.item() - loss_inverse.item()) < 1e-5, \
        f"Les loss devraient être égales: {loss_normal.item():.6f} vs {loss_inverse.item():.6f}"

    print("OK Symmetric Loss (les deux loss sont identiques)\n")
    return True


def test_similarity_metric():
    """Test de la métrique de similarité"""
    print("=" * 60)
    print("TEST 6: Métrique de Similarité")
    print("=" * 60)

    classifier = Classifier(hidden_dim=256)

    # Cas 1: identique
    pred1 = torch.tensor([[0, 1, 0, 1, 0, 1]])
    target1 = torch.tensor([[0, 1, 0, 1, 0, 1]])
    sim1 = classifier.compute_similarity(pred1, target1)
    print(f"Identique:     {pred1[0].tolist()} vs {target1[0].tolist()} → {sim1:.0%}")

    # Cas 2: inversé (symétrie)
    pred2 = torch.tensor([[1, 0, 1, 0, 1, 0]])
    target2 = torch.tensor([[0, 1, 0, 1, 0, 1]])
    sim2 = classifier.compute_similarity(pred2, target2)
    print(f"Inversé:       {pred2[0].tolist()} vs {target2[0].tolist()} → {sim2:.0%}")

    # Cas 3: partiellement correct
    pred3 = torch.tensor([[0, 1, 0, 0, 0, 1]])
    target3 = torch.tensor([[0, 1, 0, 1, 0, 1]])
    sim3 = classifier.compute_similarity(pred3, target3)
    print(f"Partiel (1 err):{pred3[0].tolist()} vs {target3[0].tolist()} → {sim3:.0%}")

    # Cas 4: tout faux
    pred4 = torch.tensor([[0, 0, 0, 0, 0, 0]])
    target4 = torch.tensor([[0, 1, 0, 1, 0, 1]])
    sim4 = classifier.compute_similarity(pred4, target4)
    print(f"Moitié:        {pred4[0].tolist()} vs {target4[0].tolist()} → {sim4:.0%}")

    assert sim1 == 1.0, f"Identique devrait être 100%, got {sim1:.0%}"
    assert sim2 == 1.0, f"Inversé devrait être 100%, got {sim2:.0%}"

    print("OK Similarité\n")
    return True


def test_full_model():
    """Test du modèle complet"""
    print("=" * 60)
    print("TEST 7: Modèle Complet (QuantumGraphModel)")
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

    # Test MaxCut (problem_id=0)
    print(f"\n--- MaxCut (problem_id=0) ---")
    output = model(x, edge_index, problem_id=0)
    print(f"Probs: {output['probs'].shape}")
    print(f"Predictions: {output['predictions'].tolist()}")

    # Test Vertex Cover (problem_id=1)
    print(f"\n--- Vertex Cover (problem_id=1) ---")
    output = model(x, edge_index, problem_id=1)
    print(f"Predictions: {output['predictions'].tolist()}")

    # Test Independent Set (problem_id=2)
    print(f"\n--- Independent Set (problem_id=2) ---")
    output = model(x, edge_index, problem_id=2)
    print(f"Predictions: {output['predictions'].tolist()}")

    assert output['probs'].shape == (1, 6)
    assert output['predictions'].shape == (1, 6)

    print("OK Modèle Complet\n")
    return True


def test_backpropagation():
    """Test de la backpropagation avec Symmetric BCE"""
    print("=" * 60)
    print("TEST 8: Backpropagation (Symmetric BCE)")
    print("=" * 60)

    x, edge_index = create_test_graph(n_nodes=6, n_features=7)

    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256
    )

    # Forward
    output = model(x, edge_index, problem_id=0)

    # Loss
    targets = torch.tensor([[1, 0, 1, 0, 1, 0]])
    loss = model.compute_loss(output['logits'], targets)
    print(f"Loss: {loss.item():.4f}")

    # Similarité
    sim = model.compute_similarity(output['predictions'], targets)
    print(f"Similarité: {sim:.0%}")

    # Backward
    loss.backward()

    # Vérifier gradients
    components = {
        'GNN Encoder': model.encoder,
        'Problem Embedding': model.problem_embedding,
        'Transformer': model.transformer,
        'Classifier': model.classifier
    }

    print("\nGradients par composant:")
    all_ok = True
    for name, component in components.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in component.parameters() if p.requires_grad)
        status = "OK" if has_grad else "FAIL"
        if not has_grad:
            all_ok = False
        print(f"  [{status}] {name}")

    assert all_ok, "Tous les composants doivent avoir des gradients"

    print("OK Backpropagation\n")
    return True


def test_training_step():
    """Simule une étape d'entraînement avec Symmetric BCE"""
    print("=" * 60)
    print("TEST 9: Training Step (Symmetric BCE)")
    print("=" * 60)

    x, edge_index = create_test_graph(n_nodes=6, n_features=7)

    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    targets = torch.tensor([[1, 0, 1, 0, 1, 0]])

    print("Simulation de 5 steps:")

    for step in range(5):
        optimizer.zero_grad()

        output, loss, similarity = model.forward_with_loss(
            x, edge_index, problem_id=0, targets=targets
        )

        loss.backward()
        optimizer.step()

        print(f"  Step {step+1}: Loss = {loss.item():.4f}, Similarité = {similarity:.0%}")

    print("OK Training Step\n")
    return True


def test_different_graph_sizes():
    """Test avec différentes tailles de graphes"""
    print("=" * 60)
    print("TEST 10: Différentes Tailles de Graphes")
    print("=" * 60)

    model = QuantumGraphModel(
        node_input_dim=7,
        embedding_dim=128,
        hidden_dim=256
    )

    for n_nodes in [4, 8, 16, 32]:
        x, edge_index = create_test_graph(n_nodes=n_nodes, n_features=7)

        output = model(x, edge_index, problem_id=0)

        print(f"  {n_nodes} nœuds: probs = {output['probs'].shape}, preds = {output['predictions'].shape}")
        assert output['predictions'].shape == (1, n_nodes)

    print("OK Différentes Tailles\n")
    return True


def run_all_tests():
    """Lance tous les tests"""
    print("\n" + "=" * 60)
    print("   TESTS PROTOTYPE v1 - Binaire + Symmetric BCE Loss")
    print("=" * 60 + "\n")

    tests = [
        ("Encoder", test_encoder),
        ("Problem Embedding", test_problem_embedding),
        ("Transformer", test_transformer),
        ("Classifier Binaire", test_classifier_binary),
        ("Symmetric BCE Loss", test_symmetric_loss),
        ("Similarité", test_similarity_metric),
        ("Modèle Complet", test_full_model),
        ("Backpropagation", test_backpropagation),
        ("Training Step", test_training_step),
        ("Différentes Tailles", test_different_graph_sizes),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"ERREUR dans {name}: {e}\n")
            results.append((name, False))

    # Résumé
    print("=" * 60)
    print("                    RÉSUMÉ")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passés")

    if passed == total:
        print("\n  TOUS LES TESTS SONT PASSES !")
    else:
        print("\n  Certains tests ont échoué.")

    print("=" * 60 + "\n")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
