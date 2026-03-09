# Historique des Benchmarks et Stratégies

Ce document synthétise les expérimentations menées sur la génération de données et l'entraînement des modèles pour FastQuantum.

## Feature Engineering

Le modèle repose sur 7 features heuristiques calculées par nœud, choisies pour leur corrélation avec la physique du problème QAOA (MaxCut) :

| Feature | Justification Physique | Cible Principale |
| :--- | :--- | :--- |
| **Normalized Degree** | Densité locale d'interactions (frustration locale). | **Beta** |
| **Clustering Coeff.** | Présence de cycles courts (triangles) rendant MaxCut difficile. | **Beta** |
| **Eigenvector Cent.** | Lié au spectre du graphe (valeurs propres), donc à l'opérateur Hamiltonien. | **Gamma** |
| **PageRank** | Simule une diffusion, proche de la dynamique quantique. | **Gamma** |
| **Betweenness Cent.** | Importance du nœud dans la connectivité globale. | Globale |
| **Closeness Cent.** | Vitesse de propagation de l'information. | Globale |
| **Core Number** | Décomposition en k-cores (structure hiérarchique). | Globale |

*Note : L'inclusion de l'Eigenvector Centrality fournit une baseline forte car $\gamma$ est directement lié aux valeurs propres du graphe.*

## Chronologie des Expérimentations

### 1. Full Random (Baseline check)
*   **Setup :** Graphes aléatoires, cibles ($\gamma, \beta$) bruit uniformément.
*   **Résultat :** Les modèles complexes (avec RWPE) ont performé moins bien que la baseline simple (MLP sur features).
*   **Conclusion :** Le modèle essayait d'apprendre du bruit (overfitting). Approche abandonnée pour l'apprentissage de structure.

### 2. Physics Proxy (Validation)
*   **Setup :** Utilisation de `generate_physics_proxy.py`. Cibles générées via des règles spectrales déterministes :
    *   $\gamma \propto 1 / \lambda_{max}$ (Inverse de la valeur propre max)
    *   $\beta \propto \text{Clustering} + \text{Degré}$
*   **Résultat :** Le modèle avec **RWPE (Random Walk Positional Encodings)** surpasse la baseline.
    *   Loss RWPE : `0.003056`
    *   Loss Baseline : `0.003179`
*   **Analyse :** Le RWPE encode la géométrie de diffusion du graphe mieux que de simples scalaires, permettant de capturer des subtilités topologiques nécessaires à la prédiction d'énergie.
*   **Statut :** Configuration retenue pour la production.

### 3. Architecture V2 "Wide & Deep" (Prototype)
*   **Setup :** Ajout d'embeddings d'identité (mémoire explicite par instance de graphe).
*   **Problème :** Manque de généralisation aux graphes instanciés hors du training set (problème de "l'annuaire").
*   **Solution prévue :** Remplacement par une architecture RAG (Retrieval-Augmented Generation) basée sur la similarité de graphes, plutôt que sur des IDs exacts.

## Configuration Technique Actuelle

Pour la phase "Deep Research", la stack retenue est :

*   **Architecture :** GAT (Graph Attention Network) avec edge-conditioning.
*   **Input Features :** Les 7 heuristiques + RWPE (Random Walk Positional Encoding).
*   **Training Data :** Dataset généré via `generate_v1_dataset.py` (Mélange de topologies + Résolution exacte brute-force pour Ground Truth).

Cette approche a été validée pour sa capacité à généraliser sur des graphes de tailles variables (100-1000 nœuds).
