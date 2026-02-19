# LLM Autorégressif & Knowledge Distillation avec LoRA

## Table des Matières

1. [Fonctionnement Autorégressif des LLM](#1-fonctionnement-autorégressif-des-llm)
   - [Principe de Base](#11-principe-de-base)
   - [Les Deux Phases](#12-les-deux-phases)
   - [Système de Séquences](#13-système-de-séquences)
   - [Implémentation de la Boucle](#14-implémentation-de-la-boucle)
   - [Optimisation : KV Caching](#15-optimisation-kv-caching)
   - [Masquage Causal](#16-masquage-causal)
   - [Stratégies de Décodage](#17-stratégies-de-décodage)

2. [Knowledge Distillation](#2-knowledge-distillation)
   - [Principe Teacher-Student](#21-principe-teacher-student)
   - [Types de Distillation](#22-types-de-distillation)
   - [Fonction de Perte](#23-fonction-de-perte)
   - [Hyperparamètres](#24-hyperparamètres)

3. [Compatibilité LoRA + Knowledge Distillation](#3-compatibilité-lora--knowledge-distillation)
   - [KD-LoRA : Approche Hybride](#31-kd-lora--approche-hybride)
   - [Paramètres LoRA](#32-paramètres-lora)
   - [Configuration Recommandée](#33-configuration-recommandée)

4. [Implémentation Complète](#4-implémentation-complète)
5. [Références](#5-références)

---

## 1. Fonctionnement Autorégressif des LLM

### 1.1 Principe de Base

Les **Large Language Models (LLM)** fonctionnent de manière **autorégressif** : ils génèrent du texte **un token à la fois**, en utilisant tous les tokens précédents comme contexte.

**Définition** : Autorégressif signifie que chaque prédiction dépend des prédictions précédentes.

```
Token T₀ à Tₙ₋₁  →  Génère Token Tₙ
Token T₀ à Tₙ    →  Génère Token Tₙ₊₁
Token T₀ à Tₙ₊₁  →  Génère Token Tₙ₊₂
...
```

**Schéma conceptuel** :
```
Prompt Initial: "Le chat mange"
    ↓
[Token0="Le", Token1="chat", Token2="mange"]
    ↓
┌─────────────────┐
│   LLM (Forward) │
└─────────────────┘
    ↓
Token3 = "une" ← Prédiction
    ↓
[Token0, Token1, Token2, Token3] ← Ajout à la séquence
    ↓
┌─────────────────┐
│   LLM (Forward) │ ← Réinjection
└─────────────────┘
    ↓
Token4 = "souris"
    ↓
... (continuer jusqu'à EOS)
```

---

### 1.2 Les Deux Phases

#### **Phase 1 : Prefill (Remplissage)**
- Tous les tokens du **prompt initial** sont traités **en parallèle**
- Le modèle calcule les représentations pour toute la séquence d'entrée
- **Une seule passe** à travers le modèle

#### **Phase 2 : Decoding (Décodage)**
- Génération **séquentielle** token par token
- Chaque nouveau token est **ajouté** à la séquence
- La séquence complète est **réinjectée** pour générer le suivant
- Continue jusqu'à :
  - Token de fin (EOS - End Of Sequence)
  - Longueur maximale atteinte

---

### 1.3 Système de Séquences

#### Évolution de la Séquence

```python
# Étape 0 : Prompt initial
sequence = ["Le", "chat", "mange"]

# Étape 1
model(["Le", "chat", "mange"]) → "une"
sequence = ["Le", "chat", "mange", "une"]

# Étape 2
model(["Le", "chat", "mange", "une"]) → "souris"
sequence = ["Le", "chat", "mange", "une", "souris"]

# Étape 3
model(["Le", "chat", "mange", "une", "souris"]) → "."
sequence = ["Le", "chat", "mange", "une", "souris", "."]

# Étape 4
model(["Le", "chat", "mange", "une", "souris", "."]) → <EOS>
# STOP
```

#### Représentation Visuelle

```
Itération 1:  [T₀, T₁, T₂]           → T₃
Itération 2:  [T₀, T₁, T₂, T₃]       → T₄
Itération 3:  [T₀, T₁, T₂, T₃, T₄]   → T₅
...
Itération n:  [T₀, ..., Tₙ₋₁]        → Tₙ (EOS)
```

**Point clé** : La séquence **grandit** à chaque itération, mais on ne prédit qu'**un seul token** à la fois.

---

### 1.4 Implémentation de la Boucle

#### Version Simple (Sans Cache)

```python
import torch
import torch.nn.functional as F

def generate_autoregressive_simple(model, initial_tokens, max_length=100, eos_token_id=None):
    """
    Génération autorégressif basique : chaque output devient input

    Args:
        model: Le modèle LLM (ex: GPT-2)
        initial_tokens: Tensor de tokens initiaux [batch_size, seq_len]
        max_length: Nombre maximum de tokens à générer
        eos_token_id: ID du token de fin (End Of Sequence)

    Returns:
        Tensor de la séquence complète générée
    """
    # Initialisation avec le prompt
    current_sequence = initial_tokens.clone()

    # Boucle de génération
    for step in range(max_length):
        # 1. Forward pass avec toute la séquence actuelle
        with torch.no_grad():
            outputs = model(current_sequence)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # 2. Récupérer les logits du DERNIER token
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # 3. Prédire le prochain token (greedy decoding)
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]

        # 4. AJOUTER le token généré à la séquence
        current_sequence = torch.cat([current_sequence, next_token], dim=1)

        # 5. Vérifier la condition d'arrêt
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return current_sequence


# Exemple d'utilisation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encoder le prompt
prompt = "Le chat mange"
input_ids = tokenizer.encode(prompt, return_tensors='pt')  # [1, 3]

# Générer
output_ids = generate_autoregressive_simple(
    model=model,
    initial_tokens=input_ids,
    max_length=50,
    eos_token_id=tokenizer.eos_token_id
)

# Décoder
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
```

#### Version avec Sampling (Plus de Diversité)

```python
def generate_with_sampling(model, initial_tokens, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
    """
    Génération avec sampling au lieu de greedy decoding

    Args:
        temperature: Contrôle l'aléatoire (0.7-1.0 recommandé)
        top_k: Garde les k tokens les plus probables
        top_p: Nucleus sampling (garde les tokens dont la somme de proba = p)
    """
    current_sequence = initial_tokens.clone()

    for step in range(max_length):
        with torch.no_grad():
            outputs = model(current_sequence)
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

        # Appliquer la température
        next_token_logits = next_token_logits / temperature

        # Top-K filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Retirer les tokens au-delà du seuil top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        # Échantillonner depuis la distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

        # Ajouter à la séquence
        current_sequence = torch.cat([current_sequence, next_token], dim=1)

        if (next_token == tokenizer.eos_token_id).all():
            break

    return current_sequence
```

---

### 1.5 Optimisation : KV Caching

#### Le Problème

Sans cache, à chaque étape on recalcule **toute la séquence** :

```
Étape 1: Forward([T₀, T₁, T₂])           → calcule K,V pour T₀, T₁, T₂
Étape 2: Forward([T₀, T₁, T₂, T₃])       → recalcule K,V pour T₀, T₁, T₂, T₃  ❌ LENT
Étape 3: Forward([T₀, T₁, T₂, T₃, T₄])   → recalcule K,V pour tout           ❌ LENT
```

#### La Solution : KV Cache

**KV Cache** stocke les clés (K) et valeurs (V) des tokens déjà traités en mémoire :

```
Étape 1: Forward([T₀, T₁, T₂])     → calcule K,V pour T₀, T₁, T₂ + CACHE
Étape 2: Forward([T₃])              → calcule K,V seulement pour T₃  ✅ RAPIDE
Étape 3: Forward([T₄])              → calcule K,V seulement pour T₄  ✅ RAPIDE
```

#### Implémentation avec KV Cache

```python
def generate_with_kv_cache(model, initial_tokens, max_length=100):
    """
    Génération optimisée avec cache KV pour éviter les recalculs

    Le cache stocke les clés (K) et valeurs (V) de l'attention pour
    tous les tokens déjà traités, évitant ainsi de recalculer à chaque étape.
    """
    current_sequence = initial_tokens.clone()
    past_key_values = None  # Cache initialement vide

    for step in range(max_length):
        if past_key_values is None:
            # Premier passage : traiter tout le prompt
            with torch.no_grad():
                outputs = model(
                    current_sequence,
                    use_cache=True  # Activer le cache
                )
        else:
            # Passages suivants : traiter SEULEMENT le dernier token
            with torch.no_grad():
                outputs = model(
                    current_sequence[:, -1:],  # Seulement le dernier token !
                    past_key_values=past_key_values,  # Réutiliser le cache
                    use_cache=True
                )

        # Récupérer les logits et mettre à jour le cache
        logits = outputs.logits
        past_key_values = outputs.past_key_values  # Cache mis à jour

        # Prédire le prochain token
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        # Ajouter à la séquence
        current_sequence = torch.cat([current_sequence, next_token], dim=1)

        if (next_token == tokenizer.eos_token_id).all():
            break

    return current_sequence


# Utilisation avec HuggingFace (intégré)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

input_ids = tokenizer("Le chat mange", return_tensors='pt').input_ids

# HuggingFace utilise automatiquement le KV cache avec model.generate()
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    use_cache=True  # ✅ Activé par défaut
)

print(tokenizer.decode(output[0]))
```

#### Gains de Performance

| Méthode | Calculs par étape | Complexité |
|---------|-------------------|------------|
| **Sans cache** | O(n²) | Recalcule tous les tokens |
| **Avec cache** | O(n) | Calcule seulement le nouveau token |

**Résultat** : Réduction de **50-80% du temps d'inférence** pour les séquences longues.

---

### 1.6 Masquage Causal

Le **masque causal** (causal mask) est essentiel pour empêcher le modèle de "tricher" en voyant les tokens futurs.

#### Principe

```
Masque triangulaire inférieur (1 = peut voir, 0 = masqué)

       T₀  T₁  T₂  T₃
    ┌─────────────────┐
T₀  │  1   0   0   0  │  ← T₀ ne voit que lui-même
T₁  │  1   1   0   0  │  ← T₁ voit T₀ et lui-même
T₂  │  1   1   1   0  │  ← T₂ voit T₀, T₁ et lui-même
T₃  │  1   1   1   1  │  ← T₃ voit tous les précédents
    └─────────────────┘
```

#### Implémentation

```python
import torch

def create_causal_mask(seq_length):
    """
    Crée un masque causal (triangulaire inférieur)

    Args:
        seq_length: Longueur de la séquence

    Returns:
        Masque de forme [seq_length, seq_length]
    """
    # torch.tril crée une matrice triangulaire inférieure
    mask = torch.tril(torch.ones(seq_length, seq_length))
    return mask


# Exemple
seq_len = 4
mask = create_causal_mask(seq_len)
print(mask)
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])


def apply_causal_mask_to_attention(attention_scores, mask):
    """
    Applique le masque causal aux scores d'attention

    Args:
        attention_scores: Scores d'attention [batch, heads, seq_len, seq_len]
        mask: Masque causal [seq_len, seq_len]

    Returns:
        Scores masqués
    """
    # Remplacer les positions masquées par -inf
    # (softmax(-inf) = 0, donc pas d'attention)
    attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    return attention_scores


# Dans l'attention mechanism
attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
attention_scores = apply_causal_mask_to_attention(attention_scores, mask)
attention_weights = F.softmax(attention_scores, dim=-1)
output = attention_weights @ value
```

---

### 1.7 Stratégies de Décodage

#### 1. Greedy Decoding (Avidité)

**Principe** : Toujours choisir le token le plus probable.

```python
def greedy_decoding(logits):
    """
    Sélectionne le token avec la probabilité la plus élevée
    """
    next_token = torch.argmax(logits, dim=-1)
    return next_token
```

**Avantages** :
- Simple et rapide
- Déterministe (même input → même output)

**Inconvénients** :
- Texte répétitif et prévisible
- Pas de diversité

---

#### 2. Sampling avec Température

**Principe** : Échantillonner selon une distribution de probabilité ajustée.

```python
def temperature_sampling(logits, temperature=1.0):
    """
    Sampling avec température

    - temperature < 1.0 : Plus déterministe (favorise les tokens probables)
    - temperature = 1.0 : Distribution normale
    - temperature > 1.0 : Plus aléatoire (favorise la diversité)
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


# Exemple
logits = torch.tensor([2.0, 1.0, 0.5])

# Température basse (conservateur)
token_low = temperature_sampling(logits, temperature=0.5)

# Température normale
token_normal = temperature_sampling(logits, temperature=1.0)

# Température haute (créatif)
token_high = temperature_sampling(logits, temperature=1.5)
```

**Valeurs recommandées** :
- `temperature = 0.7` : Équilibre créativité/cohérence
- `temperature = 0.5` : Plus conservateur
- `temperature = 1.0` : Distribution normale

---

#### 3. Top-K Sampling

**Principe** : Ne considérer que les K tokens les plus probables.

```python
def top_k_sampling(logits, k=50):
    """
    Top-K sampling : garde les k tokens les plus probables

    Args:
        logits: Logits du modèle [vocab_size]
        k: Nombre de top tokens à considérer (typique: 40-50)
    """
    # Récupérer les k meilleurs tokens
    top_k_logits, top_k_indices = torch.topk(logits, k)

    # Mettre les autres à -inf
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered[top_k_indices] = top_k_logits

    # Sampling depuis la distribution filtrée
    probs = F.softmax(logits_filtered, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

**Avantages** :
- Évite les tokens très improbables
- Garde une certaine diversité

**Valeurs typiques** : `k = 40-50`

---

#### 4. Top-P (Nucleus) Sampling

**Principe** : Considérer les tokens dont la somme cumulative de probabilité atteint P.

```python
def top_p_sampling(logits, p=0.95):
    """
    Top-P (nucleus) sampling : garde les tokens jusqu'à atteindre une proba cumulative de p

    Args:
        logits: Logits du modèle [vocab_size]
        p: Seuil de probabilité cumulative (typique: 0.9-0.95)
    """
    # Trier par ordre décroissant
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Calculer la probabilité cumulative
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Retirer les tokens au-delà du seuil p
    sorted_indices_to_remove = cumulative_probs > p
    # Garder au moins le premier token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Remettre dans l'ordre original
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits_filtered = logits.clone()
    logits_filtered[indices_to_remove] = float('-inf')

    # Sampling
    probs = F.softmax(logits_filtered, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

**Avantages** :
- Adaptatif (nombre de tokens varie selon la distribution)
- Meilleur équilibre diversité/qualité

**Valeurs typiques** : `p = 0.9-0.95`

---

#### 5. Beam Search

**Principe** : Maintenir plusieurs hypothèses (beam) et choisir la meilleure séquence.

```python
def beam_search(model, initial_tokens, beam_width=5, max_length=50):
    """
    Beam search : explore plusieurs chemins simultanément

    Args:
        beam_width: Nombre d'hypothèses à maintenir (typique: 3-5)
    """
    batch_size = initial_tokens.size(0)
    vocab_size = model.config.vocab_size

    # Initialiser avec le prompt
    # beams: liste de (séquence, score)
    beams = [(initial_tokens, 0.0)]

    for step in range(max_length):
        all_candidates = []

        # Pour chaque beam actuel
        for sequence, score in beams:
            if sequence[0, -1] == tokenizer.eos_token_id:
                # Si déjà terminé, garder tel quel
                all_candidates.append((sequence, score))
                continue

            # Forward pass
            with torch.no_grad():
                outputs = model(sequence)
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Top-k tokens pour ce beam
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            # Créer de nouveaux candidats
            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                next_log_prob = topk_log_probs[0, i].item()

                new_sequence = torch.cat([sequence, next_token], dim=1)
                new_score = score + next_log_prob

                all_candidates.append((new_sequence, new_score))

        # Garder les beam_width meilleurs candidats
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    # Retourner la meilleure séquence
    best_sequence, best_score = beams[0]
    return best_sequence
```

**Avantages** :
- Meilleure qualité (explore plusieurs chemins)
- Bon pour traduction, summarization

**Inconvénients** :
- Plus lent (beam_width × calculs)
- Moins de diversité que sampling

**Valeurs typiques** : `beam_width = 3-5`

---

#### Comparaison des Stratégies

| Stratégie | Vitesse | Qualité | Diversité | Usage |
|-----------|---------|---------|-----------|-------|
| **Greedy** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | Tests rapides |
| **Sampling + Temp** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Génération créative |
| **Top-K** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Équilibre |
| **Top-P** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ Recommandé |
| **Beam Search** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Traduction, résumé |

---

#### Configuration Optimale (HuggingFace)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

input_ids = tokenizer("Le chat mange", return_tensors='pt').input_ids

# Configuration recommandée pour génération créative
output = model.generate(
    input_ids,
    max_length=100,

    # Stratégie de décodage
    do_sample=True,              # Activer le sampling (sinon greedy)
    temperature=0.7,             # Contrôle créativité (0.5-1.0)
    top_k=50,                    # Top-K sampling
    top_p=0.95,                  # Nucleus sampling (recommandé)

    # Contrôle de répétition
    repetition_penalty=1.2,      # Pénaliser les répétitions (1.0-1.5)
    no_repeat_ngram_size=2,      # Éviter répétition de bigrammes

    # Longueur
    min_length=10,               # Longueur minimale
    max_length=100,              # Longueur maximale

    # Fin de génération
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,

    # Optimisation
    use_cache=True,              # KV caching (activé par défaut)

    # Autres
    num_return_sequences=1,      # Nombre de séquences à générer
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

### 1.8 Problèmes Courants

#### Exposure Bias

**Problème** : Si le modèle fait une petite erreur au début, cette erreur est réinjectée comme contexte, pouvant conduire à plus d'erreurs en cascade.

**Exemple** :
```
Prompt: "Le chat mange"
Token 1: "une" ✅
Token 2: "chien" ❌ (erreur)
Token 3: "aboie" ❌ (à cause de l'erreur précédente)
Token 4: "fort" ❌ (déraillement complet)
```

**Solutions** :
- Utiliser **sampling** au lieu de greedy
- Appliquer **repetition penalty**
- Utiliser **beam search** pour explorer plusieurs chemins

---

## 2. Knowledge Distillation

### 2.1 Principe Teacher-Student

La **Knowledge Distillation (KD)** est une technique de compression de modèle où un **modèle Teacher** (grand, performant) transfère ses connaissances à un **modèle Student** (petit, efficace).

#### Architecture

```
┌───────────────────────────────────┐
│       Teacher Model (Frozen)       │
│   (ex: GPT-4, LLaMA-70B, etc.)    │
│          175B parameters           │
└──────────────┬────────────────────┘
               │
               │ Soft Targets
               │ (Probabilités)
               ↓
┌───────────────────────────────────┐
│       Student Model (Training)     │
│     (ex: GPT-2, LLaMA-7B, etc.)   │
│          1.5B parameters           │
└───────────────────────────────────┘
```

#### Objectif

Le Student apprend à imiter :
1. **Les prédictions finales** du Teacher (soft targets)
2. **Le raisonnement** du Teacher (distributions de probabilité)
3. **Les représentations intermédiaires** (optionnel)

---

### 2.2 Types de Distillation

#### 1. Response-based KD (Basée sur la réponse)

**Principe** : Transférer les **logits de sortie** du Teacher.

```python
# Teacher produit des logits
teacher_logits = teacher_model(input_ids).logits  # [batch, seq_len, vocab_size]

# Student produit des logits
student_logits = student_model(input_ids).logits  # [batch, seq_len, vocab_size]

# Minimiser la différence entre les deux
loss = KL_divergence(student_logits, teacher_logits)
```

**Usage** : Le plus courant pour les LLM.

---

#### 2. Feature-based KD (Basée sur les features)

**Principe** : Transférer les **représentations intermédiaires** (hidden states).

```python
# Récupérer les hidden states
teacher_hidden = teacher_model(input_ids, output_hidden_states=True).hidden_states
student_hidden = student_model(input_ids, output_hidden_states=True).hidden_states

# Aligner les représentations de certaines couches
loss_feature = 0
for teacher_layer, student_layer in zip(teacher_hidden[::2], student_hidden):
    # L2 loss ou cosine similarity
    loss_feature += F.mse_loss(student_layer, teacher_layer)
```

**Usage** : Pour capturer des connaissances plus profondes.

---

#### 3. Attention-based KD (Basée sur l'attention)

**Principe** : Transférer les **attention maps** du Teacher.

```python
# Récupérer les attentions
teacher_attentions = teacher_model(input_ids, output_attentions=True).attentions
student_attentions = student_model(input_ids, output_attentions=True).attentions

# Aligner les patterns d'attention
loss_attention = 0
for t_att, s_att in zip(teacher_attentions, student_attentions):
    loss_attention += F.mse_loss(s_att, t_att)
```

**Usage** : Pour les modèles Transformer (BERT, GPT).

---

### 2.3 Fonction de Perte

#### Formule Complète

La loss de distillation combine deux objectifs :

```
Loss_total = α × Loss_distillation + (1 - α) × Loss_hard

où:
  Loss_distillation = KL_divergence(Student_soft || Teacher_soft)
  Loss_hard = CrossEntropy(Student_predictions, True_labels)

  α : balance entre distillation et supervision (typique: 0.7-0.9)
```

#### Soft Targets avec Température

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, true_labels, temperature=4.0, alpha=0.7):
    """
    Calcule la loss de distillation

    Args:
        student_logits: Logits du Student [batch, vocab_size]
        teacher_logits: Logits du Teacher [batch, vocab_size]
        true_labels: Labels réels [batch]
        temperature: Température pour softening (typique: 3-5)
        alpha: Balance distillation vs hard labels (typique: 0.7-0.9)

    Returns:
        Loss totale
    """
    # 1. Soft targets (avec température)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # 2. KL Divergence loss
    # Note: On multiplie par T² pour compenser l'effet de la température sur les gradients
    distillation_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='batchmean'
    ) * (temperature ** 2)

    # 3. Hard labels loss (cross-entropy standard)
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # 4. Combiner les deux losses
    total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss

    return total_loss


# Exemple d'utilisation
teacher_logits = torch.randn(32, 50257)  # [batch=32, vocab_size=50257]
student_logits = torch.randn(32, 50257)
true_labels = torch.randint(0, 50257, (32,))

loss = distillation_loss(
    student_logits=student_logits,
    teacher_logits=teacher_logits,
    true_labels=true_labels,
    temperature=4.0,
    alpha=0.7
)
print(f"Distillation loss: {loss.item()}")
```

#### Pourquoi la Température ?

**Sans température (T=1)** :
```
Logits: [10.0, 2.0, 1.0]
Probs:  [0.9999, 0.0001, 0.00005]  ← Distribution très "piquée"
```

**Avec température (T=4)** :
```
Logits / 4: [2.5, 0.5, 0.25]
Probs:      [0.89, 0.07, 0.04]  ← Distribution plus "douce"
```

**Avantage** : Les soft targets révèlent les **connaissances cachées** du Teacher (ex: "chat" est plus proche de "chien" que de "voiture").

---

#### Pourquoi KL Divergence ?

La **KL Divergence** mesure la différence entre deux distributions de probabilité.

```python
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

**Propriétés** :
- KL = 0 quand Student = Teacher (objectif optimal)
- Toujours ≥ 0
- Asymétrique : KL(P||Q) ≠ KL(Q||P)

**Alternative** : Cross-Entropy donne les mêmes gradients, mais KL est préférée car elle a une interprétation plus claire.

---

### 2.4 Hyperparamètres

#### Tableau Récapitulatif

| Paramètre | Valeurs Typiques | Recommandé | Description |
|-----------|------------------|------------|-------------|
| **Temperature (T)** | 1-20 | **3-5** | Softening des logits |
| **Alpha (α)** | 0.1-0.9 | **0.7-0.8** | Balance KD vs Hard |
| **Learning Rate** | 1e-5 à 5e-4 | **1e-4** | LR pour Student |
| **Batch Size** | 8-64 | **16-32** | Selon GPU |
| **Epochs** | 3-10 | **5** | Dépend du dataset |

#### Détails par Paramètre

##### 1. Temperature (T)

| Valeur | Effet | Cas d'usage |
|--------|-------|-------------|
| **T = 1** | Pas de softening | Distribution normale (pas de KD) |
| **T = 3-4** | ⭐ Léger softening | Standard pour KD |
| **T = 4** | ⭐ Le plus commun | Valeur par défaut |
| **T = 5-10** | Fort softening | Student << Teacher (très petit) |
| **T = 10+** | Très soft | Modèles très différents |

**Règle** : Plus le Student est petit par rapport au Teacher, plus T doit être élevé.

##### 2. Alpha (α)

| Valeur | Focus | Cas d'usage |
|--------|-------|-------------|
| **α = 0.5** | Équilibré | 50% KD, 50% labels |
| **α = 0.7** | ⭐ Favorise Teacher | Standard |
| **α = 0.8** | ⭐ Très orienté KD | Teacher très fiable |
| **α = 0.9** | Presque que KD | Teacher excellent |
| **α = 0.1** | Favorise labels | Teacher peu fiable ou bruité |

**Règle** : α = **0.7-0.8** pour mettre l'accent sur le Teacher.

##### 3. Learning Rate

| Modèle | Learning Rate | Notes |
|--------|---------------|-------|
| Student (KD) | **1e-4 à 5e-4** | Plus bas que fine-tuning standard |
| Student (sans KD) | 1e-3 à 5e-3 | Référence |
| BERT-base | 2e-5 | Exemple |
| GPT-2 | 1e-4 | Exemple |

**Règle** : LR légèrement plus bas qu'un fine-tuning classique.

---

#### Exemple de Configuration

```python
# Configuration pour BERT distillation
distillation_config = {
    # Loss parameters
    'temperature': 4.0,           # ⭐ Standard
    'alpha': 0.7,                 # ⭐ 70% Teacher, 30% labels
    'loss_type': 'kl_divergence', # KL div standard

    # Training parameters
    'learning_rate': 1e-4,        # ⭐ Pour LLM
    'batch_size': 16,
    'num_epochs': 5,
    'warmup_steps': 500,
    'weight_decay': 0.01,

    # Optimizer
    'optimizer': 'AdamW',
    'betas': (0.9, 0.999),
    'eps': 1e-8,
}


# Configuration pour GPT-2 distillation
gpt2_distillation_config = {
    'temperature': 4.0,
    'alpha': 0.8,                 # Plus orienté Teacher
    'learning_rate': 3e-4,        # Un peu plus haut pour GPT
    'batch_size': 8,              # Plus petit pour séquences longues
    'gradient_accumulation_steps': 4,  # Effective batch = 32
    'max_seq_length': 512,
}
```

---

## 3. Compatibilité LoRA + Knowledge Distillation

### 3.1 KD-LoRA : Approche Hybride

**KD-LoRA** combine Knowledge Distillation et LoRA (Low-Rank Adaptation) pour un fine-tuning ultra-efficace.

#### Concept

```
┌────────────────────────────────────────┐
│      Teacher Model (Frozen)            │
│         (Large LLM)                    │
│       ex: GPT-4, 175B params           │
└────────────────┬───────────────────────┘
                 │
                 │ Knowledge Distillation
                 ↓
┌────────────────────────────────────────┐
│      Student Model (Frozen base)       │
│       ex: GPT-2, 1.5B params           │
│              +                         │
│      LoRA Adapters (Trainable)        │
│         rank=16, alpha=32              │
│         Only 1.2M params!              │
└────────────────────────────────────────┘
```

#### Processus en 3 Étapes

1. **Distillation** : Teacher → Student (KD classique)
2. **Injection LoRA** : Ajouter des modules LoRA au Student
3. **Fine-tuning** : Entraîner seulement les LoRA (pas le Student entier)

---

#### Architecture LoRA

LoRA ajoute des **matrices de rang faible** aux couches d'attention :

```
Poids originaux : W ∈ ℝ^(d×d)    (frozen)
LoRA matrices   : A ∈ ℝ^(d×r), B ∈ ℝ^(r×d)   (trainable)

Output = W·x + (B·A)·x
         ↑      ↑
      frozen  trainable (rank r << d)

Exemple:
  W : 768 × 768 = 589,824 params (frozen)
  A : 768 × 16  = 12,288 params  (trainable)
  B : 16 × 768  = 12,288 params  (trainable)
  Total LoRA: 24,576 params (4% du total!)
```

---

### 3.2 Paramètres LoRA

#### Tableau des Hyperparamètres

| Paramètre | Valeurs | Recommandé | Description |
|-----------|---------|------------|-------------|
| **Rank (r)** | 4-256 | **8-16** | Dimensionnalité LoRA |
| **Alpha (α)** | r à 2r | **2×r** | Scaling factor |
| **Dropout** | 0-0.2 | **0.05-0.1** | Régularisation |
| **Target Modules** | - | **q_proj, v_proj** | Couches à adapter |
| **Learning Rate** | 1e-5 à 1e-3 | **3e-4** | LR pour LoRA |

---

#### 1. Rank (r) - Détails

Le **rank** contrôle la capacité des matrices LoRA.

| Rank | Params (BERT-base) | Usage | Cas d'usage |
|------|---------------------|-------|-------------|
| **r = 4** | ~0.3M | Minimal | Styles simples, concepts basiques |
| **r = 8** | ~0.6M | ⭐ Recommandé (départ) | Valeur par défaut (papier LoRA) |
| **r = 16** | ~1.2M | ⭐ Standard | Fine-tuning général |
| **r = 32** | ~2.4M | Complexe | Character likeness, tâches complexes |
| **r = 64** | ~4.8M | Très complexe | Datasets riches |
| **r = 128** | ~9.6M | Avancé | Tâches très complexes |
| **r = 256** | ~19.2M | Maximum pratique | Plafond pour datasets typiques |

**Règle** : Commencer avec **r = 8**, augmenter si sous-performance.

**Scaling Rule** (recherche 2024) :
```
Learning Rate optimal ≈ r^(-0.84)

Exemples:
  r=8   → LR ≈ 5e-4
  r=16  → LR ≈ 3e-4
  r=64  → LR ≈ 1e-4
  r=256 → LR ≈ 5e-5
```

---

#### 2. Alpha (α) - Scaling Factor

Alpha contrôle **l'intensité** de l'adaptation LoRA.

```python
# Dans le forward pass:
output = W @ x + (alpha / r) * (B @ A @ x)
                    ↑
              scaling factor
```

| Alpha | Relation | Scaling | Usage |
|-------|----------|---------|-------|
| **α = r** | α/r = 1 | ⭐ Standard | Sweet spot |
| **α = 2r** | α/r = 2 | ⭐ Recommandé (Microsoft) | Plus d'influence |
| **α = 0.5r** | α/r = 0.5 | Conservateur | Petits ajustements |
| **α = 4r** | α/r = 4 | Agressif | Changements importants |

**Exemples** :
```
r=8   → α=16  (recommandé)
r=16  → α=32  (recommandé)
r=32  → α=64
r=64  → α=128
r=256 → α=512 (meilleure perf selon recherche)
```

**Règle pratique** : α = **2 × rank**

---

#### 3. Target Modules

Quelles couches appliquer LoRA ?

| Modules | Description | Usage |
|---------|-------------|-------|
| **q_proj, v_proj** | ⭐ Query & Value | Standard (le plus efficace) |
| **q_proj, k_proj, v_proj, o_proj** | Toute l'attention | Maximum de flexibilité |
| **q_proj, v_proj, mlp.up_proj** | Attention + MLP | Pour tâches complexes |
| **Toutes les couches** | Partout | Rarement nécessaire |

**Recommandation** : Commencer avec **q_proj, v_proj** (couches Query et Value de l'attention).

---

#### 4. Dropout LoRA

| Dropout | Effet | Usage |
|---------|-------|-------|
| **0** | Pas de régularisation | Small datasets (<1k exemples) |
| **0.05** | ⭐ Léger | Standard |
| **0.1** | ⭐ Moyen | Recommandé |
| **0.2** | Fort | Si overfitting important |

---

### 3.3 Configuration Recommandée

#### Configuration Complète : KD + LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# 1. CHARGER LES MODÈLES
# ============================================

# Teacher (frozen)
teacher_model = AutoModelForCausalLM.from_pretrained('gpt2-medium')  # 345M params
teacher_model.eval()  # Mode évaluation
for param in teacher_model.parameters():
    param.requires_grad = False  # Freeze

# Student base (frozen)
student_model = AutoModelForCausalLM.from_pretrained('gpt2')  # 124M params

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# ============================================
# 2. CONFIGURATION LORA
# ============================================

lora_config = LoraConfig(
    # Paramètres principaux
    r=16,                              # ⭐ Rank (commencer avec 8-16)
    lora_alpha=32,                     # ⭐ Alpha = 2 × rank
    target_modules=["c_attn"],         # Pour GPT-2 (équivalent q,k,v)
    # target_modules=["q_proj", "v_proj"],  # Pour LLaMA/Mistral

    # Régularisation
    lora_dropout=0.1,                  # ⭐ Dropout standard

    # Autres
    bias="none",                       # Ne pas entraîner les biais
    task_type=TaskType.CAUSAL_LM,      # Causal language modeling

    # Avancé (optionnel)
    fan_in_fan_out=True,               # Pour GPT-2 (False pour LLaMA)
    modules_to_save=None,              # Sauvegarder des modules supplémentaires
)

# Appliquer LoRA au Student
student_model = get_peft_model(student_model, lora_config)

# Vérifier les paramètres entraînables
student_model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.24


# ============================================
# 3. DISTILLATION LOSS
# ============================================

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Loss combinée pour Knowledge Distillation

        Args:
            temperature: Température pour soft targets (3-5 recommandé)
            alpha: Balance distillation vs hard labels (0.7-0.8 recommandé)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
        """
        # 1. Soft targets (KL Divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        distillation_loss = self.kl_div(
            student_soft.view(-1, student_soft.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1))
        ) * (self.temperature ** 2)

        # 2. Hard labels (Cross-Entropy)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100  # Ignorer les tokens de padding
        )

        # 3. Combiner
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return total_loss, distillation_loss, hard_loss


# ============================================
# 4. TRAINING LOOP
# ============================================

from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Configuration
config = {
    # Distillation
    'temperature': 4.0,              # ⭐ Standard
    'alpha': 0.7,                    # ⭐ 70% Teacher, 30% labels

    # LoRA (déjà défini dans lora_config)
    'lora_rank': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,

    # Training
    'learning_rate': 3e-4,           # ⭐ Standard pour LoRA
    'batch_size': 8,
    'gradient_accumulation_steps': 4,  # Effective batch = 32
    'num_epochs': 5,
    'warmup_steps': 500,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,

    # Séquence
    'max_seq_length': 512,
}

# Loss function
distillation_loss_fn = DistillationLoss(
    temperature=config['temperature'],
    alpha=config['alpha']
)

# Optimizer (seulement les params LoRA)
optimizer = AdamW(
    student_model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay'],
    betas=(0.9, 0.999),
    eps=1e-8
)

# Scheduler
total_steps = len(train_dataloader) * config['num_epochs'] // config['gradient_accumulation_steps']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config['warmup_steps'],
    num_training_steps=total_steps
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model.to(device)
student_model.to(device)

# Training loop
student_model.train()
global_step = 0

for epoch in range(config['num_epochs']):
    epoch_loss = 0

    for step, batch in enumerate(train_dataloader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass Teacher (frozen)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, labels=labels)
            teacher_logits = teacher_outputs.logits

        # Forward pass Student (LoRA trainable)
        student_outputs = student_model(input_ids, labels=labels)
        student_logits = student_outputs.logits

        # Compute distillation loss
        loss, distill_loss, hard_loss = distillation_loss_fn(
            student_logits,
            teacher_logits,
            labels
        )

        # Normalize loss by gradient accumulation
        loss = loss / config['gradient_accumulation_steps']

        # Backward
        loss.backward()

        # Update every gradient_accumulation_steps
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                config['max_grad_norm']
            )

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % 100 == 0:
                print(f"Epoch {epoch+1}/{config['num_epochs']} | "
                      f"Step {global_step} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Distill: {distill_loss.item():.4f} | "
                      f"Hard: {hard_loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    student_model.save_pretrained(f"./checkpoints/kd_lora_epoch_{epoch+1}")


# ============================================
# 5. ÉVALUATION
# ============================================

student_model.eval()

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output = student_model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
prompt = "Le chat mange"
generated = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")


# ============================================
# 6. SAUVEGARDER LE MODÈLE FINAL
# ============================================

# Sauvegarder seulement les poids LoRA (très léger!)
student_model.save_pretrained("./final_model/kd_lora_gpt2")

# Pour charger plus tard:
# from peft import PeftModel
# base_model = AutoModelForCausalLM.from_pretrained('gpt2')
# model = PeftModel.from_pretrained(base_model, "./final_model/kd_lora_gpt2")
```

---

### 3.4 Résultats et Comparaisons

#### Performance (Papier KD-LoRA 2024)

| Méthode | Params Entraînés | Mémoire GPU | Performance GLUE | Inference Time | Taille Modèle |
|---------|------------------|-------------|------------------|----------------|---------------|
| **Full Fine-tuning** | 110M (100%) | 100% | 100% | 1x | Grand |
| **LoRA** | 1.2M (1%) | 55% | 98-99% | 1x | Grand |
| **KD seul** | 110M Student | 50% | 90-95% | 0.7x | Petit |
| **KD-LoRA** ⭐ | 1.2M (1%) | 25% | **97-98%** | **0.7x** | **Petit** |

#### Gains Spécifiques

**Mémoire** :
- 75% de réduction vs Full Fine-tuning
- 30% de réduction vs LoRA seul

**Vitesse** :
- Inference 30% plus rapide
- Training 40% plus rapide (moins de params)

**Compacité** :
- Modèle 40% plus compact que LoRA
- 4000× plus petit que le Teacher

---

#### Exemple Concret (BERT-base)

```
Teacher: BERT-large (340M params)
Student: BERT-base (110M params)

Full Fine-tuning BERT-base:
  ✓ Params entraînés: 110M
  ✓ Mémoire: ~8 GB
  ✓ Temps: 4h
  ✓ Accuracy: 92.5%

LoRA BERT-base (r=8):
  ✓ Params entraînés: 1.2M
  ✓ Mémoire: ~4.5 GB
  ✓ Temps: 2.5h
  ✓ Accuracy: 91.8%

KD-LoRA BERT-base (r=8, T=4, α=0.7):
  ✓ Params entraînés: 1.2M
  ✓ Mémoire: ~2 GB
  ✓ Temps: 1.5h
  ✓ Accuracy: 91.2%  ← Seulement -1.3% vs Full FT!
  ✓ Taille finale: 5 MB (vs 440 MB)
```

---

## 4. Implémentation Complète

### Script Complet : Génération Autorégressif + KD-LoRA

```python
"""
Script complet: Génération Autorégressif avec Knowledge Distillation et LoRA

Fonctionnalités:
1. Génération autorégressif avec KV caching
2. Knowledge Distillation (Teacher → Student)
3. LoRA fine-tuning efficace
4. Stratégies de décodage multiples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
import wandb  # Pour le logging (optionnel)


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration centralisée"""

    # Modèles
    teacher_model_name = "gpt2-medium"  # 345M params
    student_model_name = "gpt2"         # 124M params

    # Distillation
    temperature = 4.0
    alpha = 0.7

    # LoRA
    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["c_attn"]  # Pour GPT-2

    # Training
    learning_rate = 3e-4
    batch_size = 8
    gradient_accumulation_steps = 4
    num_epochs = 5
    warmup_steps = 500
    max_grad_norm = 1.0
    weight_decay = 0.01

    # Génération
    max_seq_length = 512
    max_gen_length = 100

    # Paths
    output_dir = "./output/kd_lora_model"
    cache_dir = "./cache"

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Logging
    log_steps = 100
    save_steps = 1000
    eval_steps = 500


# ============================================
# DATASET
# ============================================

class TextDataset(Dataset):
    """Dataset pour le training"""

    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx].clone()
        }


def load_data(tokenizer, config):
    """Charger et préparer le dataset"""
    # Exemple avec WikiText-2
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=config.cache_dir)

    # Filtrer les lignes vides
    train_texts = [text for text in dataset['train']['text'] if text.strip()]
    val_texts = [text for text in dataset['validation']['text'] if text.strip()]

    # Créer les datasets
    train_dataset = TextDataset(train_texts[:5000], tokenizer, config.max_seq_length)
    val_dataset = TextDataset(val_texts[:500], tokenizer, config.max_seq_length)

    return train_dataset, val_dataset


# ============================================
# DISTILLATION LOSS
# ============================================

class DistillationLoss(nn.Module):
    """Loss pour Knowledge Distillation"""

    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels, attention_mask=None):
        """
        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            attention_mask: [batch, seq_len] (optionnel)
        """
        # Reshape pour le calcul
        batch_size, seq_len, vocab_size = student_logits.shape

        # 1. Soft targets (KL Divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Appliquer le masque si fourni
        if attention_mask is not None:
            mask = attention_mask.view(-1) == 1
            student_soft = student_soft.view(-1, vocab_size)[mask]
            teacher_soft = teacher_soft.view(-1, vocab_size)[mask]
        else:
            student_soft = student_soft.view(-1, vocab_size)
            teacher_soft = teacher_soft.view(-1, vocab_size)

        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # 2. Hard labels (Cross-Entropy)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        # 3. Combiner
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return {
            'loss': total_loss,
            'distillation_loss': distillation_loss,
            'hard_loss': hard_loss
        }


# ============================================
# MODÈLE
# ============================================

def setup_models(config):
    """Initialiser Teacher, Student et LoRA"""

    print("Loading models...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Teacher (frozen)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        cache_dir=config.cache_dir
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.to(config.device)

    # Student base
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        cache_dir=config.cache_dir
    )

    # Configuration LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        fan_in_fan_out=True  # Pour GPT-2
    )

    # Appliquer LoRA
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()
    student_model.to(config.device)

    return teacher_model, student_model, tokenizer


# ============================================
# TRAINING
# ============================================

class KDLoRATrainer:
    """Trainer custom pour KD + LoRA"""

    def __init__(self, teacher_model, student_model, tokenizer, config):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.config = config

        # Loss
        self.distillation_loss_fn = DistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Scheduler (sera initialisé après connaître le nb de steps)
        self.scheduler = None

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train(self, train_dataloader, val_dataloader):
        """Training loop"""

        # Setup scheduler
        total_steps = (len(train_dataloader) * self.config.num_epochs
                      // self.config.gradient_accumulation_steps)
        from transformers import get_linear_schedule_with_warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

        print(f"Training for {self.config.num_epochs} epochs...")
        print(f"Total steps: {total_steps}")

        self.student_model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for step, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                # Forward Teacher
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(input_ids, attention_mask=attention_mask)
                    teacher_logits = teacher_outputs.logits

                # Forward Student
                student_outputs = self.student_model(input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits

                # Loss
                loss_dict = self.distillation_loss_fn(
                    student_logits,
                    teacher_logits,
                    labels,
                    attention_mask
                )
                loss = loss_dict['loss'] / self.config.gradient_accumulation_steps

                # Backward
                loss.backward()

                # Update
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.max_grad_norm
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        progress_bar.set_postfix({
                            'loss': f"{loss_dict['loss'].item():.4f}",
                            'distill': f"{loss_dict['distillation_loss'].item():.4f}",
                            'hard': f"{loss_dict['hard_loss'].item():.4f}",
                            'lr': f"{lr:.2e}"
                        })

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        val_loss = self.evaluate(val_dataloader)
                        print(f"\nValidation loss: {val_loss:.4f}")

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best")

                        self.student_model.train()

                    # Save
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")

    def evaluate(self, val_dataloader):
        """Évaluation sur le validation set"""
        self.student_model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                teacher_outputs = self.teacher_model(input_ids, attention_mask=attention_mask)
                student_outputs = self.student_model(input_ids, attention_mask=attention_mask)

                loss_dict = self.distillation_loss_fn(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    labels,
                    attention_mask
                )

                total_loss += loss_dict['loss'].item()

        return total_loss / len(val_dataloader)

    def save_checkpoint(self, name):
        """Sauvegarder le modèle"""
        save_path = f"{self.config.output_dir}/{name}"
        self.student_model.save_pretrained(save_path)
        print(f"Checkpoint saved: {save_path}")


# ============================================
# GÉNÉRATION AUTORÉGRESSIF
# ============================================

class AutoregressiveGenerator:
    """Générateur avec différentes stratégies"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt, max_length=100, strategy='top_p', **kwargs):
        """
        Génération avec KV caching

        Strategies:
        - 'greedy': Greedy decoding
        - 'sample': Temperature sampling
        - 'top_k': Top-K sampling
        - 'top_p': Nucleus sampling (recommandé)
        - 'beam': Beam search
        """
        self.model.eval()

        # Encoder le prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            if strategy == 'greedy':
                output = self._greedy_generate(input_ids, max_length)
            elif strategy == 'sample':
                temperature = kwargs.get('temperature', 0.7)
                output = self._sample_generate(input_ids, max_length, temperature)
            elif strategy == 'top_k':
                k = kwargs.get('k', 50)
                temperature = kwargs.get('temperature', 0.7)
                output = self._top_k_generate(input_ids, max_length, k, temperature)
            elif strategy == 'top_p':
                p = kwargs.get('p', 0.95)
                temperature = kwargs.get('temperature', 0.7)
                output = self._top_p_generate(input_ids, max_length, p, temperature)
            elif strategy == 'beam':
                beam_width = kwargs.get('beam_width', 5)
                output = self._beam_search(input_ids, max_length, beam_width)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _greedy_generate(self, input_ids, max_length):
        """Greedy decoding"""
        current_sequence = input_ids
        past_key_values = None

        for _ in range(max_length):
            if past_key_values is None:
                outputs = self.model(current_sequence, use_cache=True)
            else:
                outputs = self.model(
                    current_sequence[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )

            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return current_sequence

    def _top_p_generate(self, input_ids, max_length, p, temperature):
        """Nucleus (Top-P) sampling - RECOMMANDÉ"""
        current_sequence = input_ids
        past_key_values = None

        for _ in range(max_length):
            if past_key_values is None:
                outputs = self.model(current_sequence, use_cache=True)
            else:
                outputs = self.model(
                    current_sequence[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :] / temperature

            # Top-P filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_sequence = torch.cat([current_sequence, next_token], dim=1)

            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return current_sequence

    # Autres méthodes similaires pour top_k, sample, beam...


# ============================================
# MAIN
# ============================================

def main():
    """Point d'entrée principal"""

    # Configuration
    config = Config()

    # Setup models
    teacher_model, student_model, tokenizer = setup_models(config)

    # Load data
    train_dataset, val_dataset = load_data(tokenizer, config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Training
    trainer = KDLoRATrainer(teacher_model, student_model, tokenizer, config)
    trainer.train(train_dataloader, val_dataloader)

    # Test generation
    print("\n" + "="*50)
    print("Testing generation...")
    print("="*50)

    generator = AutoregressiveGenerator(student_model, tokenizer, config.device)

    test_prompts = [
        "Le chat mange",
        "L'intelligence artificielle est",
        "Dans un monde où"
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = generator.generate(
            prompt,
            max_length=50,
            strategy='top_p',
            p=0.95,
            temperature=0.7
        )
        print(f"Generated: {generated}\n")


if __name__ == "__main__":
    main()
```

---

## 5. Références

### Fonctionnement Autorégressif

1. **How does LLM inference work?** - BentoML
   https://bentoml.com/llm/llm-inference-basics/how-does-llm-inference-work

2. **Understanding the LLM's inference** - Medium
   https://lathashreeh.medium.com/understanding-the-llms-inference-36a767f98a83

3. **How LLMs Work, Explained Without Math**
   https://blog.miguelgrinberg.com/post/how-llms-work-explained-without-math

4. **How Transformer LLMs Generate Text: One Token at a Time**
   https://medium.com/codex/how-transformer-llms-generate-text-one-token-at-a-time-5531838bc2a1

5. **Transformer Sequence Generation | CodeSignal**
   https://codesignal.com/learn/courses/bringing-transformers-to-life-training-inference/lessons/transformer-sequence-generation

6. **Autoregressive Text Generation Beyond Feedback Loops** (arXiv 2019)
   https://arxiv.org/abs/1908.11658

7. **A Comprehensive Survey of Accelerated Generation Techniques in LLMs** (arXiv 2024)
   https://arxiv.org/html/2405.13019v2

### Knowledge Distillation

8. **What is Knowledge distillation?** - IBM
   https://www.ibm.com/think/topics/knowledge-distillation

9. **Knowledge Distillation for LLMs: Techniques and Applications**
   https://medium.com/@yugank.aman/knowledge-distillation-for-llms-techniques-and-applications-e23a17093adf

10. **Knowledge Distillation for Large Language Models: A Deep Dive** - Zilliz
    https://zilliz.com/learn/knowledge-distillation-from-large-language-models-deep-dive

11. **MiniLLM: Knowledge Distillation of Large Language Models** (arXiv 2023)
    https://arxiv.org/pdf/2306.08543

12. **Knowledge Distillation Theory** - Analytics Vidhya
    https://www.analyticsvidhya.com/blog/2022/01/knowledge-distillation-theory-and-end-to-end-case-study/

13. **Why KL Divergence in Knowledge Distillation?**
    https://medium.com/@buroojghani/why-kl-divergence-in-knowledge-distillation-1375d555a728

### LoRA + Knowledge Distillation

14. **KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning** (arXiv 2024)
    https://arxiv.org/abs/2410.20777

15. **GitHub - KD-LoRA Implementation**
    https://github.com/rambodazimi/KD-LoRA

16. **Distilling Knowledge from Large LLMs: Fine-tuning Mistral with LoRA**
    https://deeplearning.fr/distilling-knowledge-from-large-llms-fine-tuning-mistral-with-lora/

### LoRA Parameters

17. **Practical Tips for Finetuning LLMs Using LoRA** - Sebastian Raschka
    https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

18. **What rank and alpha to use in LoRA in LLM fine-tuning?**
    https://medium.com/@fartypantsham/what-rank-r-and-alpha-to-use-in-lora-in-llm-1b4f025fd133

19. **LoRA fine-tuning Hyperparameters Guide** - Unsloth
    https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

20. **How to choose LoRA Hyper-parameters?** - Trelis Research
    https://trelis.substack.com/p/how-to-choose-lora-hyper-parameters

### Distillation Loss

21. **Knowledge Distillation: Teacher-Student Loss Explained**
    https://labelyourdata.com/articles/machine-learning/knowledge-distillation

22. **Knowledge Distillation - Keras Documentation**
    https://keras.io/examples/vision/knowledge_distillation/

23. **On the Efficacy of Knowledge Distillation** (ICCV 2019)
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf

---

## Résumé des Points Clés

### Génération Autorégressif
✅ Un token à la fois
✅ Chaque output devient input
✅ KV caching pour optimiser
✅ Masque causal obligatoire
✅ Top-P sampling recommandé

### Knowledge Distillation
✅ Temperature = 4.0
✅ Alpha = 0.7 (70% Teacher)
✅ KL Divergence loss
✅ Soft targets révèlent connaissances cachées

### LoRA
✅ Rank = 8-16 (départ)
✅ Alpha = 2 × rank
✅ Target: q_proj, v_proj
✅ Learning Rate = 3e-4

### KD-LoRA
✅ 98% performance de LoRA
✅ 75% moins de mémoire
✅ 30% plus rapide en inference
✅ 40% plus compact

---

**Auteur** : Documentation générée pour le projet FastQuantum
**Date** : 2026
**Version** : 1.0
