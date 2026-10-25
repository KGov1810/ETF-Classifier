# Spec : module `costs`

Statut : **validée** (25/10/2026)

## Objectif

Rendre visible et maîtrisable chaque dépense Azure OpenAI du projet :
- **estimer avant** un appel ou un lot d'appels (« calculer les embeddings de 8 000 ETF coûtera environ 0,05 $ ») ;
- **enregistrer après** chaque appel le coût réel, à partir du nombre de tokens renvoyé par Azure ;
- **résumer** les dépenses par jour, par module (`features`, `llm-justify`…) et par opération.

Utilisateurs : les modules `features` et `llm-justify` (appels), le job `retraining` (estimation sans interaction), l'équipe via un rapport, puis plus tard l'interface Streamlit.

Hors périmètre : le rapprochement avec la facture Azure (API Cost Management), les budgets et alertes (pas de contrainte de budget), le tarif « Provisioned » (PTU).

## Données d'entrée

1. **Grille tarifaire** `config/pricing.yaml`, en **USD par million de tokens**, datée et sourcée :
   ```yaml
   currency: USD
   as_of: "2026-10-25"
   source: "Tarifs publics Data Zone Standard (borne haute de la fourchette régionale), à vérifier dans votre contrat Azure"
   models:
     text-embedding-3-large:
       kind: embedding
       tokenizer: cl100k_base
       input_per_million: 0.13
     gpt-6.1-sol:
       kind: chat
       tokenizer: o200k_base          # approximation : tokenizer non publié
       input_per_million: 2.40
       cached_input_per_million: 0.12
       output_per_million: 12.00       # inclut les tokens de raisonnement
       long_context_threshold: 272000  # au-delà, tarif long contexte
       long_context:
         input_per_million: 4.80
         cached_input_per_million: 0.24
         output_per_million: 18.00
   deployments:                        # nom du déploiement Azure → modèle de la grille
     text-embedding-3-large: text-embedding-3-large
     gpt-6.1-sol: gpt-6.1-sol
   ```
   Les valeurs par défaut sont les tarifs publics **Data Zone Standard**, à la borne haute de la fourchette régionale (estimation prudente).
   **Elles sont à vérifier** dans votre contrat : le prix exact dépend de la région. Le tarif de `text-embedding-3-large` en Data Zone n'a pas été confirmé par une source publique (0,13 $ est le tarif standard).
2. **Usage renvoyé par Azure** après chaque appel : `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens` le cas échéant.

## Fonctionnalités

### 1. Comptage des tokens
- `count_tokens(text, model) -> int`, avec le tokenizer déclaré dans la grille (bibliothèque `tiktoken`).
- Pour `text-embedding-3-large`, le tokenizer `cl100k_base` est exact. Pour GPT-6.1 Sol, il n'est pas publié : le comptage est une **approximation**, signalée comme telle. Le coût réel enregistré après l'appel fait toujours foi.

### 2. Estimation avant appel
- `estimate_embeddings(texts, deployment) -> CostEstimate` : tokens et coût en USD pour un lot de textes.
- `estimate_chat(messages, deployment, expected_output_tokens) -> CostEstimate` : coût d'entrée exact au tokenizer près, coût de sortie selon une hypothèse explicite.
- `CostEstimate` contient : nombre de tokens, coût en USD, date de la grille utilisée, indicateur « approximatif ».

### 3. Enregistrement du coût réel
- `CostTracker.record(module, operation, deployment, usage, metadata=None)` calcule le coût et ajoute une ligne au journal `logs/azure_usage.csv` (séparateur `;`, UTF-8) :

  | Colonne | Exemple |
  |---|---|
  | `timestamp_utc` | `2026-10-25T14:03:11Z` |
  | `module` | `features` |
  | `operation` | `embeddings_batch` |
  | `deployment` | `text-embedding-3-large` |
  | `input_tokens` / `cached_input_tokens` / `output_tokens` | `48211` / `0` / `0` |
  | `cost_usd` | `0.006267` |
  | `pricing_as_of` | `2026-10-25` |
  | `estimated_cost_usd` | estimation préalable, si elle existe |
  | `note` | identifiant de lot, nombre de textes… |
- Le tarif long contexte s'applique quand les tokens d'entrée d'un appel dépassent le seuil.
- **Déploiement absent de la grille** : l'appel est quand même enregistré, avec ses tokens, mais `cost_usd` vide et un avertissement. On ne perd jamais une trace, et on n'affiche jamais un faux coût nul.
- Écriture : une ligne complète par écriture, en mode ajout. Le verrou de fichier partagé avec l'interface viendra avec le module `feedback`.

### 4. Rapport des coûts
`python -m etf_classifier.costs.report [--since 2026-10-01] [--until …]` affiche et écrit dans `reports/azure_costs.md` :
- le total en USD sur la période ;
- les totaux par jour, par module, par opération et par déploiement ;
- l'écart entre estimations et coûts réels, quand les deux existent ;
- la date de la grille tarifaire utilisée, avec un avertissement si elle a plus de 90 jours.

## Tech Stack
- Python 3.11+, pandas, PyYAML (déjà présents)
- **Nouvelle dépendance** : `tiktoken` (comptage des tokens)

## Commandes
```
pytest tests/costs -v
python -m etf_classifier.costs.report --since 2026-10-01
```

## Structure
```
config/pricing.yaml                 # grille tarifaire datée (versionnée)
logs/azure_usage.csv                # journal des appels (non versionné)
src/etf_classifier/costs/
├── pricing.py                      # lecture et contrôle de la grille
├── tokens.py                       # comptage des tokens
├── estimate.py                     # estimations avant appel
├── tracker.py                      # enregistrement du coût réel
└── report.py                       # rapport des coûts (CLI)
tests/costs/
```

## Style de code
Identique au module `data` : fonctions courtes et typées, docstrings en français, seuils et prix uniquement dans la configuration. Montants calculés en `float` et arrondis à 6 décimales dans le journal.

## Stratégie de test
- Calculs de coût vérifiés à la main : 1 000 000 de tokens d'embedding coûtent exactement 0,13 $ avec la grille de test.
- Tarif long contexte, tokens en cache, tokens de sortie.
- Déploiement inconnu : la ligne est enregistrée sans coût, avec un avertissement.
- Grille invalide (prix manquant ou négatif) : erreur claire au chargement.
- Rapport : totaux par jour et par module sur un journal fictif.
- Aucun appel réseau dans les tests.

## Limites
- **Toujours :** passer par `CostTracker` pour tout appel Azure ; dater la grille tarifaire.
- **Demander d'abord :** modifier la grille par défaut ; ajouter une dépendance autre que `tiktoken`.
- **Jamais :** écrire un prix en dur dans le code ; versionner `logs/` ; enregistrer le contenu des textes envoyés dans le journal (seulement les compteurs).

## Critères de succès
1. Chaque appel Azure du projet produit une ligne dans le journal.
2. Sur un lot d'embeddings, l'estimation préalable et le coût réel diffèrent de moins de 2 %.
3. Le rapport donne le total par jour, par module et par opération.
4. Les tests passent sans accès réseau.

## Décisions prises
- LLM : **GPT-6.1 Sol**, déploiement **Data Zone Standard**.
- Devise : USD. Grille par défaut à la borne haute des tarifs Data Zone, à ajuster selon votre région.
