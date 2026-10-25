# ETF Classifier

Outil de **catégorisation multi-étiquettes d'ETF** par machine learning : à partir d'un jeu de
5 000 à 10 000 ETF déjà catégorisés, il propose une ou plusieurs catégories pour chaque nouvel ETF.
Une personne de l'équipe valide ou corrige la proposition, et l'ETF validé enrichit le jeu de données.
Le modèle est réentraîné une fois par jour.

Le projet sert aussi de terrain d'apprentissage du machine learning : il est construit par étapes,
chacune mesurée par rapport à une référence simple.

## État d'avancement

| Module | Rôle | État |
|---|---|---|
| `data` | Chargement, contrôle qualité, normalisation, quasi-doublons, découpage | ✅ Fait |
| `features` | Vectorisation du texte et de la composition | À spécifier |
| `model` | Classifieur multi-étiquettes et évaluation | — |
| `similarity` | ETF les plus proches | — |
| `feedback` | Enregistrement des validations dans le CSV | — |
| `app` | Interface Streamlit | — |
| `llm-justify` | Justification par LLM (Azure OpenAI) | — |
| `retraining` | Réentraînement quotidien | — |

Voir `CAPABILITY-MAP.md` pour l'architecture et l'ordre de construction, et `SPEC-data.md` pour le détail du module `data`.

## Format des données attendu

CSV en UTF-8, séparateur `;`, avec ces colonnes :

| Colonne | Exemple |
|---|---|
| `Name` | `iShares Core MSCI World` |
| `Description` | `Réplique l'indice MSCI World…` |
| `Composition` | `[{'AAPL': 5.1}, {'MSFT': 4.6}, {'NVDA': 4.2}]` (tickers et poids en %) |
| `Category` | `['Actions Monde', 'ESG']` |

Les cellules contenant un `;` (par exemple une description) doivent être entre guillemets.
La composition peut être partielle (principales lignes seulement).

## Installation

Python 3.11 ou plus récent.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

Placer le CSV dans `data/raw/` (ce dossier n'est jamais versionné), puis générer le rapport de qualité :

```bash
python -m etf_classifier.data.report --csv data/raw/etfs.csv
```

Le rapport s'affiche et est écrit dans `reports/data_quality.md`. Il indique les lignes exclues
et leur raison, les catégories, la couverture des compositions, les tickers à harmoniser,
les groupes de quasi-doublons et le découpage entraînement/test.

Depuis Python, pour les modules suivants :

```python
from etf_classifier.data.loader import prepare_dataset

prepared = prepare_dataset("data/raw/etfs.csv")
prepared.split.train   # ETF d'entraînement
prepared.split.test    # ETF de test (aucun quasi-doublon commun avec l'entraînement)
prepared.log           # exclusions et avertissements
```

## Tests

```bash
pytest -v                 # suite complète (≈ 15 s)
pytest -m "not slow"      # sans le test de performance
```

## Configuration

Tous les seuils sont dans `config/settings.yaml` (séparateur, noms de colonnes, seuil de
recouvrement des quasi-doublons, taille du jeu de test…). Les tickers à fusionner manuellement
(par exemple `GOOG` et `GOOGL`) se déclarent dans `config/holding_aliases.csv`.

## Structure

```
config/                  configuration et alias de tickers
data/raw/                CSV réel (non versionné)
reports/                 rapports générés (non versionnés)
src/etf_classifier/data/ module data
tests/data/              tests et CSV fictif
tasks/                   plan et tâches
```
