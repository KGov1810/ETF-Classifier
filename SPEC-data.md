# Spec : module `data`

Statut : **validée** (25/10/2026)

## Objectif

Fournir aux autres modules un jeu de données **fiable, propre et reproductible** à partir du CSV d'ETF :
- lire et convertir les colonnes Composition et Catégories en structures Python ;
- détecter et signaler les lignes invalides sans jamais modifier le CSV source ;
- normaliser les noms des sous-jacents pour que « Apple », « Apple Inc. » et « APPLE INC » soient reconnus comme un seul titre ;
- regrouper les quasi-doublons (ETF répliquant le même indice) ;
- produire un découpage entraînement/test qui évite les fuites entre ces groupes.

Utilisateurs du module : les modules `features`, `model`, `similarity`, `feedback` et `retraining`, ainsi que vous, via un rapport de qualité lisible.

## Données d'entrée (confirmées)

1. CSV encodé en **UTF-8**, séparateur **point-virgule** (`;`). Ces réglages restent modifiables dans `settings.yaml`.
   Les cellules qui contiennent elles-mêmes un `;`, par exemple dans une description, doivent être entre guillemets ; sinon la ligne est signalée comme mal formée.
2. Colonnes : `Name`, `Description`, `Composition`, `Category`.
3. Composition : `[{'AAPL': 6.5}, {'MSFT': 6.1}]`. Les sous-jacents sont des **tickers**, avec des poids en pourcentage et un point décimal.
4. Category : `['Actions Monde', 'ESG']`, de 1 à N catégories de natures différentes par ETF.
5. **Couverture variable** : la composition est complète pour les ETF ayant peu de lignes, et limitée aux principales lignes pour les autres. La somme des poids peut donc aller d'environ 10 à 100.

## Fonctionnalités

### 1. Chargement et parsing
- `load_dataset(path) -> pd.DataFrame` renvoie les colonnes `name`, `description`, `composition`, `categories`, ainsi que `row_id`, la position de la ligne dans le CSV source.
- `composition` est convertie en `dict[str, float]` avec `ast.literal_eval`. On n'utilise jamais `eval`, et le format n'est pas du JSON strict. La liste de dictionnaires à une clé devient un seul dictionnaire.
- `categories` est convertie en `list[str]`. Les espaces autour de chaque catégorie sont retirés et les doublons supprimés.

### 2. Contrôle qualité
Chaque ligne invalide est **exclue du jeu exploitable et listée dans un rapport** avec la raison, mais elle n'est jamais supprimée du CSV. Les règles sont les suivantes :

| Règle | Traitement |
|---|---|
| Composition ou Catégories impossible à lire | Exclue |
| Aucune catégorie | Exclue |
| Poids négatif ou non numérique | Exclue |
| Somme des poids > 105 | Avertissement |
| Composition vide | Conservée, avec un avertissement (la description seule servira) |
| Somme des poids < 100 | Normal (composition partielle) ; la couverture est enregistrée dans une colonne `coverage` |
| Description vide | Conservée, avec un avertissement |
| Sous-jacent présent deux fois dans le même ETF | Poids additionnés, avec un avertissement |
| Nom d'ETF en double | Avertissement |

### 3. Normalisation des tickers
- Règles prudentes : passage en majuscules et suppression des espaces superflus.
- **Les suffixes de place de cotation ne sont pas retirés par défaut** (` US`, `.PA`, ` UW Equity`…). Un même ticker peut désigner des sociétés différentes selon la place, et fusionner à tort serait plus grave que de ne pas fusionner. Le retrait des suffixes reste une option à activer dans `settings.yaml`.
- Un fichier optionnel `config/holding_aliases.csv` (colonnes `raw;canonical`) permet de corriger les cas particuliers à la main, par exemple `GOOG` et `GOOGL`.
- Le rapport liste les tickers ressemblants (même racine, suffixe différent) pour vous aider à remplir ce fichier.
- Le ticker d'origine est conservé ; seule la clé normalisée sert aux calculs.

### 4. Quasi-doublons
- Le recouvrement entre deux ETF A et B vaut `Σ min(wA, wB) / min(ΣwA, ΣwB)`, calculé sur les sous-jacents communs.
  On divise par la **plus petite couverture** et non par 100, à cause des compositions partielles. Deux trackers MSCI World limités à leurs 10 premières lignes (environ 20 % du poids) auraient sinon un recouvrement d'environ 0,20 et ne seraient jamais regroupés.
- Deux ETF dont le recouvrement est ≥ **0,80** (seuil validé, configurable) appartiennent au même groupe. Les groupes sont transitifs (composantes connexes).
- Garde-fou : le regroupement exige au moins 3 tickers en commun, pour éviter qu'une seule grosse ligne partagée suffise à rapprocher deux ETF peu diversifiés.
- Pour les performances, on ne compare que les paires qui partagent au moins un sous-jacent, grâce à un index inversé et des matrices creuses, et non les 50 millions de paires possibles.

### 5. Découpage entraînement/test
- 80/20 par défaut, avec une graine aléatoire fixe pour la reproductibilité.
- **Découpage par groupe** : tous les ETF d'un même groupe de quasi-doublons vont du même côté.
- On vise à ce que chaque catégorie ayant au moins 2 groupes figure des deux côtés, sans librairie supplémentaire en v1.
- Les catégories trop rares (moins de 5 ETF, seuil configurable) sont listées dans le rapport.

### 6. Rapport de qualité
`python -m etf_classifier.data.report` affiche :
- le nombre de lignes valides, en avertissement et exclues, avec la raison ;
- le nombre de catégories, l'effectif par catégorie et le nombre moyen de catégories par ETF ;
- le nombre de groupes de quasi-doublons et la taille du plus gros ;
- les sous-jacents les plus fréquents avant et après normalisation.

Le rapport est aussi écrit dans `reports/data_quality.md`.

## Tech Stack
- Python 3.11 ou plus récent
- pandas, numpy, scipy (matrices creuses), PyYAML
- pytest

## Commandes
```
python -m venv .venv
source .venv/bin/activate        # Linux/macOS  (.venv\Scripts\activate sous Windows)
pip install -r requirements.txt
pytest tests/data -v
python -m etf_classifier.data.report --csv data/raw/etfs.csv
```

## Structure du projet
```
etf-classifier/
├── CAPABILITY-MAP.md
├── SPEC-data.md
├── config/
│   ├── settings.yaml          # noms de colonnes, séparateur, seuils
│   └── holding_aliases.csv    # corrections manuelles (optionnel)
├── data/raw/etfs.csv          # non versionné (.gitignore)
├── reports/                   # rapports générés (non versionnés)
├── src/etf_classifier/data/
│   ├── loader.py              # lecture CSV + orchestration
│   ├── parsing.py             # Composition / Catégories
│   ├── validation.py          # règles qualité
│   ├── normalize.py           # tickers
│   ├── dedup.py               # quasi-doublons
│   ├── split.py               # découpage par groupe
│   └── report.py              # rapport qualité (CLI)
└── tests/data/
    ├── fixtures/sample.csv    # ~20 lignes fictives couvrant les cas limites
    └── test_*.py
```

## Style de code
Fonctions courtes et typées, docstrings en français, aucun effet de bord caché.
```python
def parse_composition(raw: str) -> dict[str, float]:
    """Convertit "[{'Apple': 6.5}, ...]" en {'Apple': 6.5, ...}.

    Lève ParsingError si la cellule n'est pas lisible.
    """
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ParsingError(f"Composition illisible : {raw[:50]}") from exc
    ...
```
- Nommage : `snake_case` pour les fonctions et variables, `PascalCase` pour les classes, constantes en majuscules.
- La configuration est lue depuis `settings.yaml` ; aucun seuil n'est écrit en dur dans le code.
- Code **indépendant du système d'exploitation** (`pathlib`, aucun chemin ni commande propre à un OS), puisque le serveur n'est pas encore choisi.

## Stratégie de test
- pytest, une fixture CSV fictive couvrant chaque règle du tableau de contrôle qualité.
- Tests unitaires pour `parsing`, `validation`, `normalize`, `dedup` et `split`.
- Propriétés vérifiées :
  - aucune fuite de groupe entre entraînement et test ;
  - découpage identique à graine égale ;
  - le CSV source n'est jamais modifié (comparaison d'empreinte avant et après).
- Test de performance : la détection des quasi-doublons sur 10 000 ETF synthétiques prend moins de 60 secondes sur un poste standard.

## Limites
- **Toujours :** lancer les tests avant de proposer un commit ; laisser le CSV source en lecture seule ; journaliser chaque exclusion.
- **Demander d'abord :** ajouter une dépendance ; modifier les règles de qualité ou les seuils par défaut ; changer le format du CSV.
- **Jamais :** versionner `data/raw/` ou des données réelles ; faire un commit ou un push sans validation explicite ; utiliser `eval`.

## Critères de succès
1. Votre CSV complet se charge, et chaque ligne exclue a une raison explicite.
2. Le rapport de qualité est généré en moins de 2 minutes pour 10 000 ETF.
3. Les tests passent, y compris ceux sur l'absence de fuite entre groupes et sur la reproductibilité.
4. Les modules suivants peuvent appeler `load_dataset()` et `make_split()` sans aucun traitement supplémentaire.

## Décisions prises
- Colonnes `Name;Description;Composition;Category`, séparateur `;`, encodage UTF-8.
- Quasi-doublons regroupés par recouvrement pondéré, seuil à 0,80, rapporté à la plus petite couverture.
- Sous-jacents identifiés par ticker ; suffixes de place conservés par défaut.
- Couverture des compositions variable, mesurée dans la colonne `coverage`.
- Serveur et Git pas encore choisis : code portable. Le choix du planificateur est reporté au module `retraining`.

## Questions reportées à des modules ultérieurs
- **`features`** : l'équipe n'a pas de table ticker → secteur/pays. En v1, les variables de composition reposeront uniquement sur les tickers et leurs poids. Un enrichissement (suffixe de place comme indice du pays, ou table construite plus tard) sera étudié dans la spec `features`.
- **`retraining`** : le planificateur dépendra du système d'exploitation retenu pour le serveur.
