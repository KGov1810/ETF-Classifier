# Capability Map : Catégorisation multi-étiquettes d'ETF

Statut : **validée** (25/10/2026)

## Intention confirmée

Outil de catégorisation multi-étiquettes d'ETF, entraîné sur un CSV de 5 000 à 10 000 ETF
(Name, Description, Composition pondérée, Catégories). Il propose des catégories pour chaque
nouvel ETF avec un score de confiance, les ETF similaires et une justification LLM (Azure OpenAI).
L'équipe valide ou corrige dans Streamlit ; les validations enrichissent le CSV avec traçabilité ;
le modèle est réentraîné une fois par jour. Projet également destiné à l'apprentissage du ML,
construit par étapes mesurées.

## Modules

| Module id | Responsabilité | Dépend de |
|---|---|---|
| `data` | Chargement du CSV, parsing Composition/Catégories, contrôle qualité, normalisation des sous-jacents, détection des quasi-doublons, split train/test | — |
| `features` | Vectorisation : texte (TF-IDF, puis embeddings Azure mis en cache) et composition pondérée | `data` |
| `model` | Classifieur multi-étiquettes, seuils par catégorie, évaluation (F1 micro/macro), versionnement | `features` |
| `similarity` | ETF les plus proches (recouvrement pondéré de composition + proximité sémantique) | `features` |
| `feedback` | Écriture sécurisée dans le CSV (verrou + sauvegarde), colonnes de traçabilité, création de catégories | `data` |
| `app` | Interface Streamlit : saisie, propositions, ETF similaires, validation | `model`, `similarity`, `feedback` |
| `llm-justify` | Justification LLM Azure, suggestion sur les cas incertains ou les catégories nouvelles | `model`, `similarity` |
| `retraining` | Réentraînement **quotidien** (1 fois par jour), comparaison au modèle en place, promotion seulement si meilleur | `model`, `feedback` |

## Ordre de construction

`data` → `features` → `model` + `similarity` → `feedback` → `app` (v1, sans LLM) → `llm-justify` → `retraining`

## Décisions

- Le CSV reste la source de vérité (pas de base de données en v1).
- Réentraînement quotidien déclenché par le planificateur du serveur (cron ou Planificateur de tâches Windows),
  pas par Streamlit. Le job est ignoré s'il n'y a eu aucune nouvelle validation depuis le dernier entraînement,
  et il respecte le même verrou de fichier que l'interface pour éviter les conflits d'écriture.
- Specs par module : `SPEC-<module-id>.md`, à la racine du projet.
