# Spec : module `feedback`

Statut : **validée** (25/10/2026)

## Objectif

Enregistrer **de façon sûre et traçable** les validations de l'équipe, pour que chaque ETF validé enrichisse
le jeu de référence et serve au réentraînement quotidien. C'est la brique qui ferme la boucle d'apprentissage.

Exigences :
- **ne jamais perdre ni corrompre de donnée**, même si plusieurs personnes valident en même temps ou si le réentraînement lit le fichier au même moment ;
- **tracer** qui a validé quoi, quand, avec quel modèle, et ce que le modèle avait proposé ;
- **mesurer** si les validateurs acceptent les propositions sans les relire. C'est le risque de biais identifié au début du projet : le modèle finirait par renforcer ses propres erreurs.

Utilisateurs : le module `app` (bouton « Valider »), le job `retraining`, l'équipe via un rapport.
Avant l'interface, une commande permet d'importer un lot de validations depuis un CSV.

Hors périmètre (v1) : la gestion des droits et l'authentification ; la modification des catégories des ETF historiques.

## Stockage

- **Le CSV de référence d'origine reste intact.** Les validations sont ajoutées, ligne après ligne, dans un **journal séparé** `data/validations.csv`, qu'on ne fait que compléter.
- **Le module `data` fusionne les deux fichiers au chargement** : le jeu de référence = CSV d'origine + ETF validés (moins les validations annulées). Le réentraînement voit donc automatiquement les nouveaux ETF.
- Avantages : retour arrière simple, CSV d'origine jamais réécrit, traçabilité complète, écriture courte donc peu de conflits.

Colonnes du journal :

| Colonne | Contenu |
|---|---|
| `ValidationId` | Identifiant unique de la validation |
| `Name`, `Description`, `Composition`, `Category` | L'ETF, au format du CSV de référence ; `Category` = catégories validées |
| `ValidatedBy` | Nom ou e-mail du validateur |
| `ValidatedAt` | Date et heure UTC |
| `ModelVersion` | Version du modèle qui a fait la proposition |
| `PredictedCategories` | Catégories proposées par le modèle |
| `Decision` | `accepté` (proposition inchangée), `corrigé` (catégories ajoutées ou retirées), `annulation` |
| `CancelsId` | Pour une annulation, la validation annulée |
| `Comment` | Commentaire libre facultatif |

## Fonctionnalités

### 1. Enregistrer une validation
- `record_validation(etf, categories, validated_by, prediction=None, model_version=None, comment="")`.
- Contrôles avant écriture, avec des messages clairs :
  - nom non vide ;
  - composition au bon format (même lecture que le module `data`) ;
  - au moins une catégorie ;
  - catégories connues, sauf création explicite (point 3).
- **ETF déjà présent** (même nom dans la référence ou déjà validé) : refus par défaut, avec la référence existante. Le validateur peut confirmer qu'il s'agit d'un ETF différent.
- La décision (`accepté` ou `corrigé`) est déduite en comparant aux catégories proposées.

### 2. Écriture sûre
- **Verrou de fichier** pris pendant chaque écriture et pendant la lecture du journal par le module `data`. Il fonctionne sous Windows, Linux et macOS.
- Une ligne complète écrite en une seule opération ; fichier en UTF-8, séparateur `;`.
- **Sauvegarde quotidienne** du journal dans `data/backups/` (les 30 dernières conservées).
- Le verrou a un délai maximal : passé ce délai, erreur claire (« un autre enregistrement est en cours, réessayez »), jamais d'écriture sans verrou.

### 3. Catégories
- Registre `data/categories.csv` : les catégories connues (historiques et créées), avec qui les a créées, quand, et une description.
- Une catégorie inconnue n'est acceptée qu'avec `create_categories=True`. Sa création est enregistrée.
- Information affichée : une nouvelle catégorie ne sera proposée par le modèle qu'après le réentraînement suivant, et à condition d'avoir assez d'exemples.

### 4. Annulation
- `cancel_validation(validation_id, cancelled_by, comment)` ajoute une ligne `annulation`. On ne supprime jamais rien : l'historique reste complet, et l'ETF annulé disparaît du jeu de référence au chargement suivant.

### 5. Import d'un lot (avant l'interface)
`python -m etf_classifier.feedback.import --csv valides.csv --by "Prénom Nom" [--create-categories]`
- Format d'entrée : celui produit par `model.predict`, complété par l'équipe d'une colonne `Category` (catégories validées).
- Chaque ligne passe par les mêmes contrôles. Le rapport d'import indique les lignes acceptées et refusées, avec la raison.

### 6. Rapport de suivi
`python -m etf_classifier.feedback.report [--since …]` écrit `reports/feedback.md` :
- nombre de validations par jour et par validateur ;
- **taux d'acceptation sans modification**, global et par validateur ; un taux proche de 100 % sur une longue période est signalé comme risque de biais ;
- catégories le plus souvent ajoutées ou retirées par les validateurs, ce qui indique les faiblesses du modèle ;
- nouvelles catégories créées, et validations annulées.

## Tech Stack
- **Nouvelle dépendance** : `filelock` (verrou de fichier portable entre Windows, Linux et macOS)
- Existant : pandas, PyYAML

## Structure
```
data/validations.csv            # journal des validations (non versionné, comme data/raw/)
data/categories.csv             # registre des catégories (non versionné)
data/backups/                   # sauvegardes quotidiennes (non versionnées)
src/etf_classifier/feedback/
├── store.py                    # journal : écriture verrouillée, annulation, lecture, sauvegarde
├── checks.py                   # contrôles avant écriture
├── categories.py               # registre des catégories
├── importer.py                 # CLI d'import d'un lot
└── report.py                   # CLI du rapport de suivi
tests/feedback/
```
Modification d'un module existant : `data.loader` fusionne le CSV d'origine et le journal (validations non annulées), sous le même verrou.

## Stratégie de test
- Contrôles : nom vide, composition illisible, aucune catégorie, catégorie inconnue sans création, ETF déjà présent.
- **Écritures concurrentes** : plusieurs processus valident en même temps ; aucune ligne perdue ni mélangée.
- Verrou indisponible au-delà du délai : erreur claire, aucune écriture.
- Annulation : l'ETF disparaît du jeu de référence, la ligne d'annulation reste dans le journal.
- Fusion par `data` : les ETF validés apparaissent, avec les mêmes contrôles de qualité que le CSV d'origine ; le CSV d'origine n'est jamais modifié (empreinte).
- Rapport : taux d'acceptation calculés à la main sur un journal fictif, alerte de biais.
- Import : lignes acceptées et refusées avec leur raison.

## Limites
- **Toujours :** écrire sous verrou ; ajouter, ne jamais réécrire ni supprimer ; enregistrer qui, quand et avec quel modèle.
- **Demander d'abord :** modifier le format du journal ; toucher au CSV d'origine.
- **Jamais :** écrire sans verrou ; supprimer une ligne du journal ; versionner `data/`.

## Critères de succès
1. Une validation enregistrée apparaît dans le jeu de référence au chargement suivant, donc au prochain réentraînement.
2. Aucune perte ni corruption lors d'écritures simultanées (test de concurrence).
3. Le rapport de suivi donne le taux d'acceptation par validateur et signale un risque de biais.
4. Un lot validé hors interface s'importe, avec un compte rendu ligne par ligne.

## Décisions prises
- **Stockage B** : journal séparé `data/validations.csv`, fusionné au chargement avec le CSV d'origine, qui reste intact.
- **Dépendance `filelock`** acceptée (verrou portable Windows / Linux / macOS).
- **Corrections d'ETF historiques hors v1** ; elles pourront venir sous forme de lignes `correction` dans le journal.
- **Identité du validateur** : nom ou e-mail, sans authentification en v1.
