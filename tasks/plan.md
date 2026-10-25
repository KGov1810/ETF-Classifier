# Plan de mise en œuvre : module `data`

Référence : `SPEC-data.md` (validée le 25/10/2026) · `CAPABILITY-MAP.md`

## Vue d'ensemble

Construire le module qui charge le CSV d'ETF (`Name;Description;Composition;Category`), le contrôle,
normalise les tickers, regroupe les quasi-doublons et produit un découpage entraînement/test sans fuite.
On avance par tranches verticales : dès la tâche 4, une commande produit un rapport sur le CSV réel,
ce qui fait remonter très tôt les surprises de format.

## Décisions d'architecture

- **Rapport de qualité dès la phase 1** : le premier risque est l'écart entre le format supposé et le CSV réel. Le rapport sert de test grandeur nature et grandit à chaque tâche.
- **Fonctions pures + orchestrateur** : `parsing`, `validation`, `normalize`, `dedup` et `split` ne lisent ni n'écrivent de fichier ; seuls `loader` et `report` font des entrées/sorties. Cela facilite les tests et la réutilisation par `features` et `retraining`.
- **Configuration centralisée** dans `config/settings.yaml`, lue une seule fois et passée explicitement aux fonctions.
- **Matrices creuses (scipy) + index inversé** pour les quasi-doublons : seules les paires d'ETF partageant au moins un ticker sont comparées.
- **Aucune dépendance hors spec** : pandas, numpy, scipy, PyYAML, pytest.

## Liste des tâches

Détail complet dans `tasks/todo.md`.

### Phase 1 : chargement de bout en bout
- [x] Tâche 1 : squelette du projet et configuration
- [x] Tâche 2 : parsing de Composition/Category et `load_dataset()` minimal
- [x] Tâche 3 : règles de qualité, exclusions motivées, colonne `coverage`
- [x] Tâche 4 : rapport de qualité minimal (CLI + Markdown)

### Point d'étape 1 : premier passage sur le CSV réel
- [ ] Tests verts
- [ ] Rapport généré sur le CSV réel ; taux d'exclusion et anomalies de format relus avec vous

### Phase 2 : fiabilisation des données
- [x] Tâche 5 : normalisation des tickers et fichier d'alias
- [x] Tâche 6 : détection des quasi-doublons (recouvrement pondéré, groupes)

### Point d'étape 2
- [ ] Tests verts, test de performance (10 000 ETF en moins de 60 s) passé
- [ ] Revue avec vous des groupes de quasi-doublons sur le CSV réel (échantillon des 10 plus gros groupes)

### Phase 3 : découpage et finalisation
- [x] Tâche 7 : découpage entraînement/test par groupe
- [x] Tâche 8 : rapport complet et contrôle de non-modification du CSV source

### Point d'étape final
- [ ] Les 4 critères de succès de `SPEC-data.md` sont remplis
- [ ] Revue avec vous avant de passer au module `features`

## Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Le CSV réel s'écarte du format supposé (guillemets, `;` dans les descriptions, poids en texte) | Élevé | Rapport dès la tâche 4 ; chaque ligne rejetée affiche sa raison et un extrait |
| Effet de chaîne dans les quasi-doublons : A≈B, B≈C… crée un groupe géant | Moyen | Taille du plus gros groupe dans le rapport ; avertissement au-delà de 2 % du jeu ; seuil ajustable |
| Détection des quasi-doublons trop lente | Moyen | Index inversé + matrices creuses ; test de performance dédié |
| Catégories rares absentes du jeu de test | Moyen | Liste des catégories sous le seuil dans le rapport ; regroupement éventuel décidé avec vous |
| Inclusion des compositions partielles : un ETF ne listant que quelques grandes lignes peut être regroupé avec un ETF plus large qui les contient (constaté sur la fixture : World ESG avec US Tech) | Moyen | À examiner au point d'étape 2 sur le CSV réel ; si besoin, condition supplémentaire (couverture minimale, ratio de nombre de lignes) à valider avec vous |
| Confidentialité du CSV réel | Élevé | `data/raw/` exclu du versionnement ; tests uniquement sur données fictives |

## Statut (25/10/2026)

Tâches 1 à 8 codées et testées (90 tests verts). Points d'étape 1 et 2 en attente du CSV réel.

## Questions ouvertes

1. **Où vivra le code ?** Soit ici, dans cet espace de travail, et je vous envoie les fichiers ; soit dans un dossier de votre ordinateur connecté via l'app de bureau Claude, où j'écris directement.
2. **Le CSV réel** : pourrez-vous le déposer pour le point d'étape 1, ou préférez-vous lancer le rapport vous-même et me transmettre le résultat (qui ne contient que des statistiques et des extraits de lignes rejetées) ?
