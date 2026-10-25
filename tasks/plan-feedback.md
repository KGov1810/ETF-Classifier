# Plan de mise en œuvre : module `feedback`

Référence : `SPEC-feedback.md` (validée le 25/10/2026) · Tâches détaillées : `tasks/todo-feedback.md`

## Vue d'ensemble

1. **Le journal sûr** : écriture verrouillée, contrôles et registre des catégories, annulation et sauvegarde. Le test de concurrence est le point d'étape critique : aucune ligne perdue ni mélangée quand plusieurs processus valident en même temps.
2. **La boucle fermée** : le module `data` fusionne le CSV d'origine et le journal, puis viennent l'import par lot et le rapport de suivi.

## Décisions d'architecture

- **Un seul point d'écriture** (`store.py`) : toute écriture passe par la même fonction, sous verrou. Les autres modules ne touchent jamais au fichier directement.
- **Fichier verrou à côté du journal** (`validations.csv.lock`, bibliothèque `filelock`), avec un délai maximal configurable.
- **Identifiant de validation** : date UTC + suffixe aléatoire court (`20261025-142233-a1b2`), lisible et unique sans compteur partagé.
- **Lignes du journal identifiées à part** au chargement (`source = validation`), avec des `row_id` qui ne peuvent pas entrer en collision avec ceux du CSV d'origine. Les contrôles de qualité s'appliquent à l'identique.
- **Registre des catégories initialisé automatiquement** à partir du CSV d'origine au premier usage. Les catégories historiques sont donc connues sans saisie.

## Modification d'un module existant

`data.loader` : `load_dataset()` et `prepare_dataset()` ajoutent les validations non annulées du journal, lues sous le même verrou. Sans journal, le comportement reste strictement identique : les 92 tests du module `data` doivent rester verts.

## Liste des tâches

### Phase A : le journal sûr
- [x] F1 : configuration, dépendance `filelock` et écriture verrouillée du journal
- [x] F2 : contrôles, registre des catégories et enregistrement d'une validation
- [x] F3 : annulation et sauvegarde quotidienne

### Point d'étape A
- [x] Test de concurrence vert : plusieurs processus, aucune ligne perdue ni mélangée
- [x] Verrou indisponible : erreur claire, aucune écriture

### Phase B : la boucle fermée
- [x] F4 : fusion du journal par le module `data`
- [x] F5 : commande d'import d'un lot
- [x] F6 : rapport de suivi (taux d'acceptation, alerte de biais)

### Point d'étape final
- [ ] Les 4 critères de succès de `SPEC-feedback.md`
- [x] Boucle complète sur la fixture : `predict` → import des validations → `train` qui voit les nouveaux ETF

## Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Verrou peu fiable sur un lecteur réseau (partage Windows, NFS) | Élevé | Recommander un disque local au serveur pour `data/` ; documenté dans le README |
| Journal ouvert dans Excel sous Windows (fichier bloqué) | Moyen | Erreur claire « fichier utilisé par une autre application » ; jamais d'écriture partielle |
| Collision de `row_id` entre CSV d'origine et journal | Moyen | Espace d'identifiants séparé et colonne `source` ; test dédié |
| Seuil d'alerte de biais mal calibré | Faible | Seuil et nombre minimal de validations configurables (95 % sur au moins 50 validations par défaut) |
| Régression du module `data` | Moyen | Sans journal, comportement identique ; suite `data` relancée |

## Statut (25/10/2026)

Tâches F1 à F6 codées et testées (291 tests verts sur l'ensemble du projet). Concurrence : 4 processus,
60 validations simultanées, aucune perte. Boucle complète vérifiée sur la fixture (15 → 17 ETF au réentraînement).

## Questions ouvertes

Aucune bloquante.
