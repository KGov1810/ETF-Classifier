# Plan de mise en œuvre : modules `costs` puis `features`

Références : `SPEC-costs.md`, `SPEC-features.md` (validées le 25/10/2026) · `CAPABILITY-MAP.md`
Tâches détaillées : `tasks/todo-costs-features.md`

## Vue d'ensemble

1. `costs` d'abord : il ne dépend de rien, et `features` s'en sert dès le premier appel d'embedding.
2. `features` ensuite, en deux temps :
   - les **trois blocs locaux** (TF-IDF, tickers, statistiques de composition) et le pipeline complet,
     utilisables sans Azure ; ils constituent la référence chiffrée du futur module `model` ;
   - puis le **bloc embeddings** (client Azure, cache, coûts, ligne de commande).

Cet ordre permet d'avoir un pipeline fonctionnel de bout en bout avant de toucher au réseau,
et de tester tout le code Azure avec un faux client.

## Décisions d'architecture

- **Interface commune des blocs** (`fit`, `transform`, `feature_names`) : ajouter ou retirer un bloc ne touche pas au pipeline, ce qui facilite les comparaisons pédagogiques.
- **Client Azure injectable** : `embeddings.py` reçoit un client en paramètre. En production c'est `AzureOpenAI` ; en test, un faux client qui renvoie des vecteurs déterministes et un usage de tokens. Aucun test ne touche le réseau.
- **Cache SQLite** (bibliothèque standard) : robuste aux accès concurrents (application et job quotidien), sans dépendance supplémentaire.
- **`CostTracker` appelé dans `embeddings.py` après chaque lot**, jamais par l'appelant : impossible d'oublier de tracer un appel.
- **Sauvegarde du pipeline avec `joblib`** (fourni avec scikit-learn), accompagnée d'un fichier JSON lisible (configuration, version, empreinte du jeu d'entraînement).

## Liste des tâches

### Phase A : module `costs`
- [ ] A1 : grille tarifaire et chargement contrôlé
- [ ] A2 : comptage des tokens et estimations
- [ ] A3 : journal du coût réel (`CostTracker`)
- [ ] A4 : rapport des coûts (CLI)

### Point d'étape A
- [ ] Tests verts sans réseau
- [ ] Vous vérifiez les prix de `config/pricing.yaml` par rapport à votre contrat Azure

### Phase B : `features` sans Azure
- [ ] B1 : dépendances, configuration et encodage des catégories
- [ ] B2 : bloc `text_tfidf`
- [ ] B3 : blocs `holdings` et `composition_stats`
- [ ] B4 : pipeline apprendre/appliquer, sauvegarde et rechargement

### Point d'étape B
- [ ] Tests d'absence de fuite et de reproductibilité verts
- [ ] Construction des 3 blocs sur 10 000 ETF synthétiques en moins de 60 s

### Phase C : `features` avec Azure
- [ ] C1 : client d'embeddings (lots, troncature, nouvelles tentatives), testé avec un faux client
- [ ] C2 : cache SQLite et suivi des coûts par lot
- [ ] C3 : ligne de commande `build` (estimation, confirmation, `--yes`) et `.env.example`

### Point d'étape C : essai réel
- [ ] Vous lancez `build --estimate-only` puis `build` sur un petit échantillon avec votre clé Azure
- [ ] Écart entre estimation et coût réel inférieur à 2 % ; deuxième lancement sans aucun appel Azure

## Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Tarifs par défaut inexacts pour votre région | Moyen | Grille datée et sourcée ; point d'étape A dédié à la vérification ; avertissement si la grille a plus de 90 jours |
| Tokenizer de GPT-6.1 Sol non publié | Faible | Estimation signalée comme approximative ; le coût réel renvoyé par Azure fait foi (servira surtout à `llm-justify`) |
| Limites de débit Azure (erreurs 429) sur 10 000 textes | Moyen | Lots de 64, nouvelles tentatives avec attente croissante, reprise possible grâce au cache |
| Changement de modèle ou de dimension invalidant le cache | Moyen | Modèle et dimension inclus dans la clé du cache ; changement soumis à votre accord |
| Fuite d'information du test vers l'entraînement | Élevé | Apprentissage sur l'entraînement seulement ; test dédié avec un mot et un ticker présents uniquement dans le test |
| Clé Azure exposée | Élevé | `.env` exclu du versionnement ; clé jamais lue ailleurs que dans le client ; `.env.example` sans valeur |

## Questions ouvertes

Aucune bloquante. Pour le point d'étape C, il vous faudra les noms exacts de vos déploiements Azure et la version d'API, à saisir dans votre `.env`.
