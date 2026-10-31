# Guide du machine learning du projet (pour débutants)

Ce document explique, sans prérequis, ce que fait la partie « machine learning » du projet, avec les termes
techniques définis au fur et à mesure et un fonctionnement détaillé « en coulisses ». Il décrit le code tel qu'il
est dans le dépôt (`src/etf_classifier/`) ; les réglages cités viennent de `config/settings.yaml`.

Sommaire : [1. Vue d'ensemble](#1-vue-densemble) · [2. Glossaire](#2-glossaire) ·
[3. Les données et leur préparation](#3-les-données-et-leur-préparation) ·
[4. Transformer un ETF en nombres](#4-transformer-un-etf-en-nombres-les-features) ·
[5. Le modèle](#5-le-modèle-la-régression-logistique-multinomiale) ·
[6. Mesurer la qualité](#6-mesurer-la-qualité) · [7. Réentraînement quotidien](#7-réentraînement-quotidien-et-promotion) ·
[8. Humain dans la boucle](#8-lhumain-dans-la-boucle) · [9. Limites](#9-limites-et-points-de-vigilance) ·
[10. Où est quoi dans le code](#10-où-est-quoi-dans-le-code)

---

## 1. Vue d'ensemble

**Le problème.** Nous avons des milliers d'ETF (fonds cotés). Chacun a un ticker, un nom, une description, une
composition (les titres qu'il détient, avec leur poids en %) et une **catégorie** (« Equities – Technology »,
« Fixed Income – Government »…). Classer à la main est long et sujet à erreurs. On veut qu'un programme **propose**
la catégorie d'un ETF, et qu'un humain **valide**.

**L'idée du machine learning (ML).** Au lieu d'écrire des règles à la main (« si le nom contient *gold* alors
Commodities »), on montre au programme des milliers d'exemples déjà classés, et il **déduit lui-même** des règles
chiffrées. On dit qu'il *apprend*.

**Analogie.** Imagine un nouveau collègue à qui on donne 1 500 fiches d'ETF déjà classées. Après les avoir lues,
il remarque que les fonds qui parlent de « bullion » et détiennent des mineurs d'or sont presque toujours en
« Commodities ». Face à une fiche nouvelle, il propose une catégorie par ressemblance avec ce qu'il a vu. Notre
modèle fait la même chose, mais avec des calculs.

**Le déroulé complet :**

```
CSV (Ticker;Nom;Description;Classification;Composition)
   │  contrôle qualité : on exclut les lignes inutilisables
   ▼
Séparation train / test (par groupes de quasi-doublons)
   │
   ▼
Features : chaque ETF devient une liste de nombres (texte, titres détenus, forme du portefeuille, embeddings)
   │
   ▼
Entraînement : la régression logistique ajuste ses coefficients sur le train
   │
   ▼
Évaluation sur le test : accuracy, F1 macro, comparaison à la baseline
   │
   ▼
Prédiction : une catégorie proposée par ETF + scores des autres catégories
   │
   ▼
Validation humaine (app) → journal → réentraînement quotidien → promotion si le nouveau modèle est meilleur
```

---

## 2. Glossaire

### Les bases

| Terme | Définition simple |
|---|---|
| **Apprentissage supervisé** | Le modèle apprend à partir d'exemples dont on connaît la bonne réponse. Ici : des ETF avec leur catégorie. |
| **Classification** | Prédire une *catégorie* parmi un nombre fini (par opposition à la *régression*, qui prédit un nombre). |
| **Mono-catégorie** | Chaque ETF a exactement une catégorie (c'est le choix fait dans ce projet). |
| **Label / étiquette / cible** | La bonne réponse connue : la colonne `Classification`. |
| **Feature (variable)** | Une information chiffrée décrivant un exemple (par ex. « le mot *gold* apparaît », « le poids de la plus grosse ligne est 12 % »). Le modèle ne lit que des nombres. |
| **Matrice de features** | Un tableau : une ligne par ETF, une colonne par feature. |
| **Matrice creuse (sparse)** | Matrice où presque toutes les cases valent 0. On ne stocke que les cases non nulles, ce qui économise énormément de mémoire (une description n'utilise que quelques mots sur 50 000 possibles). |
| **Modèle** | La « formule » apprise : des nombres (coefficients) qui transforment les features d'un ETF en scores par catégorie. |
| **Entraîner (fit)** | Ajuster les coefficients pour que les réponses du modèle collent au mieux aux exemples connus. |
| **Prédire / inférer** | Appliquer le modèle entraîné à un ETF qu'il n'a pas vu. |
| **Score / probabilité** | Nombre entre 0 et 1 attribué à chaque catégorie ; ici les scores d'un ETF totalisent 1 (100 %). |
| **argmax** | « La position du plus grand » : la catégorie au score le plus élevé est celle qu'on propose. |

### Les jeux de données

| Terme | Définition simple |
|---|---|
| **Jeu d'entraînement (train)** | Les exemples (≈ 80 %) sur lesquels le modèle apprend. |
| **Jeu de test** | Les exemples (≈ 20 %) mis de côté, jamais vus pendant l'apprentissage, pour mesurer honnêtement la qualité. C'est l'examen final. |
| **Jeu de validation** | Une petite partie du train (20 %) servant à *choisir les réglages* sans toucher au test. |
| **Fuite de données (data leakage)** | Quand une information du test « fuit » dans l'entraînement. Le score paraît alors excellent mais est faux. Exemple : deux classes de parts d'un même fonds, l'une en train, l'autre en test : le modèle « connaît déjà la réponse ». |
| **Surapprentissage (overfitting)** | Le modèle apprend les exemples *par cœur* (y compris leur bruit) et généralise mal aux nouveaux cas. Comme un élève qui mémorise les annales sans comprendre. |
| **Généralisation** | Capacité à bien traiter des cas jamais vus. C'est le vrai objectif. |

### Les réglages

| Terme | Définition simple |
|---|---|
| **Paramètre** | Ce que le modèle apprend seul (les coefficients). |
| **Hyperparamètre** | Un réglage choisi *avant* l'apprentissage (ici `C`). On l'essaie sur plusieurs valeurs et on garde la meilleure. |
| **Régularisation (paramètre C)** | Une « pénalité » qui empêche les coefficients de devenir trop grands, donc qui limite le surapprentissage. **C petit = pénalité forte = modèle plus simple** ; C grand = modèle plus libre. |
| **Seed (graine)** | Nombre qui fixe le hasard, pour obtenir les mêmes résultats à chaque exécution. |
| **`class_weight="balanced"`** | Donne plus d'importance aux exemples des catégories rares, pour qu'elles ne soient pas écrasées par les catégories fréquentes. |

### Les mesures de qualité

| Terme | Définition simple |
|---|---|
| **Accuracy (exactitude)** | Part des ETF correctement classés. 93 % = 93 ETF sur 100 bien classés. |
| **Précision** (d'une catégorie) | Parmi les ETF *annoncés* « Gold », quelle part l'est vraiment ? |
| **Rappel (recall)** | Parmi les ETF *réellement* « Gold », quelle part le modèle a-t-il retrouvée ? |
| **F1** | Moyenne harmonique de la précision et du rappel : un seul chiffre qui n'est bon que si les deux le sont. |
| **Macro F1** | Moyenne des F1 de *toutes* les catégories, chacune comptant autant. C'est la **métrique principale** du projet, car elle ne laisse pas une catégorie rare mal classée passer inaperçue. |
| **Baseline (référence triviale)** | Un « modèle » idiot qui répond toujours la catégorie la plus fréquente. Si notre modèle ne le bat pas nettement, il n'apprend rien. |
| **Matrice / table de confusion** | La liste des erreurs typiques (« vrai Gold, prédit Silver : 7 fois »). |
| **Support** | Le nombre d'exemples réels d'une catégorie dans le jeu de test. |

### Les techniques de transformation du texte

| Terme | Définition simple |
|---|---|
| **TF-IDF** | Une façon de transformer un texte en nombres : un mot compte plus s'il est *fréquent dans ce texte* (TF) et *rare dans l'ensemble des textes* (IDF). |
| **n-gramme** | Une suite de n mots. Ici on garde les mots seuls et les paires (« emerging markets »). |
| **Stop-words** | Mots trop courants et sans sens (the, of, and) que l'on retire. |
| **Embedding (plongement)** | Un texte converti en liste de 3 072 nombres par un modèle d'IA (Azure OpenAI), de sorte que **deux textes de sens proche donnent des listes proches**. |
| **Normalisation L2** | Mise à l'échelle d'une ligne de nombres pour que sa « longueur » vaille 1, afin que les ETF à long texte ne dominent pas. |
| **Standardisation** | Recentrer une variable (moyenne 0) et la mettre à une échelle commune (écart-type 1). |

---

## 3. Les données et leur préparation

### 3.1 Le contrôle qualité

Le CSV d'entrée est `Ticker;Nom;Description;Classification;Composition`. Avant tout apprentissage
(`data/validation.py`), certaines lignes sont **exclues** et listées dans un rapport :

- ticker vide, nom vide, classification vide ;
- description **et** composition vides (le modèle n'aurait rien à analyser) ;
- composition illisible, poids négatif ou non numérique ;
- ticker présent plusieurs fois : **les deux lignes** sont exclues, car on ne peut pas savoir laquelle est juste.

Certaines situations ne donnent qu'un **avertissement** (la ligne est gardée) : description vide mais composition
présente, composition vide mais description présente, poids total supérieur à 105 %.

> **Pourquoi c'est important en ML :** un modèle est aussi bon que ses données. Des étiquettes fausses ou
> contradictoires lui apprennent des erreurs (« garbage in, garbage out »).

### 3.2 Les groupes de quasi-doublons

Deux ETF dont les portefeuilles se recouvrent à plus de 80 % (avec au moins 3 titres communs) sont placés dans
un même **groupe** (par ex. deux classes de parts du même fonds).

### 3.3 La séparation train / test

On met environ 20 % des ETF de côté pour le test. **Un groupe n'est jamais coupé** entre train et test : sinon on
retombe dans la fuite de données décrite plus haut. Le découpage veille aussi à ce que chaque catégorie présente
dans au moins deux groupes ait un représentant en test, tout en gardant au moins un exemple d'entraînement par
catégorie.

---

## 4. Transformer un ETF en nombres (les features)

Le modèle ne comprend que des nombres. Quatre « blocs » transforment chaque ETF en une longue liste de nombres ;
ces listes sont ensuite **mises bout à bout** (`features/pipeline.py`).

Règle d'or : **tout ce qui est « appris » (vocabulaire, IDF, moyennes) vient du train uniquement**, puis est
appliqué tel quel au test et aux nouveaux ETF. Sinon, le test influencerait la préparation : c'est de la fuite.

### 4.1 Bloc `text_tfidf` : les mots

Le texte est « Nom. Description ». Il est découpé en mots et en paires de mots, sans les stop-words. Chaque terme
reçoit un poids TF-IDF.

**Intuition de l'IDF.** Si un mot apparaît dans presque tous les ETF (« fund »), il ne distingue rien : poids
faible. S'il n'apparaît que dans quelques-uns (« bullion »), il est très informatif : poids élevé. Formule
(celle de scikit-learn) : `IDF = ln((1 + N) / (1 + df)) + 1`, avec N le nombre d'ETF et df le nombre d'ETF
contenant le terme.

Exemple avec N = 1 000 :

| Terme | df | IDF | Lecture |
|---|---|---|---|
| fund | 900 | ln(1001/901) + 1 ≈ **1,1** | banal, peu informatif |
| bullion | 10 | ln(1001/11) + 1 ≈ **5,5** | rare, très informatif |

Résultat : pour chaque ETF, un vecteur d'environ 50 000 cases (une par terme), presque toutes à 0. Un terme doit
apparaître dans au moins 2 ETF pour être gardé (`min_df: 2`), pour éviter d'apprendre des mots uniques.

### 4.2 Bloc `holdings` : les titres détenus

Même idée, mais le « vocabulaire » est l'ensemble des **tickers des titres détenus**. Chaque case contient
`poids (%) × IDF du titre`. Un fonds qui détient 8 % de NEM (mine d'or) a donc une case NEM élevée. Les titres
présents dans tous les fonds (Apple, Microsoft…) ont un IDF faible, car ils ne renseignent pas sur la catégorie.
Un titre inconnu à l'entraînement est ignoré.

> **Pourquoi c'est puissant :** deux ETF aux descriptions différentes mais aux portefeuilles proches finissent
> avec des vecteurs proches, donc probablement dans la même catégorie.

### 4.3 Bloc `composition_stats` : la forme du portefeuille

Sept nombres résumant le portefeuille, standardisés :

| Statistique | Ce qu'elle dit |
|---|---|
| `coverage` | Somme des poids connus (le portefeuille est-il complet ?) |
| `log_n_holdings` | Nombre de lignes (en échelle logarithmique) |
| `top1_weight` | Poids de la plus grosse ligne |
| `top10_concentration` | Part des 10 plus grosses lignes |
| `herfindahl` | Indice de concentration (somme des carrés des parts) : élevé = portefeuille concentré |
| `is_empty` | 1 si la composition est vide |
| `unknown_weight_share` | Part du poids sur des titres jamais vus à l'entraînement |

Ces chiffres aident à distinguer, par exemple, un fonds sur un seul actif (très concentré) d'un fonds indiciel
large (des centaines de lignes).

### 4.4 Bloc `embeddings` : le sens du texte

Le texte « Nom. Description » est envoyé à **Azure OpenAI** (`text-embedding-3-large`), qui renvoie 3 072 nombres.
Ces nombres positionnent le texte dans un « espace de sens » : des textes proches par le sens y sont proches,
même sans mot commun (« bullion » et « precious metals »). TF-IDF, lui, ne voit que les mots exacts.

Garde-fous du projet :
- un **cache SQLite** : un même texte n'est jamais payé deux fois ;
- une **estimation du coût** avant l'appel, avec confirmation, puis le coût réel suivi en USD ;
- si Azure n'est pas configuré, ce bloc est désactivé et le reste fonctionne.

### 4.5 L'assemblage

Les quatre matrices sont multipliées par un poids (tous à 1,0 pour l'instant) puis concaténées : un ETF devient
une ligne d'environ 50 000 + (nombre de titres) + 7 + 3 072 cases. C'est cette ligne que le modèle reçoit.

---

## 5. Le modèle : la régression logistique multinomiale

Malgré son nom, c'est un modèle de **classification**. Il est volontairement simple : rapide, stable, et surtout
**lisible** (on peut voir quelles features poussent vers quelle catégorie).

### 5.1 Calculer un score par catégorie

Pour chaque catégorie *k*, le modèle possède un coefficient par feature et un décalage (biais). Le **score brut**
d'un ETF est :

```
score_k = (coef_k,1 × feature_1) + (coef_k,2 × feature_2) + … + biais_k
```

Un coefficient **positif** signifie « cette feature pousse vers la catégorie k » ; **négatif**, « elle en éloigne ».
Exemple fictif pour « Commodities » : coefficient élevé sur le terme `bullion` et sur le titre `NEM`.

### 5.2 Le softmax : des scores bruts aux probabilités

Les scores bruts peuvent être n'importe quels nombres. Le **softmax** les convertit en probabilités qui totalisent 1 :
`proba_k = exp(score_k) / somme des exp(score_j)`.

Exemple avec trois catégories :

| Catégorie | Score brut | exp(score) | Probabilité |
|---|---|---|---|
| Commodities | 2,0 | 7,39 | **65,9 %** |
| Equities – Technology | 1,0 | 2,72 | 24,2 % |
| Fixed Income | 0,1 | 1,11 | 9,9 % |

Le modèle propose **Commodities** (le plus grand score, c'est l'`argmax`). Si le meilleur score est inférieur à
0,5 (`uncertain_threshold`), l'ETF est marqué **« incertain »** : le validateur sait qu'il faut regarder de plus
près. L'app montre aussi les 2–3 catégories suivantes avec leur score, pour pouvoir corriger en un clic.

### 5.3 L'entraînement : comment trouve-t-on les coefficients ?

Au départ, les coefficients sont quelconques. L'entraînement les ajuste pour **réduire une « perte »** (la
*log-loss*) : elle est faible quand le modèle donne une forte probabilité à la *bonne* catégorie, et très élevée
quand il donne une faible probabilité à la bonne réponse.

En pratique, un algorithme d'optimisation (par défaut `lbfgs` dans scikit-learn) répète : calculer la perte sur
tous les exemples, regarder dans quelle direction chaque coefficient devrait bouger pour la réduire, les bouger un
peu, recommencer. Il s'arrête à la convergence ou après `max_iter = 2000` tours.

À cette perte s'ajoute une **pénalité sur la taille des coefficients** (régularisation L2), dont la force dépend
de `C` : plus `C` est petit, plus les coefficients sont freinés, et moins le modèle risque le surapprentissage.

### 5.4 Choisir C sans tricher

On ne sait pas d'avance quel `C` est le meilleur : on essaie 0,1, 1 et 10.

1. On coupe le **train** en deux (sous-train 80 % / validation 20 %, avec le même découpage par groupes).
2. Pour chaque valeur de C, on entraîne sur le sous-train et on mesure le macro F1 sur la validation.
3. On garde le C au meilleur macro F1 (en cas d'égalité, celui qui régularise le plus).
4. On ré-entraîne sur **tout** le train avec ce C.

Le jeu de test n'intervient jamais dans ce choix : il reste un examen final honnête.

### 5.5 Cas particuliers gérés

- Une seule catégorie présente à l'entraînement : le modèle renvoie cette catégorie avec un score de 1.
- Catégories rares : `class_weight="balanced"` leur donne plus de poids.
- Catégorie vue seulement en test : signalée dans le rapport, jamais ignorée en silence.

---

## 6. Mesurer la qualité

Le test est évalué **une seule fois** par modèle.

### 6.1 La baseline

Elle répond toujours la catégorie la plus fréquente. Si 30 % des ETF sont « Equities – Large Cap », la baseline a
30 % d'accuracy sans rien comprendre. Notre modèle doit faire nettement mieux.

### 6.2 Accuracy, précision, rappel, F1 : un exemple

Sur 100 ETF de test, la catégorie « Gold » compte 10 vrais ETF Gold. Le modèle en annonce 8 « Gold » : 6 le sont
vraiment, 2 non.

- **Précision** = 6 / 8 = **0,75** (quand il dit Gold, il a raison 3 fois sur 4).
- **Rappel** = 6 / 10 = **0,60** (il retrouve 6 vrais Gold sur 10).
- **F1** = 2 × 0,75 × 0,60 / (0,75 + 0,60) ≈ **0,67**.

Le F1 pénalise un déséquilibre : un modèle qui annoncerait « Gold » pour tout aurait un rappel de 1 mais une
précision très basse, donc un mauvais F1.

### 6.3 Pourquoi le macro F1 plutôt que l'accuracy seule ?

Avec 20 catégories dont 3 énormes et 17 minuscules, l'accuracy peut être excellente alors que les 17 petites sont
mal classées. Le **macro F1** fait la moyenne simple des F1 par catégorie : une petite catégorie ratée fait
baisser le score autant qu'une grosse. C'est pour cela qu'il est la métrique principale.

### 6.4 Le rapport d'évaluation

Il contient : macro F1 et accuracy du modèle contre la baseline ; précision, rappel, F1 et support par catégorie ;
les 10 confusions les plus fréquentes ; les features les plus influentes par catégorie (les plus gros
coefficients positifs) ; les catégories impossibles à apprendre.

### 6.5 Les expériences comparables

`config/settings.yaml` définit des expériences qui activent différents blocs (`local`, `text_only`,
`holdings_only`, `embeddings_only`…). Elles mesurent ce que chaque bloc apporte. Sur les données d'exemple :
texte seul → macro F1 0,77 ; composition seule → 0,87 ; tout → 0,94. Les blocs sont complémentaires.

---

## 7. Réentraînement quotidien et promotion

Le monde change : de nouveaux ETF sont validés chaque jour. Un job quotidien :

1. calcule une **empreinte** des données (un code court qui change si quoi que ce soit change) ; si rien n'a
   changé, il ne fait rien ;
2. sinon entraîne un **modèle candidat** ;
3. compare le candidat au modèle courant, puis le **promeut** ou le refuse.

### 7.1 Une comparaison équitable

Les deux modèles sont notés sur les **mêmes ETF de test**, en excluant ceux sur lesquels le modèle courant s'est
entraîné (la liste des tickers est sauvée dans `train_tickers.json`) et en se limitant aux catégories que les deux
connaissent. Sinon, l'ancien modèle serait avantagé sur des exemples qu'il a déjà vus.

### 7.2 La règle de promotion

Le candidat remplace le modèle courant seulement si **les quatre conditions** sont vraies :

| # | Condition | Pourquoi |
|---|---|---|
| 1 | Au moins 20 ETF dans le jeu de comparaison | Sinon la mesure est trop bruitée pour décider |
| 2 | Macro F1 candidat ≥ macro F1 courant − 0,01 | Il ne doit pas être moins bon (tolérance de 0,01 pour le bruit) |
| 3 | Aucune catégorie (≥ 5 exemples de test) ne perd plus de 0,20 de F1 | Éviter qu'un bon score moyen cache l'effondrement d'une catégorie |
| 4 | Le candidat bat la baseline sur son propre test | Vérification de bon sens |

Chaque condition est rapportée (réussie ou non) dans l'historique. L'objectif : **un réentraînement automatique
ne doit jamais dégrader silencieusement le modèle.**

---

## 8. L'humain dans la boucle

- L'app propose une catégorie et des alternatives ; l'humain **valide ou corrige**.
- Chaque validation est écrite dans un **journal en ajout seul** (jamais réécrit) avec le ticker, la catégorie
  choisie, la catégorie proposée par le modèle (`PredictedCategory`), l'auteur et la date.
- Au chargement, la dernière validation active d'un ticker **remplace** la ligne du CSV de référence. Les
  corrections humaines deviennent donc des exemples d'entraînement pour le modèle du lendemain : c'est la boucle
  d'amélioration continue.
- Si plus de 95 % des propositions sont acceptées telles quelles (sur au moins 50 validations), une alerte
  signale un **biais d'acceptation possible** : les validateurs cliquent peut-être sans vérifier.
- **Contrôle d'une classification existante** : si le CSV à traiter contient déjà une `Classification`, le modèle
  la compare à la sienne (`agreement`, `disagreement` ou `unknown`). Les désaccords passent en tête de la file de
  validation. Une classification établie n'est jamais écrite dans le journal toute seule : seule une validation
  humaine l'est.

### Couches annexes

- **Similarité** : retrouve les ETF les plus proches (50 % composition, 50 % texte) et fait voter les voisins. Une
  seconde méthode, comparable au modèle, qui sert aussi à repérer les quasi-doublons.
- **Justification par LLM** : un modèle de langage (GPT-6.1 Sol) donne un avis sur la catégorie proposée
  (`fits`, `doubtful`, `does_not_fit`) avec une alternative possible. Il **ne classe pas** et ne remplace pas la
  validation humaine ; il ne reçoit que des données approuvées (nom, description et contexte).

---

## 9. Limites et points de vigilance

- **Scores d'exemple optimistes.** Les données de `data/examples/` ont des compositions fictives avec des
  recouvrements fabriqués : les scores affichés (macro F1 0,94) ne prédisent pas ceux d'un vrai jeu de données.
- **Le modèle est linéaire.** Il combine les features en sommes pondérées ; il ne capte pas des interactions
  complexes (« A *et* B ensemble signifie C »). C'est un choix de simplicité et de lisibilité.
- **Il ne connaît que les catégories vues.** Une catégorie absente de l'entraînement ne sera jamais proposée.
- **Qualité des étiquettes.** Si les classifications de référence sont incohérentes, le modèle apprendra ces
  incohérences. Les désaccords signalés peuvent révéler des erreurs *dans les données*, pas seulement dans le modèle.
- **Les embeddings dépendent d'Azure** : coût, disponibilité, et données envoyées (nom + description seulement).
- **Question ouverte.** Pour un ETF sans description, faut-il injecter les principaux titres dans le texte des
  embeddings (option `include_top_tickers`, désactivée) ? À tester sur de vraies données.

---

## 10. Où est quoi dans le code

| Sujet | Fichier |
|---|---|
| Règles de qualité | `src/etf_classifier/data/validation.py` |
| Groupes de quasi-doublons | `src/etf_classifier/data/` (dedup) |
| Séparation train / test | `src/etf_classifier/data/split.py` |
| Orchestration des features | `src/etf_classifier/features/pipeline.py` |
| TF-IDF du texte | `src/etf_classifier/features/text.py` |
| Titres détenus et statistiques | `src/etf_classifier/features/holdings.py` |
| Embeddings Azure, cache, coût | `src/etf_classifier/features/embeddings.py` |
| Encodage des catégories | `src/etf_classifier/features/labels.py` |
| Modèle, réglage de C | `src/etf_classifier/model/classifier.py` |
| Baseline | `src/etf_classifier/model/baseline.py` |
| Métriques et rapport | `src/etf_classifier/model/evaluate.py` |
| Vérification d'une classification établie | `src/etf_classifier/model/check.py` |
| Explication (features influentes) | `src/etf_classifier/model/explain.py` |
| Entraînement / prédiction (ligne de commande) | `src/etf_classifier/model/train.py`, `predict.py` |
| Versions de modèles | `src/etf_classifier/model/registry.py` |
| Comparaison et règle de promotion | `src/etf_classifier/retraining/compare.py` (+ `model/compare.py`) |
| Job quotidien | `src/etf_classifier/retraining/run.py` |
| Similarité et vote des voisins | `src/etf_classifier/similarity/` |
| Réglages | `config/settings.yaml` |

Commandes utiles :

```bash
python -m etf_classifier.data.report --csv data/raw/etfs.csv     # contrôle qualité
python -m etf_classifier.model.train --csv data/raw/etfs.csv --no-azure   # entraînement sans embeddings
python -m etf_classifier.retraining.run --force                  # réentraînement + règle de promotion
```
