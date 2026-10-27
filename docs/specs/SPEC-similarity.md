# Spec: `similarity` module

Status: **approved** (2026-10-25)

> **Amended on 2026-10-27** by [CHANGE-input-format](CHANGE-input-format.md): input CSV `Ticker;Nom;Description;Classification;Composition`, one category per ETF, ticker as the ETF identifier. Where this spec and the change spec differ, the change spec prevails.

## Objective

For an ETF (new or existing), find the **closest already-categorised ETFs** and explain
why they look alike. Three uses:
1. **Help the validator** in the interface: "this ETF looks like X and Y, categorised Gold and Commodities".
2. **Give context to the LLM** (`llm-justify`) to justify a proposal or handle an uncertain case.
3. **Report a near-duplicate**: a new ETF that tracks the same index as an existing ETF very probably has the same categories.

It is also a second categorisation method, by **neighbour vote**, compared with the model
for learning purposes (two approaches, same data, same test set).

Out of scope: combining the neighbour vote with the model for the final prediction (to be decided after
measurement), the interface (`app` module).

## Input data
- **Reference set**: the categorised ETFs of the CSV (usable rows from the `data` module).
- **Feature pipeline** of the current model (`features` module): ticker vectors, TF-IDF, embeddings if active.
- The new ETFs to compare, with or without a category.

## Features

### 1. Two similarities, then a combined score

| Similarity | Computation | Why |
|---|---|---|
| **Composition** | Cosine between the ticker weight vectors | Symmetric: a small 3-holding ETF contained in a large ETF is not judged identical to it, unlike the overlap "relative to the smallest coverage" |
| **Text** | Cosine between the embeddings if the model uses them, otherwise between the TF-IDF vectors | Brings together ETFs with the same theme, even without common securities |

- **Combined score** = 0.5 × composition + 0.5 × text (configurable weights).
- All neighbours of the same search are scored on the same scale:
  - if the **searched ETF** has no composition, all neighbours are scored on text only, and this is indicated;
  - if only the **neighbour** has no composition, its composition similarity counts as 0 in the combined score.
  *(Correction of 2026-10-25: the initial rule, "text only if either of the two has no composition", ranked
  an ETF without composition ahead of ETFs sharing the same tickers, because its score was not on the same scale.)*

### 2. Neighbour search
- `SimilarityIndex.build(reference_df, pipeline)` builds the index; `index.query(df, k=5)` returns the `k` neighbours of every ETF.
- For every neighbour:
  - name and categories;
  - combined score, composition similarity, text similarity;
  - **share of the weight in common** (weighted overlap, in %) and the 5 main shared tickers, to explain the match in a readable way.
- An ETF already present in the reference is never its own neighbour.
- **Near-duplicate alert**: same rule as the `data` module (overlap ≥ 0.80 and at least 3 common tickers).

### 3. Neighbour vote
- `neighbor_vote(neighbors)` gives a score for every category: sum of the scores of the neighbours that have it, divided by the sum of all scores.
- Evaluation: `python -m etf_classifier.similarity.evaluate --csv …` builds the index **on training only**, has the neighbours vote for every test ETF, and compares with the current model and the trivial baseline (macro and micro F1, for k = 5, 10 and 20). This tells us whether the vote helps, especially on rare categories.

### 4. Building, saving, command
- The index is built by `model.train` on all usable rows (training + test), then saved **in the model folder** (`similarity.joblib`). The model and its index always stay consistent.
- `python -m etf_classifier.similarity.search --csv new_etfs.csv [--model current] [--k 5]` writes a CSV of neighbours (one ETF × one neighbour per line) and a Markdown summary.
- Embeddings of new ETFs go through the cache and the cost confirmation of the `features` module.

## Tech Stack
No new dependency (numpy, scipy, scikit-learn).

## Structure
```
src/etf_classifier/similarity/
├── index.py       # SimilarityIndex: building, search, explanation, saving
├── vote.py        # neighbour vote
├── search.py      # search CLI
└── evaluate.py    # evaluation CLI (vote vs model vs trivial baseline)
tests/similarity/
```
Change to an existing module: `model.train` builds and saves the index in the model folder; `model.registry` reloads it.

## Test strategy
- Two ETFs with identical compositions: composition similarity equal to 1 and near-duplicate alert.
- A small ETF contained in a large one: composition similarity clearly below 1, whereas the overlap is 100 %.
- An existing ETF is never its own neighbour; neighbours are sorted by decreasing score.
- Empty composition: only the text counts, with a mention.
- Neighbour vote on the synthetic dataset: clearly better than the trivial baseline.
- Saving then reloading: same neighbours, same scores.
- Performance: 100 queries on a reference of 10,000 ETFs in less than 10 s.

## Boundaries
- **Always:** explain every neighbour (share of the weight in common, shared tickers); exclude the ETF itself; evaluate the vote with an index built on training only.
- **Ask first:** using the vote to change the model's predictions; changing the similarity measure.
- **Never:** send Azure anything other than the name and the description.

## Success criteria
1. For an ETF of the fixture, the proposed neighbours have consistent categories, with a readable explanation.
2. A near-duplicate of an existing ETF is reported.
3. The evaluation report quantifies the neighbour vote against the model and the trivial baseline.
4. Tests pass without network; performance targets are met.

## Decisions made
- Combined score weights: **0.5 composition / 0.5 text** to start with, to be adjusted after measuring the vote on the test set.
- **5 neighbours** displayed by default.
- The neighbour vote is first **limited to an evaluation**; its use to complement the model (uncertain cases) will be decided after measurement.
