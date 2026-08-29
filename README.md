# Threadline — Local AI Shopping Copilot

Threadline is a fully local, multi-turn shopping agent built for TikTok TechJam 2026 Track 4. It searches the supplied 50,000-product catalog, remembers preferences across turns, detects when a shopper changes their mind, and asks useful clarification questions while continuing to recommend products.

The main idea is to keep simple searches fast and only use the more advanced retrieval steps when they are actually needed. Clear requests go through a field-weighted keyword ranker, while vague requests can use local query expansion and candidate-aware follow-up questions. No API key, network connection, paid model, or evaluator-owned credential is required.

## Why this is different

Most shopping assistants either search once or put a chatbot in front of keyword search. Threadline treats the conversation itself as a changing retrieval problem:

- Buying and browsing requests follow different state paths.
- Later answers refine the original request instead of replacing it.
- Explicit intent changes clear stale preferences and safely reconsider products.
- Recommendations appear on every turn; clarification does not block discovery.
- Questions can adapt to the live candidate set using entropy and coverage.
- Products already shown are not repeated unless the intent genuinely changes.

## Measured public-set result

Run on the organizer-provided 200-session public development set:

| Metric | Starter baseline | Threadline local |
|---|---:|---:|
| TechnicalScore | 0.106710 | **0.789582** |
| Hit Rate@10 | 0.125 | **0.925** |
| MRR | 0.068034 | **0.589605** |
| MTTC | 9.81 | **3.49** |
| Model tokens | — | **0** |

The detailed run is stored in `results.json`. Public-set tuning can overfit, so this score is evidence of progress rather than a private-set guarantee.

## Architecture

```text
Customer turn
    |
    v
Local intent tracker -----> route: buying / browsing
    |                       slots, budget, exclusions, override
    v
Confidence-gated retrieval
    |-- lexical anchor: weighted SQLite FTS5 BM25
    `-- low-recall recovery: query expansion + pseudo-relevance feedback
    |
    v
Candidate-grounded clarification
    |-- reliable high-yield opening sequence
    `-- entropy × coverage × answerability for uncertain shoppers
    |
    v
Catalog-valid Top 10 + isolated session memory
```

The AI part runs locally. It tracks intent from natural language, chooses a search route, expands weak queries, and selects useful follow-up questions from the current candidates. It is not a generative model, which keeps the system repeatable, private, free to run, and suitable for a network-restricted scoring environment.

## Judging criteria

| Criterion | Evidence |
|---|---|
| Technical Execution (35%) | Modular intent, dialogue, and retrieval layers; FTS5 index; route fusion; bounded candidate work; isolated state; automated tests; measured evaluation |
| Innovation & Problem Insight (20%) | Confidence-gated semantic recovery and candidate-aware clarification address uncertainty without paid inference |
| Impact & Relevance (20%) | Supports exploration, follow-up answers, corrections, budgets, exclusions, and non-repeating results |
| Feasibility & Practicality (15%) | Standard-library-only runtime, zero credentials, zero model tokens, frozen-catalog grounding, and roughly 20-second public evaluation |
| Presentation & Communication (10%) | Reproducible metrics, clear architecture, honest limitations, documented decisions, and a demo-ready flow |

## Repository structure

```text
starter/agent.py              multi-turn orchestration and session isolation
starter/intent.py             local intent, constraint, route, and override tracking
starter/retrieval.py          catalog index, lexical anchor, and semantic recovery
starter/dialogue.py           information-gain clarification policy
evaluator/local_evaluator.py  organizer-provided evaluation harness
tests/                        behavior and evaluator contract tests
docs/                         supplied rules, specification, and baseline evidence
results.json                  latest reproducible public evaluation
```

## Setup and installation

Requirements:

- Python 3.10 or newer
- SQLite with FTS5 (included with normal Python installations)
- The supplied `data/catalog.jsonl`

No package installation is needed. The runtime uses only the Python standard library. If the catalog is missing, verify and extract the participant-kit download, then place it at `data/catalog.jsonl` as described in `data/README.md`.

## Reproduce the results

From the repository root, run:

```bash
python3 -m unittest discover -v
python3 -m evaluator.local_evaluator \
  --dataset data/public_set.jsonl \
  --catalog data/catalog.jsonl \
  --output results.json
```

Expected test result:

```text
Ran 12 tests
OK
```

The evaluator prints the summary and writes session-level evidence to `results.json`.

## Important implementation choices

### Grounded recommendations

The agent never invents product identifiers. Every returned `parent_asin` comes from the local catalog index.

### Confidence-gated retrieval

The weighted BM25 anchor is fast and reliable for specific product language. More expensive multi-route expansion only runs when the first pass cannot fill the result slate. This provides semantic recovery without slowing every ordinary query.

### Adaptive clarification

Early questions use attributes the public customer simulator can answer reliably. If the shopper remains uncertain, the policy measures how well each attribute separates the current candidates. It balances normalized entropy, catalog coverage, and expected answerability.

### Dynamic intent

Each session has isolated memory. The tracker records category, preference slots, exclusions, budget, route, and intent changes. A confirmed override clears stale state and allows previously seen products to be reconsidered.

## Testing

Tests cover session isolation, routing, follow-up slot updates, intent overrides, simultaneous recommendation and clarification, result diversity, reconsideration after an override, constraint tracking, zero external model usage, and evaluator scoring.

## Limitations and reflection

- The semantic layer is lightweight distributional retrieval, not a neural embedding model. It can miss wording gaps that a trained encoder would recognize.
- Catalog metadata is incomplete. Hard filtering every inferred constraint can remove a relevant product, so the lexical path is recall-first while constraints guide state and recovery.
- The parser targets the challenge's clean English turns; slang, multilingual input, and spelling errors need broader normalization.
- Price parsing handles common numeric formats but not currency conversion.
- The 200 public sessions are too small to prove generalization to the private set.
- There is no graphical interface because the challenge evaluates a headless API; a polished demo UI would improve the final presentation.

Given more time, I would add a locally bundled embedding model, train a small reranker on a held-out split, calibrate confidence, run systematic ablations, and build an interactive product-card demo with visible intent changes and clarification rationale.

## External services and privacy

Threadline requires **no external service**, API key, live credential, or network access. Evaluation is entirely local. Publishing this repository cannot expose or consume paid credits because none are used.

## Team contribution

This is a solo submission done by Er Teng Sheng Elgin.

## Data attribution

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). The starter kit, evaluator, and competition specification were provided by TechJam 2026.
