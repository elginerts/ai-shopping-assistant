# Threadline — Local AI Shopping Copilot

Threadline is a multi-turn shopping agent built for TikTok TechJam 2026 Track 4. It searches the supplied catalogue of 50,000 products, remembers what a shopper has said, asks useful follow-up questions, and adjusts when the shopper changes their mind.

The project uses the open-source `nomic-embed-text` model through Ollama. Runtime search stays local, so there are no API keys, paid credits, or external requests after setup.

## The problem I wanted to solve

Shopping search is rarely one perfect sentence. A shopper might begin with “I need running shoes,” add a material preference later, reject a colour, or replace an earlier requirement completely. A useful assistant needs to search while that conversation is still developing.

Threadline handles four parts of that problem:

- It separates specific Buying requests from open-ended Browsing requests.
- It combines exact keyword evidence with neural meaning similarity.
- It asks questions that help separate the current product candidates.
- It removes stale preferences when the shopper changes their intent.

Recommendations are returned on every turn. The shopper does not have to finish a long questionnaire before seeing products.

## What makes Threadline different

The main idea is a **correction-aware search and decision engine**.

Ollama embeddings are useful when two phrases mean the same thing but use different words. However, embedding raw corrections can preserve stale meaning, such as “blue” inside “not blue anymore.” Threadline solves this by compiling a fresh semantic query from only the active intent-ledger revisions. Retired preferences never reach the reranker again.

Threadline also keeps a versioned intent ledger. Preferences are not stored as one block of chat text: every addition, replacement, and removal has a status and source turn. Before asking a follow-up question, the agent simulates the possible candidate groups for each attribute and estimates the expected candidate reduction and Top-10 confidence gain. The trace makes that decision visible instead of treating the model as a black box.

Other design choices include:

- A field-weighted SQLite FTS5 index for fast catalogue retrieval
- An independent full-catalogue Nomic entrance for meaning-based discovery
- A margin gate that protects the lexical ranking from weak semantic challengers
- Neural reranking over only the top 16 candidates instead of all 50,000 products
- Reciprocal-rank fusion so BM25 and model scores can be combined safely
- A disk cache so product embeddings are generated once and reused
- A counterfactual question planner with a public-set-validated safety policy
- Open constraint capture before slot-specific questions, reducing wrong-question paths
- Confidence-gated Top-10 reranking for exact multi-constraint matches
- A reproducible NumPy pairwise ranker for conservative dense-candidate experiments
- A versioned intent ledger that preserves unrelated preferences during corrections
- An optional decision trace explaining state, retrieval strategy, and question value
- Separate memory and recommendation history for every shopper session
- Catalogue grounding, which means the model never invents product IDs

## Last verified public result

These measurements are the deployed default. The opening question captures requirements without guessing a slot, while learned dense promotion remains experimental because its measured score was lower.

| Metric | Starter baseline | Previous verified | Current default |
|---|---:|---:|---:|
| TechnicalScore | 0.106710 | 0.793614 | **0.809642** |
| Hit Rate@10 | 0.125 | 0.925 | **0.940** |
| MRR | 0.068034 | 0.602381 | **0.609472** |
| MTTC | 9.81 | 3.48 | **3.16** |
| Reported generative tokens | — | 0 | **0** |

Scenario results:

| Scenario | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Boundary | 0.8000 | 0.549286 | 4.70 |
| Browsing | 1.0000 | 0.609196 | 2.65 |
| Buying | 0.9625 | 0.656121 | 2.475 |
| Intent Override | 0.7667 | 0.505873 | 5.8333 |

The first full run on the development Mac took about 4 minutes 22 seconds while building a 21 MB embedding cache. The final warm-cache verification took about 40 seconds. Hardware will affect these timings.

Public-set tuning can overfit, so these numbers are not a promise about the private set. The ablations and rejected settings are documented in [docs/ollama_ablation.md](docs/ollama_ablation.md).

Dense-retrieval ablation:

| Mode | TechnicalScore | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Verified default | **0.793614** | 0.925 | **0.602381** | 3.480 |
| Dense promotion every turn | 0.786133 | 0.925 | 0.575776 | **3.455** |
| Dense promotion on revisions | 0.792364 | 0.925 | 0.597881 | 3.475 |
| Learned gate + open capture | 0.798071 | 0.940 | 0.568903 | **3.130** |
| Open capture, dense gate off | **0.798992** | **0.940** | **0.573972** | 3.160 |
| Structured Top-10 reranker | **0.809642** | **0.940** | **0.609472** | **3.160** |

## Architecture

```text
Customer message
       |
       v
Versioned intent ledger + clean query compiler
       |
       +-------------------------+
       |                         |
       v                         v
Field-weighted BM25       Full-catalogue Nomic search
exact-word entrance       meaning-based entrance
       |                         |
       +------------+------------+
                    v
       Margin-gated challenger promotion
       protect lexical head, replace rank 10
                    |
                    v
          Grounded Top 10 products
                    |
                    v
       Confidence-gated constraint reranker
       reorder only; never admit new products
                       |
                       v
          Counterfactual question planner
     simulate candidate reduction + Top-10 gain
```

The model does not generate products. Both entrances return IDs from the frozen catalogue, and the promotion gate can only reorder those grounded IDs.

More detail is available in [docs/architecture.md](docs/architecture.md).

## How this matches the judging criteria

| Criterion | Evidence in this repository |
|---|---|
| Technical Execution (35%) | Separate intent, retrieval, reranking, dense-index, model-client, dialogue, promotion, and diagnostic components; checksum validation; 29 automated tests |
| Innovation & Problem Insight (20%) | A versioned intent ledger and counterfactual question simulation address stale preferences and unnecessary questions; correction-aware query compilation prevents semantic prompt inertia |
| Impact & Relevance (20%) | Handles browsing, specific buying, follow-up answers, uncertainty, non-repeating results, and changed requirements |
| Feasibility & Practicality (15%) | Local Apache-2.0 model, NumPy in-memory search, downloadable prebuilt index, resumable builder, no paid calls, and catalogue-grounded output |
| Presentation & Communication (10%) | Reproducible commands, architecture notes, ablation evidence, honest limitations, and readable project structure |

## Repository structure

```text
starter/agent.py                 conversation flow and session memory
starter/intent.py                versioned preference ledger and active intent state
starter/retrieval.py             typed retrieval pipeline, BM25, reranking, and rank fusion
starter/dense_index.py           portable NumPy index, verification, and dense search
starter/ollama_embeddings.py     Ollama client and persistent embedding cache
starter/promotion.py             portable pairwise model and feature schema
starter/structured_reranker.py   exact-constraint Top-10 ordering
scripts/build_dense_index.py     resumable full-catalogue index builder
scripts/download_dense_index.py  release-asset download and verification
scripts/train_promotion_model.py reproducible public-set ranker training
scripts/verify_dense_index.py    standalone compatibility check
starter/dialogue.py              counterfactual question simulation and selection
evaluator/local_evaluator.py     organizer-provided evaluation harness
tests/                           behaviour, model-client, cache, and evaluator tests
docs/architecture.md             design details and data flow
docs/ollama_ablation.md          measured experiments and decisions
docs/decision_engine.md          ledger, planner formula, and trace format
```

## Setup and installation

You need:

- Python 3.10 or newer
- NumPy 2.x
- SQLite with FTS5, which is included with normal Python installations
- [Ollama](https://ollama.com/download)
- The supplied `data/catalog.jsonl`
- About 274 MB for `nomic-embed-text` and about 160 MB for the release index

Install the Python dependency:

```bash
python3 -m pip install -r requirements.txt
```

On macOS with Homebrew:

```bash
brew install ollama
brew services start ollama
ollama pull nomic-embed-text
```

On another operating system, install Ollama using its official instructions, start the service, and then run:

```bash
ollama pull nomic-embed-text
```

Confirm that the model is ready:

```bash
ollama list
```

The output should include `nomic-embed-text`.

If the catalogue is missing, verify and extract the participant-kit download, then place it at `data/catalog.jsonl` as described in [data/README.md](data/README.md).

## Run the project

Start Ollama if it is not already running:

```bash
ollama serve
```

The verified default does not require the full dense index. To reproduce the experimental dense mode, download the prebuilt release index and verify it:

```bash
python3 -m scripts.download_dense_index --url <release-asset-url>
python3 -m scripts.verify_dense_index
```

To reproduce the artifact locally instead, run the separate builder. It resumes from the SQLite embedding cache if interrupted:

```bash
python3 -m scripts.build_dense_index
python3 -m scripts.verify_dense_index
```

The learned-promotion ablation can also be rebuilt and evaluated separately:

```bash
python3 -m scripts.train_promotion_model
THREADLINE_DENSE_MODE=learned python3 -m evaluator.local_evaluator
```

The measured build time is about 53 minutes from an empty cache, or about 44 minutes with the development cache. Building is preparation work and is not performed by the evaluator.

Run the tests:

```bash
python3 -m unittest discover -v
```

Expected result:

```text
Ran 29 tests
OK
```

Run the public evaluator:

```bash
python3 -m evaluator.local_evaluator \
  --dataset data/public_set.jsonl \
  --catalog data/catalog.jsonl \
  --output results.json \
  --diagnostics-output results.diagnostics.json
```

The diagnostic report records every failed session and shows whether the target was absent from BM25, pushed below the Top 10 by Nomic, left below the final cutoff, shown before an override, or affected by a question that could not reveal a remaining constraint. Stage ranks are collected without passing the target ID into the agent, so the audit cannot influence recommendations.

The builder creates `.threadline_cache/product_embeddings.sqlite3` and `.threadline_cache/dense_index.npz`. The cache, model, catalogue, dense index, and evaluation output are intentionally not committed to Git. The release index is required only when dense challenger mode is enabled.

Threadline requires Ollama and `nomic-embed-text`. There is no non-model fallback. If either is missing, startup stops with a clear command showing how to fix the setup.

## Configuration

The measured defaults should normally be kept unchanged.

| Variable | Default | Purpose |
|---|---:|---|
| `THREADLINE_SEMANTIC_WEIGHT` | `0.18` | Influence of semantic rank during reciprocal-rank fusion |
| `THREADLINE_RERANK_LIMIT` | `16` | Number of BM25 candidates sent to the model |
| `THREADLINE_DENSE_MODE` | `off` | `challenger` runs the hand-built gate; `learned` loads the pairwise ranker |
| `THREADLINE_DENSE_INDEX` | `.threadline_cache/dense_index.npz` | Required portable dense-index path |
| `THREADLINE_PROMOTION_MARGIN` | `0.03` | Minimum evidence advantage before a dense challenger replaces rank 10 |
| `THREADLINE_CORRECTION_SEMANTIC` | `clean` | `clean` reranks a query compiled from active ledger slots; `lexical` is the ablation mode |
| `THREADLINE_QUESTION_POLICY` | `guarded` | `guarded` keeps the verified policy; `counterfactual` enables the experimental unrestricted planner |

## Testing

The 29 tests cover:

- Session isolation and non-repeating recommendations
- Buying and Browsing routing
- Follow-up preference updates
- Intent correction and stale-preference removal
- Selective slot replacement and attribute removal
- Versioned ledger state and public decision traces
- Counterfactual question-value metrics
- Clean-intent semantic reranking after corrections
- Recommendations returned while asking a question
- Budget and exclusion tracking
- Required-model error messages
- Embedding normalization and cache round trips
- Dense-index round trips, semantic search, and catalogue mismatch rejection
- Pairwise-ranker training and portable model round trips
- Structured-reranker promotion, weak-evidence stability, and exclusion safety
- Verified-default startup without a full dense index
- Evaluator response normalization and scoring
- Per-stage failure-diagnostic classification

Tests use a small deterministic embedder so unit tests stay quick. The reported public score was produced with the real Ollama model.

## Limitations and reflection

- Ollama and `nomic-embed-text` must be installed before the program starts.
- The 274 MB model cannot fit inside the submission form's 35 MB file-upload limit, so judges must download it during setup or already have it installed.
- The experimental dense index is distributed separately because generated vectors are too large for normal source control.
- Dense setup assumes evaluators can download the project release asset. The organizer information sheet permits dense retrieval but does not explicitly guarantee release-asset access.
- The learned dense gate scored 0.798071 versus 0.798992 for the simpler default. It remains a reproducible experiment rather than weakening deployment.
- Semantic reranking improved ranking quality but did not improve Hit Rate@10 on the public set.
- Open constraint capture reduced wrong-question paths from nine failed sessions to three. Slot-specific counterfactual questions handle later turns.
- Structured reranking relies on explicit catalogue text. Weak or incomplete metadata keeps the original ranking unchanged.
- Intent Override and Boundary sessions remain the weakest scenarios.
- The intent parser targets the challenge's clean English turns and needs more work for multilingual queries, slang, and spelling errors.
- Catalogue metadata is incomplete, so aggressive hard filtering can remove useful products.
- There is no graphical interface because the supplied challenge evaluates a headless agent.

With more time, I would calibrate the challenger margin on a separate validation split, add multilingual intent tests, compress the index, and build a product-card demo that shows why a semantic challenger was promoted.

## External service and model disclosure

Ollama runs as a required service on `localhost`. After Ollama and the model have been downloaded, Threadline does not need internet access, live credentials, or paid credits. The model is not stored in this GitHub repository.

`nomic-embed-text` is distributed under the Apache License 2.0. See the [Ollama model page](https://ollama.com/library/nomic-embed-text) for model details.

## Team contribution

This is a solo submission done by Er Teng Sheng Elgin.

## Data attribution

The catalogue and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). The starter kit, evaluator, and competition specification were provided by TechJam 2026.
