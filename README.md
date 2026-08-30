# Threadline — Local AI Shopping Copilot

Threadline is a multi-turn shopping agent built for TikTok TechJam 2026 Track 4. It searches the supplied catalogue of 50,000 products, remembers what a shopper has said, asks useful follow-up questions, and adjusts when the shopper changes their mind.

The project uses the open-source `nomic-embed-text` model through Ollama. Everything runs locally, so there are no API keys, paid credits, or external requests during evaluation.

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

Ollama embeddings are useful when two phrases mean the same thing but use different words. However, embedding models can be less reliable around negation, such as “not blue anymore.” Threadline therefore uses semantic reranking during normal discovery and switches to the cleaned lexical state after an explicit correction. This avoids carrying the meaning of an old preference into the new search.

Threadline also keeps a versioned intent ledger. Preferences are not stored as one block of chat text: every addition, replacement, and removal has a status and source turn. Before asking a follow-up question, the agent simulates the possible candidate groups for each attribute and estimates the expected candidate reduction and Top-10 confidence gain. The trace makes that decision visible instead of treating the model as a black box.

Other design choices include:

- A field-weighted SQLite FTS5 index for fast catalogue retrieval
- Neural reranking over only the top 16 candidates instead of all 50,000 products
- Reciprocal-rank fusion so BM25 and model scores can be combined safely
- A disk cache so product embeddings are generated once and reused
- A counterfactual question planner with a public-set-validated safety policy
- A versioned intent ledger that preserves unrelated preferences during corrections
- An optional decision trace explaining state, retrieval strategy, and question value
- Separate memory and recommendation history for every shopper session
- Catalogue grounding, which means the model never invents product IDs

## Verified public result

The latest result uses the organizer-provided 200-session public development set.

| Metric | Starter baseline | Before Ollama | Threadline + Ollama |
|---|---:|---:|---:|
| TechnicalScore | 0.106710 | 0.789582 | **0.792766** |
| Hit Rate@10 | 0.125 | 0.925 | **0.925** |
| MRR | 0.068034 | 0.589605 | **0.600220** |
| MTTC | 9.81 | 3.49 | **3.49** |
| Reported generative tokens | — | 0 | **0** |

Scenario results:

| Scenario | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Boundary | 0.6000 | 0.533333 | 6.50 |
| Browsing | 0.9875 | 0.596637 | 2.8875 |
| Buying | 0.9500 | 0.625342 | 2.75 |
| Intent Override | 0.8000 | 0.565079 | 6.0667 |

The first full run on the development Mac took about 4 minutes 22 seconds while building a 21 MB embedding cache. The final warm-cache verification took about 40 seconds. Hardware will affect these timings.

Public-set tuning can overfit, so these numbers are not a promise about the private set. The ablations and rejected settings are documented in [docs/ollama_ablation.md](docs/ollama_ablation.md).

## Architecture

```text
Customer message
       |
       v
Versioned intent ledger
  active + replaced + removed preferences
       |
       v
Field-weighted BM25 candidate search
       |
       +------------------------------+
       | normal request               | corrected request
       v                              v
Ollama semantic reranker         Clean lexical ranking
nomic-embed-text                 correction safety gate
       |                              |
       +---------------+--------------+
                       v
             Grounded Top 10 products
                       |
                       v
          Counterfactual question planner
     simulate candidate reduction + Top-10 gain
```

The model does not generate recommendations directly. It only compares the meaning of a customer request with real catalogue products returned by BM25.

More detail is available in [docs/architecture.md](docs/architecture.md).

## How this matches the judging criteria

| Criterion | Evidence in this repository |
|---|---|
| Technical Execution (35%) | Separate intent, retrieval, model-client, dialogue, and cache components; bounded neural reranking; clear startup errors; 19 automated tests; measured evaluation |
| Innovation & Problem Insight (20%) | A versioned intent ledger and counterfactual question simulation address stale preferences and unnecessary questions; correction-aware semantic gating handles embedding weakness around negation |
| Impact & Relevance (20%) | Handles browsing, specific buying, follow-up answers, uncertainty, non-repeating results, and changed requirements |
| Feasibility & Practicality (15%) | Local Apache-2.0 model, no paid calls, reusable embedding cache, small candidate window, and catalogue-grounded output |
| Presentation & Communication (10%) | Reproducible commands, architecture notes, ablation evidence, honest limitations, and readable project structure |

## Repository structure

```text
starter/agent.py                 conversation flow and session memory
starter/intent.py                versioned preference ledger and active intent state
starter/retrieval.py             BM25 search, semantic reranking, and rank fusion
starter/ollama_embeddings.py     Ollama client and persistent embedding cache
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
- SQLite with FTS5, which is included with normal Python installations
- [Ollama](https://ollama.com/download)
- The supplied `data/catalog.jsonl`
- About 274 MB for `nomic-embed-text`, plus space for the generated cache

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

The output should include `nomic-embed-text`. No Python packages need to be installed.

If the catalogue is missing, verify and extract the participant-kit download, then place it at `data/catalog.jsonl` as described in [data/README.md](data/README.md).

## Run the project

Start Ollama if it is not already running:

```bash
ollama serve
```

Run the tests:

```bash
python3 -m unittest discover -v
```

Expected result:

```text
Ran 19 tests
OK
```

Run the public evaluator:

```bash
python3 -m evaluator.local_evaluator \
  --dataset data/public_set.jsonl \
  --catalog data/catalog.jsonl \
  --output results.json
```

The first run creates `.threadline_cache/product_embeddings.sqlite3`. Later runs reuse it. The cache, model, catalogue, and evaluation output are intentionally not committed to GitHub.

Threadline requires Ollama and `nomic-embed-text`. There is no non-model fallback. If either is missing, startup stops with a clear command showing how to fix the setup.

## Configuration

The measured defaults should normally be kept unchanged.

| Variable | Default | Purpose |
|---|---:|---|
| `THREADLINE_SEMANTIC_WEIGHT` | `0.18` | Influence of semantic rank during reciprocal-rank fusion |
| `THREADLINE_RERANK_LIMIT` | `16` | Number of BM25 candidates sent to the model |
| `THREADLINE_CACHE_DIR` | `.threadline_cache` | Location of the reusable product embedding cache |
| `THREADLINE_QUESTION_POLICY` | `guarded` | `guarded` keeps the verified policy; `counterfactual` enables the experimental unrestricted planner |

## Testing

The 19 tests cover:

- Session isolation and non-repeating recommendations
- Buying and Browsing routing
- Follow-up preference updates
- Intent correction and stale-preference removal
- Selective slot replacement and attribute removal
- Versioned ledger state and public decision traces
- Counterfactual question-value metrics
- The correction-aware semantic safety gate
- Recommendations returned while asking a question
- Budget and exclusion tracking
- Required-model error messages
- Embedding normalization and cache round trips
- Evaluator response normalization and scoring

Tests use a small deterministic embedder so unit tests stay quick. The reported public score was produced with the real Ollama model.

## Limitations and reflection

- Ollama and `nomic-embed-text` must be installed before the program starts.
- The 274 MB model cannot fit inside the submission form's 35 MB file-upload limit, so judges must download it during setup or already have it installed.
- The first evaluation is slower while the product cache is populated.
- Semantic reranking improved ranking quality but did not improve Hit Rate@10 on the public set.
- Unrestricted counterfactual question selection improved Boundary recall but reduced the overall score, so the verified default uses it behind an answerability guardrail.
- Intent Override and Boundary sessions remain the weakest scenarios.
- The intent parser targets the challenge's clean English turns and needs more work for multilingual queries, slang, and spelling errors.
- Catalogue metadata is incomplete, so aggressive hard filtering can remove useful products.
- There is no graphical interface because the supplied challenge evaluates a headless agent.

With more time, I would train a small shopping-specific reranker on a separate validation split, add multilingual intent tests, compress the embedding cache, and build a product-card demo that shows how the live intent and ranking change after every message.

## External service and model disclosure

Ollama runs as a required service on `localhost`. After Ollama and the model have been downloaded, Threadline does not need internet access, live credentials, or paid credits. The model is not stored in this GitHub repository.

`nomic-embed-text` is distributed under the Apache License 2.0. See the [Ollama model page](https://ollama.com/library/nomic-embed-text) for model details.

## Team contribution

This is a solo submission done by Er Teng Sheng Elgin.

## Data attribution

The catalogue and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). The starter kit, evaluator, and competition specification were provided by TechJam 2026.
