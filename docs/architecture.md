# Architecture notes

This document explains why Threadline uses several small components instead of sending the whole shopping conversation to one large model.

## Request flow

1. `IntentTracker` writes additions, replacements, removals, exclusions, and category pivots to a versioned ledger.
2. SQLite FTS5 retrieves a recall-focused candidate list from the frozen catalogue.
3. During normal discovery, `nomic-embed-text` compares the request with the top 16 products.
4. Reciprocal-rank fusion combines the lexical and semantic positions.
5. After a correction, a fresh query is compiled from active ledger revisions before semantic reranking.
6. `ClarificationPolicy` simulates the possible answers for every available attribute.
7. A validated guardrail combines expected question value with answerability evidence.
8. The agent filters already-seen IDs and returns catalogue-grounded recommendations plus an optional decision trace.

## Versioned intent ledger

Each preference revision records its attribute, value, status, source turn, and replacement link. The active `slots` view remains small and fast for retrieval, while the ledger keeps the history needed for debugging and explanations.

Supported transitions include:

- `active → replaced` when a shopper supplies a new value for the same attribute
- `active → removed` when the shopper says an attribute no longer matters
- a full preference retirement when the product category changes or the shopper explicitly rejects the earlier intent

Selective corrections keep unrelated requirements. For example, “black instead of blue” replaces colour without removing an earlier leather requirement.

## Counterfactual question planner

For each possible question, the planner partitions the current candidates by their likely answer. It then calculates:

- Expected candidate reduction after an answer
- Expected increase in the chance that the remaining group fits inside the Top 10
- Catalogue coverage for that attribute
- An answerability prior based on observed evaluator behaviour

The unrestricted planner is available for ablation, but it reduced the public TechnicalScore. The default keeps the measured high-yield opening and validated information-gain selector while still computing and exposing counterfactual values. This is deliberate: new reasoning is observable, but it does not replace a stronger policy without evidence.

## Decision trace

The optional `decision_trace` response field contains:

- Active and historical ledger entries
- The retrieval strategy used on that turn
- Candidate and result counts
- Selected question policy
- Expected candidate reduction, expected Top-10 gain, coverage, and utility

The official evaluator ignores extra fields. The trace is intended for debugging, a portfolio demo, and judge questions about how the system made a decision.

## Why retrieval comes before the model

Embedding every product on every turn would be slow and unnecessary. BM25 narrows the 50,000-product catalogue to a small group with strong word-level evidence. The embedding model then performs the more expensive meaning comparison only where it can change the final ranking.

This also limits risk. The model cannot create a product ID because it only reorders IDs that already came from the catalogue index.

## Correction-aware semantic compilation

Dense embeddings place similar meanings close together, but that can be a weakness when a sentence changes direction. “I want blue shoes” and “I do not want blue shoes anymore” still share many concepts. An explicit correction therefore triggers three actions:

- Old intent slots are cleared.
- Previously shown products may be considered again.
- A new semantic query is compiled from active slots, excluding retired revisions.

The public ablation supported this choice. Compiling clean intent before reranking reached 0.602381 overall MRR, compared with 0.600220 when corrections stayed lexical-only. This keeps semantic matching without allowing old preferences back into the query.

## Embedding cache

Product vectors are stable because the catalogue and model are fixed. `EmbeddingCache` stores normalized float vectors in SQLite using `(model, parent_asin)` as the key.

The cache has three benefits:

- Repeat evaluations avoid recomputing product embeddings.
- The cache can grow gradually as new candidates are encountered.
- Changing the model name creates a separate namespace instead of mixing incompatible vectors.

Query vectors are not stored because conversations are short-lived and may contain user-specific information.

## Failure behaviour

Ollama is a required dependency. Startup checks the local `/api/tags` endpoint and confirms that `nomic-embed-text` is installed. A missing service or model stops the program with the exact setup command. The agent does not silently switch to a different algorithm, so evaluation behaviour stays consistent.

## Complexity

- Catalogue indexing: linear in the number of products, completed once at startup.
- Lexical search: handled by SQLite FTS5.
- Neural reranking: at most 16 product embeddings plus one query embedding per eligible turn.
- Counterfactual planning: linear in the first 100 candidates multiplied by the eight supported question attributes.
- Session state: proportional to messages, ledger revisions, asked attributes, and shown product IDs for one session.
