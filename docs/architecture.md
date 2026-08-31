# Architecture notes

This document explains why Threadline uses several small components instead of sending the whole shopping conversation to one large model.

`AgentConfig` validates environment settings before any session starts. `SessionState` gives the mutable per-shopper dictionary an explicit type contract, while the catalogue index, Ollama client, and embedding cache are shared across sessions. The public `Agent.reset` and `Agent.respond` interface remains compatible with the supplied evaluator.

## Request flow

1. `IntentTracker` writes additions, replacements, removals, exclusions, and category pivots to a versioned ledger.
2. SQLite FTS5 retrieves exact-word incumbents from the frozen catalogue.
3. In experimental mode, a NumPy matrix search retrieves independent Nomic challengers from all 50,000 products.
4. Nomic also reranks the first 16 lexical candidates using the same query vector.
5. A structured margin gate may replace only the final incumbent with a clearly stronger challenger.
6. A structured pass may reorder the existing Top 10 when one product has clearly stronger exact-constraint evidence.
7. After a correction, a fresh query is compiled from active ledger revisions before both retrieval routes run.
8. The first clarification captures any important constraint without guessing its slot.
9. `ClarificationPolicy` then simulates the possible answers for every available attribute.
10. The agent returns catalogue-grounded recommendations plus an optional decision trace.

The retrieval boundary uses typed `RetrievalResult` and `RetrievalEvidence` objects. Recommendation IDs, planner candidates, and diagnostic stages therefore have an explicit contract instead of relying on tuple position or loosely shaped dictionaries.

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

The first question is open-ended because failure diagnostics showed that a confident but incorrect slot question delayed useful evidence. It reduced wrong-question paths from nine failed sessions to three. Once the shopper answers, the planner switches to specific questions selected from candidate partitions.

## Learned challenger gate

The experimental gate uses a small pairwise linear model implemented with NumPy. Its inputs are dense, BM25, and Nomic ranks; semantic similarity; intent coverage; constraint validity; route; and revision state. Product IDs and session IDs are never features. `scripts/train_promotion_model.py` replays public conversations and saves inspectable JSON weights.

Ranks one through nine are protected and a challenger must clear a confidence margin. The combined system scored 0.798071, slightly below the 0.798992 default, so learned promotion is available through `THREADLINE_DENSE_MODE=learned` but is not enabled automatically.

## Structured Top-10 reranker

The deployed final stage only considers products already admitted to the Top 10. It compares each product with active ledger values, exclusions, and budget bounds. Promotion requires at least two active constraints, at least two exact matches, no contradiction, and a clear coverage advantage over the current leader.

This narrow contract protects recall: the component cannot add a product, remove a product, or change conversation timing. On the public evaluator, Hit@10 stayed at 0.940 and MTTC stayed at 3.16 while MRR improved from 0.573972 to 0.609472. TechnicalScore therefore increased from 0.798992 to 0.809642.

## Decision trace

Setting `THREADLINE_DECISION_TRACE=1` adds an optional `decision_trace` field containing:

- Active and historical ledger entries
- The retrieval strategy used on that turn
- Candidate and result counts
- Selected question policy
- Expected candidate reduction, expected Top-10 gain, coverage, and utility

The flag is off by default so official responses match the machine-readable API contract exactly. The trace is intended only for local debugging, a portfolio demo, and judge questions about how the system made a decision.

## Two retrieval entrances

BM25 remains strong for exact brands, colours, sizes, and model names, while dense retrieval handles synonyms and situational language. Product embeddings are prepared once and loaded into a NumPy matrix. Each turn needs one query embedding and one optimized matrix multiplication.

The existing lexical list acts as the incumbent ranking. Dense results are challengers, not an automatic replacement list. A challenger must exceed the last incumbent by a configured margin after semantic similarity and active-constraint coverage are considered.

This lane is disabled by default because evaluation did not support deployment. Always-on promotion scored 0.786133, while revision-only promotion scored 0.792364; both trailed the verified 0.793614 system. Keeping the implementation and diagnostics makes the negative result reproducible without weakening the submitted default.

The model cannot create a product ID because both entrances contain only IDs verified against the frozen catalogue.

## Correction-aware semantic compilation

Dense embeddings place similar meanings close together, but that can be a weakness when a sentence changes direction. “I want blue shoes” and “I do not want blue shoes anymore” still share many concepts. An explicit correction therefore triggers three actions:

- Old intent slots are cleared.
- Previously shown products may be considered again.
- A new semantic query is compiled from active slots, excluding retired revisions.

The public ablation supported this choice. Compiling clean intent before reranking reached 0.602381 overall MRR, compared with 0.600220 when corrections stayed lexical-only. This keeps semantic matching without allowing old preferences back into the query.

## Embedding cache

Product vectors are stable because the catalogue and model are fixed. The builder stores progress in SQLite using `(model, parent_asin)` as the key, then exports an aligned `float32` NumPy artifact.

The cache has three benefits:

- Repeat evaluations avoid recomputing product embeddings.
- The cache can grow gradually as new candidates are encountered.
- Changing the model name creates a separate namespace instead of mixing incompatible vectors.

The release artifact includes the catalogue checksum, model name, format version, product IDs, and vectors. Startup rejects incomplete, corrupt, mismatched, or duplicated indexes. Query vectors are not stored because conversations are short-lived and may contain user-specific information.

## Failure behaviour

Ollama is a required dependency. Startup checks the local `/api/tags` endpoint and confirms that `nomic-embed-text` is installed. A missing service or model stops the program with the exact setup command. The agent does not silently switch to a different algorithm, so evaluation behaviour stays consistent.

## Candidate-recall diagnostics

The evaluator records the target's rank at the BM25, post-Nomic, final-candidate, and recommendation stages for failed sessions. It also records whether the target appeared before an intent override and whether a question could reveal any undisclosed target constraint. Ground truth stays inside the evaluator: the agent exports stage snapshots without knowing which product is correct. This keeps the diagnostic useful without creating target leakage.

## Complexity

- Catalogue indexing: linear in the number of products, completed once at startup.
- Lexical search: handled by SQLite FTS5.
- Dense retrieval: one 50,000 × 768 NumPy matrix multiplication per turn.
- Neural query work: one Nomic query embedding per turn; product vectors are prebuilt.
- Counterfactual planning: linear in the first 100 candidates multiplied by the eight supported question attributes.
- Structured reranking: linear in the Top 10 multiplied by the number of active constraints.
- Session state: proportional to messages, ledger revisions, asked attributes, and shown product IDs for one session.

## Challenge constraints

The implementation stays inside the supplied scope: the catalogue is read-only, outputs use existing parent ASINs, sessions stop within the evaluator's 10-turn limit, and runtime processing is text-only. The default uses SQLite and a bounded local rerank rather than a heavy vector database. Ollama is declared as a required local service; no credentials or paid API calls are used.
