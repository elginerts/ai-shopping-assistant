# Architecture notes

This document explains why Threadline uses several small components instead of sending the whole shopping conversation to one large model.

## Request flow

1. `IntentTracker` updates the category, route, preferences, exclusions, and budget.
2. SQLite FTS5 retrieves a recall-focused candidate list from the frozen catalogue.
3. During normal discovery, `nomic-embed-text` compares the request with the top 16 products.
4. Reciprocal-rank fusion combines the lexical and semantic positions.
5. After an explicit correction, semantic reranking is disabled for that session and the cleaned lexical conversation is used.
6. `ClarificationPolicy` examines the candidate attributes and chooses the next question.
7. The agent filters already-seen IDs and returns catalogue-grounded recommendations.

## Why retrieval comes before the model

Embedding every product on every turn would be slow and unnecessary. BM25 narrows the 50,000-product catalogue to a small group with strong word-level evidence. The embedding model then performs the more expensive meaning comparison only where it can change the final ranking.

This also limits risk. The model cannot create a product ID because it only reorders IDs that already came from the catalogue index.

## Correction-aware semantic gate

Dense embeddings place similar meanings close together, but that can be a weakness when a sentence changes direction. “I want blue shoes” and “I do not want blue shoes anymore” still share many concepts. An explicit correction therefore triggers three actions:

- Old intent slots are cleared.
- Previously shown products may be considered again.
- Semantic reranking is turned off for the rest of that session.

The public ablation supported this choice. Always-on semantic reranking reduced Intent Override MRR from 0.565079 to 0.535675. The gate recovered the lost ranking quality while keeping the gains in other scenarios.

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
- Candidate clarification: linear in the first 100 retrieved candidates and the small attribute set.
- Session state: proportional to messages, asked attributes, and shown product IDs for one session.
