# Devpost project description draft

## Project overview

Threadline is a local multi-turn shopping copilot for TikTok TechJam 2026 Track 4. It searches the organizer's frozen 50,000-product Amazon catalogue, returns recommendations on every turn, asks useful clarification questions, and adapts when a shopper replaces or removes an earlier preference.

The main idea is a correction-aware search and decision engine. A versioned intent ledger records preferences as active, replaced, or removed. Search queries are rebuilt from active entries only, preventing stale values from influencing later recommendations. The verified pipeline combines field-weighted BM25 retrieval, bounded Nomic semantic reranking, confidence-gated Top-10 constraint reranking, and candidate-aware question planning.

On the 200 public development sessions, the verified default reached TechnicalScore 0.809642, Hit Rate@10 0.940, MRR 0.609472, and MTTC 3.16.

## Development tools

- Visual Studio Code
- Python 3
- Git and GitHub
- macOS Terminal

## APIs, libraries, and frameworks

- Ollama's local HTTP API for `nomic-embed-text`
- Python standard library, including SQLite FTS5
- NumPy for portable in-memory dense-index experiments

No external paid API, hosted model credential, or live key is required. After Ollama and the model are installed, scoring runs locally without internet access.

## Dataset and assets

- Official frozen 50,000-product catalogue supplied for Track 4
- Official 200-session public development set
- Data derived by the organizer from Amazon Reviews 2023 Clothing, Shoes and Jewelry
- `nomic-embed-text` distributed under Apache License 2.0

The catalogue remains read-only, and every recommendation is an existing parent ASIN from that catalogue.

## Limitations and future work

Dense retrieval can recover products that lexical retrieval misses, but aggressive promotion can lower MRR. With more time, I would calibrate a confidence-aware admission policy on a separate validation split. Abrupt intent overrides can also leave the desired product outside the original lexical pool; I would evaluate a correction-specific expansion pass against its added latency.

## Team contribution

This is a solo submission completed by Er Teng Sheng Elgin.

## Submission reminder

Upload the required demonstration to YouTube with public visibility, then add its link to the Devpost video field and written description before submission.
