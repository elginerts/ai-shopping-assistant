# AI Shopping Assistant

My submission for the TikTok TechJam 2026 Conversational E-Commerce Search Challenge.

The agent searches a catalog of 50,000 clothing, shoe, and jewellery products. It can remember information across a conversation, ask follow-up questions, and update its search when the customer changes their mind. The goal is to place the customer's hidden target product in the Top 10 within ten turns.

## Current Results

Results from the organizer-provided 200-session public development set:

| Metric | Starter baseline | Current agent |
|---|---:|---:|
| TechnicalScore | 0.106710 | **0.785342** |
| Hit Rate@10 | 0.125 | **0.920** |
| MRR | 0.068034 | **0.584141** |
| MTTC | 9.81 | **3.495** |
| Token usage | 0 | **0** |

These are public-development results and may not represent performance on the private evaluation set.

### Results by Scenario

| Scenario | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Buying | 0.950 | 0.616037 | 2.7625 |
| Browsing | 0.975 | 0.587411 | 2.925 |
| Intent Override | 0.800 | 0.556746 | 6.033333 |
| Boundary | 0.600 | 0.385 | 6.3 |

## What I Changed

The provided starter was a stateless BM25 search agent. It searched only the latest customer message and did not ask any questions.

I added:

- Separate conversation memory for each customer session
- Search using the full conversation instead of only the latest message
- Follow-up questions about feature, material, colour, style, and size
- Recommendations on the same turn as a follow-up question
- Intent-override handling when the customer replaces an earlier preference
- A restarted clarification sequence after an intent change
- Different recommendations across turns instead of repeating failed products
- Extra stopwords to reduce noise from common conversational phrases
- Seven tests for the new multi-turn behavior

## How It Works

```text
Customer message
      |
      v
Update session memory
      |
      v
Handle changed preferences
      |
      v
Build a query from the conversation
      |
      v
Search the SQLite FTS5 product index
      |
      v
Return Top 10 products + a follow-up question
```

### Conversation Memory

Each session stores its own messages, customer profile, and previously asked attributes. This prevents one customer's information from entering another customer's session.

### Clarification Strategy

The agent currently asks questions in this order:

```text
feature -> material -> color -> style -> size
```

It still recommends products while asking a question, so a clarification turn does not remove the chance of finding the target immediately.

### Intent Changes

When a customer says to ignore an earlier preference, the agent keeps the original product category but removes the outdated preference. It also restarts its follow-up questions because the new intent may have different requirements.

### Recommendation Diversity

The agent remembers which products it has already shown and leaves them out of later turns. This increases catalog coverage across the ten-turn limit. The seen-product list is cleared after an intent change so previously shown products can be reconsidered under the new requirements.

### Product Retrieval

Products are stored in an in-memory SQLite FTS5 index. Results are ranked using field-weighted BM25. Titles and categories receive more weight than longer fields such as descriptions.

The agent is fully local and deterministic. It does not require an LLM, external API, API key, or internet connection during inference.

## Experiment History

| Version | Change | TechnicalScore | Hit Rate@10 |
|---|---|---:|---:|
| Baseline | Stateless BM25 | 0.106710 | 0.125 |
| Version 1 | Conversation memory and follow-up questions | 0.679255 | 0.800 |
| Version 2 | Restart questions after intent changes | 0.704846 | 0.830 |
| Version 3 | Avoid repeated products across turns | 0.785342 | 0.920 |

I also tested using profile tags to reorder the questions. It slightly lowered the overall public score, so I did not include that experiment in the current agent.

## Project Structure

```text
starter/agent.py                 shopping agent implementation
evaluator/local_evaluator.py     organizer-provided evaluator
tests/test_evaluator.py          organizer-provided evaluator tests
tests/test_agent_behavior.py     tests for my multi-turn changes
data/public_set.jsonl            200 public development sessions
data/catalog.jsonl               local catalog, not committed to Git
docs/                            competition rules and API contract
```

## Setup

### Requirements

- Python 3.10 or later
- No third-party Python packages
- No API key

### Download the Catalog

Download `catalog.jsonl.gz` and `SHA256SUMS` from the participant-kit release. Verify the archive before extracting it:

```bash
shasum -a 256 -c SHA256SUMS
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

The catalog should contain approximately 50,000 lines:

```bash
wc -l data/catalog.jsonl
```

### Run the Tests

```bash
python3 -m unittest
```

Current result:

```text
Ran 10 tests
OK
```

### Run the Public Evaluator

```bash
python3 -m evaluator.local_evaluator
```

The evaluator writes detailed session results to `results.json`.

## Runtime, Model, and Cost

| Item | Current setup |
|---|---|
| Retrieval | SQLite FTS5 with field-weighted BM25 |
| LLM | None |
| Network required | No |
| Prompt tokens | 0 |
| Completion tokens | 0 |
| Model/API cost | $0 |
| External dependencies | None |

A full evaluation of 200 public sessions took **15.68 seconds** on an Apple Silicon Mac running Python 3.14.7. This is about **78 ms per session**, including catalog indexing and evaluator overhead. Runtime will vary by machine.

## Tests Added

The added tests check that:

- Customer sessions keep separate conversation histories
- Questions are asked in the expected order
- Later answers refine the original request
- An intent change removes the old preference
- The agent recommends products while asking a question
- Failed recommendations are not repeated on later turns
- Products can be reconsidered after an intent change

## Current Limitations

- Boundary sessions remain the weakest scenario.
- The clarification order is fixed rather than selected dynamically.
- The customer profile is stored but is not currently used in ranking.
- BM25 depends on word overlap and may miss products described with synonyms.
- Public-set improvements may not transfer fully to the private evaluation set.
- The current implementation has been tested on a local Apple Silicon machine, not the organizer's final environment.

## Next Steps

- Improve Boundary-session handling
- Add semantic query expansion without requiring a live API
- Test profile information as a small ranking signal
- Add clearer explanations for recommended products
- Measure indexing time and per-turn latency separately

## Development Tools

- Python
- SQLite FTS5
- VS Code
- Git and GitHub

All score changes were checked using the organizer-provided local evaluator. The final agent itself does not call Codex or another LLM.

## Contribution

This is a solo submission.

## Data Attribution

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` for the full attribution. The starter kit, evaluator, and competition specification were provided by TechJam2026.
