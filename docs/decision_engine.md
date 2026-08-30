# Decision engine

Threadline treats a shopping conversation as changing state plus a sequence of decisions. This document describes the two parts that make that behaviour inspectable.

## Preference revisions

A ledger entry contains:

```json
{
  "revision_id": 4,
  "attribute": "color",
  "value": "blue",
  "status": "replaced",
  "turn": 1,
  "replaced_by": 7,
  "ended_turn": 3
}
```

The retrieval layer reads only the active state. Historical revisions remain available for explanations and debugging, but they do not become positive search terms again.

Full overrides retire every preference while preserving the category when appropriate. Selective replacements retire only one attribute. An explicit “no longer matters” instruction removes that attribute without weakening the rest of the request.

## Question simulation

For a candidate set of size `N`, a possible question divides products into answer groups with sizes `n₁ ... nₖ`.

```text
expected_remaining = Σ(nᵢ²) / N
candidate_reduction = 1 - expected_remaining / N
```

The Top-10 estimate compares the chance that a random remaining candidate fits into the first ten positions before and after the split. The final experimental utility combines:

```text
0.50 × expected candidate reduction
+ 0.30 × expected Top-10 gain
+ 0.20 × attribute coverage
```

That value is multiplied by an answerability prior. A question is not useful if it creates a mathematically clean split but shoppers rarely know the answer.

## Guarded deployment

Two modes are available:

- `guarded`: measured high-yield questions first, followed by validated information gain; counterfactual values are calculated and reported.
- `counterfactual`: the experimental utility directly selects every question.

The unrestricted mode raised Boundary Hit@10 from 0.60 to 0.70, but reduced overall Hit@10 from 0.925 to 0.885. It is kept for reproducibility rather than presented as an improvement. The guarded mode retains the verified 0.792766 TechnicalScore.

## Example trace

```json
{
  "intent": {
    "route": "buying",
    "active": {"category": ["shoes"], "color": ["black"]},
    "history": [
      {"attribute": "color", "value": "blue", "status": "replaced", "turn": 1},
      {"attribute": "color", "value": "black", "status": "active", "turn": 3}
    ]
  },
  "retrieval": {
    "strategy": "lexical_after_correction",
    "candidate_count": 100,
    "returned_count": 10
  },
  "clarification": {
    "attribute": "material",
    "policy": "high_yield_guardrail",
    "expected_candidate_reduction": 0.61,
    "expected_top_k_gain": 0.27,
    "coverage": 0.84
  }
}
```
