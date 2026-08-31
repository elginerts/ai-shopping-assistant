from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent
from starter.promotion import fit_pairwise_ranker


def collect_pairs(
    agent: Agent,
    samples: list[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]], int]:
    """Replay public conversations and collect target-versus-distractor pairs."""

    positives: list[tuple[float, ...]] = []
    negatives: list[tuple[float, ...]] = []
    useful_sessions = 0
    for sample_number, sample in enumerate(samples):
        session_id = f"training_{sample_number}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        replay_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            replay_sample, coarse_category(categories.get(target, [])), disclosed
        )
        session_added = False

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            rows = {row.parent_asin: row for row in agent.promotion_snapshot(session_id)}
            target_row = rows.get(target)
            if override_applied and target_row is not None:
                # Compare with at most ten hard negatives. This stops long
                # sessions and common product types from dominating training.
                ranked = [
                    str(item.get("parent_asin", ""))
                    for item in response.get("recommendations", [])
                    if isinstance(item, dict)
                ]
                distractors = [rows[item] for item in ranked if item in rows and item != target]
                if not distractors:
                    distractors = [row for key, row in rows.items() if key != target][:10]
                for distractor in distractors[:10]:
                    positives.append(target_row.features)
                    negatives.append(distractor.features)
                session_added = session_added or bool(distractors)

            if turn == MAX_TURNS:
                break
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                user_message, boundary_used = customer_reply(
                    replay_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
        useful_sessions += int(session_added)
        if (sample_number + 1) % 25 == 0:
            print(f"Collected training evidence from {sample_number + 1}/{len(samples)} sessions")
    return positives, negatives, useful_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Threadline's small promotion ranker")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="models/promotion_model.json")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    _catalog_ids, categories, products = catalog_index(args.catalog)
    random.Random(args.seed).shuffle(samples)
    agent = Agent(args.catalog, dense_mode="challenger")
    # Collection must observe the baseline slate rather than its old heuristic.
    agent.index.promotion_margin = 1_000_000.0
    try:
        positives, negatives, useful_sessions = collect_pairs(
            agent, samples, categories, products
        )
    finally:
        agent.connection.close()

    model = fit_pairwise_ranker(positives, negatives)
    model.save(args.output, metadata={
        "training_pairs": len(positives),
        "useful_sessions": useful_sessions,
        "seed": args.seed,
        "dataset": Path(args.dataset).name,
        "note": "Product and session identifiers are excluded from model features.",
    })
    print(json.dumps({
        "output": args.output,
        "training_pairs": len(positives),
        "useful_sessions": useful_sessions,
    }, indent=2))


if __name__ == "__main__":
    main()
