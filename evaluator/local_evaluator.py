from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import uuid
from collections import defaultdict
from pathlib import Path

from starter.agent import Agent


MAX_TURNS = 10
TOP_K = 10
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def intent_card(product: dict, limit: int = 180) -> dict:
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean_constraint(item, limit) for item in candidates if _clean_constraint(item, limit)))
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def behavior_for(scenario: str, card: dict, rng: random.Random) -> dict:
    behavior: dict = {"scenario_type": scenario}
    if scenario == "intent_override":
        hard = card["hard_constraints"]
        soft = card["soft_preferences"]
        old_value = soft[-1] if soft else "I prefer a different style."
        new_value = hard[0] if hard else "Please prioritize the target requirements."
        behavior["override"] = {
            "turn": rng.choice([3, 4]),
            "old_value": old_value,
            "new_value": new_value,
            "message": f"Actually, ignore my earlier preference. What I need is: {new_value}.",
        }
    return behavior


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= TOP_K:
            break
    return result


def catalog_index(catalog_path: str | Path) -> tuple[set[str], dict[str, list[str]], dict[str, dict]]:
    identifiers: set[str] = set()
    categories: dict[str, list[str]] = {}
    products: dict[str, dict] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            identifiers.add(parent_asin)
            categories[parent_asin] = [str(value) for value in product.get("categories") or []]
            products[parent_asin] = product
    return identifiers, categories, products


def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def initial_message(sample: dict, category: str, disclosed: set[str]) -> str:
    scenario = sample["scenario_type"]
    if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
        constraint = str(sample["intent_card"]["hard_constraints"][0])
        disclosed.add(constraint)
        return f"I'm looking for {category}. A key requirement is: {constraint}."
    if scenario == "intent_override":
        old_value = str(sample["behavior"]["override"]["old_value"])
        return f"I'm looking for {category}. {old_value}"
    return f"I'm looking for {category}, but I'm still exploring."


def customer_reply(sample: dict, ask_attribute: object, disclosed: set[str], boundary_used: bool) -> tuple[str, bool]:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"I don't have a preference for {attribute}; please use your judgment.", True
    if not attribute:
        return "Those options are not quite right yet. Ask me about one specific attribute.", boundary_used
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"
    constraints = [
        *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
        *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [
        value for value in constraints
        if value not in disclosed and (attribute == "other" or classify_constraint(value) == attribute)
    ][:2]
    if not matches:
        return f"I don't have an additional preference for {attribute}.", boundary_used
    disclosed.update(matches)
    return "For that, what matters is: " + "; ".join(matches) + ".", boundary_used


def metric_summary(sessions: list[dict]) -> dict:
    if not sessions:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
    hit_rate = sum(int(item["hit"]) for item in sessions) / len(sessions)
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else MAX_TURNS + 1 for item in sessions
    )
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
    }


def _rank_in(values: object, target: str) -> int | None:
    if not isinstance(values, list) or target not in values:
        return None
    return values.index(target) + 1


def _remaining_constraint_attributes(sample: dict, disclosed: set[str]) -> set[str]:
    card = sample.get("intent_card") or {}
    constraints = [
        *[str(value) for value in card.get("hard_constraints", [])],
        *[str(value) for value in card.get("soft_preferences", [])],
    ]
    return {classify_constraint(value) for value in constraints if value not in disclosed}


def failure_diagnostic(
    sample: dict,
    target: str,
    turns: list[dict],
) -> dict:
    """Classify a miss using evidence collected without target-aware retrieval."""
    eligible = [turn for turn in turns if turn["override_applied"]]
    before_override = [turn for turn in turns if not turn["override_applied"]]

    absent_from_bm25 = all(turn["target_bm25_rank"] is None for turn in eligible)
    pushed_down_by_nomic = any(
        turn["target_bm25_rank"] is not None
        and turn["target_bm25_rank"] <= TOP_K
        and (turn["target_post_nomic_rank"] is None or turn["target_post_nomic_rank"] > TOP_K)
        for turn in eligible
    )
    found_by_dense = any(turn.get("target_dense_rank") is not None for turn in eligible)
    below_top_10 = any(
        turn["target_final_candidate_rank"] is not None
        and turn["target_final_candidate_rank"] > TOP_K
        for turn in eligible
    )
    shown_before_override = any(turn["target_recommendation_rank"] is not None for turn in before_override)
    question_mismatch_turns = [
        turn["turn"] for turn in eligible
        if turn["question_path_mismatch"]
    ]

    reasons: list[str] = []
    if absent_from_bm25:
        reasons.append("absent_from_bm25_candidates")
    if absent_from_bm25 and found_by_dense:
        reasons.append("found_by_dense_but_not_promoted")
    if absent_from_bm25 and not found_by_dense:
        reasons.append("absent_from_bm25_and_dense")
    if pushed_down_by_nomic:
        reasons.append("pushed_down_by_nomic")
    if below_top_10:
        reasons.append("final_candidate_below_top_10")
    if shown_before_override:
        reasons.append("previously_shown_before_override")
    if question_mismatch_turns:
        reasons.append("likely_wrong_question_path")
    if not reasons:
        reasons.append("unclassified_ranking_or_filtering_failure")

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "target_parent_asin": target,
        "primary_reason": reasons[0],
        "reasons": reasons,
        "question_mismatch_turns": question_mismatch_turns,
        "turns": eligible,
    }


def diagnostic_summary(failures: list[dict]) -> dict:
    counts: defaultdict[str, int] = defaultdict(int)
    for failure in failures:
        for reason in failure["reasons"]:
            counts[reason] += 1
    return {
        "failed_session_count": len(failures),
        "reason_counts": dict(sorted(counts.items())),
        "note": "One failed session may have more than one observed reason.",
    }


def materialize_hidden_fields(sample: dict, products: dict[str, dict]) -> tuple[dict, dict]:
    if "intent_card" in sample and "behavior" in sample:
        return sample["intent_card"], sample["behavior"]
    target = str(sample["ground_truth"]["parent_asin"])
    product = products[target]
    card = intent_card(product)
    seed_source = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
    rng = random.Random(seed_source)
    behavior = behavior_for(str(sample["scenario_type"]), card, rng)
    return card, behavior


def evaluate(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    sessions: list[dict] = []
    failures: list[dict] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    promotion_turns = 0
    target_promotions = 0
    target_displacements = 0
    for sample in samples:
        session_id = f"public_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        hit_turn: int | None = None
        best_rank: int | None = None
        diagnostic_turns: list[dict] = []
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            usage = response.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                    total_prompt_tokens += usage["prompt_tokens"]
                if isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] >= 0:
                    total_completion_tokens += usage["completion_tokens"]
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            snapshot_method = getattr(agent, "diagnostic_snapshot", None)
            snapshot = snapshot_method(session_id) if callable(snapshot_method) else {}
            if snapshot.get("promoted"):
                promotion_turns += 1
            if target in snapshot.get("promoted", []):
                target_promotions += 1
            if (
                target in snapshot.get("incumbents", [])
                and target not in snapshot.get("recommended", [])
            ):
                target_displacements += 1
            remaining_attributes = _remaining_constraint_attributes(effective_sample, disclosed)
            asked_attribute = response.get("ask_attribute")
            diagnostic_turns.append({
                "turn": turn,
                "override_applied": override_applied,
                "ask_attribute": asked_attribute,
                "remaining_constraint_attributes": sorted(remaining_attributes),
                "question_path_mismatch": bool(
                    asked_attribute
                    and remaining_attributes
                    and asked_attribute != "other"
                    and asked_attribute not in remaining_attributes
                ),
                "target_bm25_rank": _rank_in(snapshot.get("bm25_ranked"), target),
                "target_post_nomic_rank": _rank_in(snapshot.get("post_nomic_ranked"), target),
                "target_final_candidate_rank": _rank_in(snapshot.get("final_candidates"), target),
                "target_dense_rank": _rank_in(snapshot.get("dense_ranked"), target),
                "target_promoted": target in snapshot.get("promoted", []),
                "target_recommendation_rank": _rank_in(ranked, target),
                "target_seen_before_turn": target in snapshot.get("seen_before", []),
            })
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                )
        session_result = {
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        }
        sessions.append(session_result)
        if hit_turn is None:
            failures.append(failure_diagnostic(effective_sample, target, diagnostic_turns))

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "failure_diagnostic_summary": diagnostic_summary(failures),
        "dense_promotion_summary": {
            "promotion_turns": promotion_turns,
            "target_promotions": target_promotions,
            "target_displacements": target_displacements,
        },
        "failure_diagnostics": failures,
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam public-set local evaluator")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument(
        "--diagnostics-output",
        help="Optional separate JSON report containing only failed-session diagnostics",
    )
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.diagnostics_output:
        diagnostic_report = {
            "summary": result["failure_diagnostic_summary"],
            "failures": result["failure_diagnostics"],
        }
        Path(args.diagnostics_output).write_text(
            json.dumps(diagnostic_report, indent=2) + "\n",
            encoding="utf-8",
        )
    hidden_from_console = {"sessions", "failure_diagnostics"}
    print(json.dumps({
        key: value for key, value in result.items() if key not in hidden_from_console
    }, indent=2))


if __name__ == "__main__":
    main()
