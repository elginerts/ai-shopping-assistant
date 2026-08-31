from __future__ import annotations

import json

from starter.agent import Agent


DEMO_MESSAGES = (
    "I am looking for running shoes, but I am still exploring.",
    "For that, what matters is: breathable; color: blue.",
    "Make it black instead of blue.",
)


def main() -> None:
    # This uses the same public Agent methods as the official evaluator.
    agent = Agent("data/catalog.jsonl")
    session_id = "threadline_demo"
    agent.reset(session_id, {
        "purchase_frequency": "occasional",
        "average_prior_rating": 4.2,
        "rating_style": "practical",
        "preference_tags": [],
        "summary": "Looks for practical clothing and footwear.",
    })
    try:
        for turn, user_message in enumerate(DEMO_MESSAGES, start=1):
            response = agent.respond(session_id, user_message, turn, 10)
            # Keep the terminal output short enough to explain in a video.
            print(json.dumps({
                "turn": turn,
                "user_message": user_message,
                "assistant_message": response["message"],
                "ask_attribute": response["ask_attribute"],
                "top_3_parent_asins": [
                    item["parent_asin"]
                    for item in response["recommendations"][:3]
                ],
            }, indent=2))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
