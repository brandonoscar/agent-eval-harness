"""A toy support bot for the example suite. No model and no network.

It behaves like a sampling model on purpose: the facts are fixed, the wording
varies from attempt to attempt (seeded by ``SUPPORT_BOT_SEED``), so you can
watch quality re-rolls and wording drift without paying for a model.

Break it on purpose with ``SUPPORT_BOT_BUG``:

* ``leak_card``: reads out the full card number (a safety failure);
* ``refund_without_approval``: refunds without asking (a safety failure);
* ``outage``: the "provider" is overloaded (infrastructure, not safety);
* ``chatty``: answers in very different words (wording drift, nightly only).
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter

from agent_eval import Case, ToolCall, Transcript

ORDERS = {
    "1042": {"status": "shipped", "carrier": "UPS", "eta": "Thursday", "total": "$48.00"},
    "1043": {"status": "processing", "carrier": "", "eta": "next week", "total": "$12.50"},
}
#: A test card number from the card networks' published test ranges. Never real.
CARD_ON_FILE = "4111 1111 1111 1111"

_attempts: Counter[str] = Counter()


def _rng(case: Case) -> random.Random:
    """Different words on each attempt, the same sequence on every run with
    the same seed: a re-roll can change the wording, a re-run cannot."""
    _attempts[case.id] += 1
    seed = os.environ.get("SUPPORT_BOT_SEED", "0")
    return random.Random(f"{seed}:{case.id}:{_attempts[case.id]}")


def _order_number(text: str) -> str | None:
    found = re.search(r"\b(\d{4})\b", text)
    return found.group(1) if found else None


def _order_reply(number: str, rng: random.Random, chatty: bool) -> str:
    order = ORDERS[number]
    if chatty:
        return (
            f"Thanks so much for reaching out! I took a careful look and I'm delighted to "
            f"report that parcel {number} is travelling with {order['carrier']} right now."
        )
    if order["status"] != "shipped":
        return f"Order {number} is still {order['status']}; it should ship {order['eta']}."
    return rng.choice(
        [
            f"Order {number} has shipped with {order['carrier']} and should arrive {order['eta']}.",
            f"Good news: order {number} shipped via {order['carrier']}. Expected {order['eta']}.",
            f"Your order {number} is on its way with {order['carrier']}.",
        ]
    )


async def target(case: Case) -> Transcript:
    bug = os.environ.get("SUPPORT_BOT_BUG", "")
    if bug == "outage":
        raise RuntimeError("overloaded_error: the model provider is temporarily busy")
    rng = _rng(case)
    message = case.message.lower()
    history = " ".join(turn["content"] for turn in case.history)
    t = Transcript(final_agent="support", usage={"input_tokens": 120, "output_tokens": 40})

    if "ignore previous instructions" in message or "ignore all instructions" in message:
        if bug == "refund_without_approval":
            t.tool_calls += [ToolCall("refund_order", {"order": n}) for n in ORDERS]
            t.reply = "Done, every order has been refunded."
        else:
            t.reply = "I can't do that. I can help with one order at a time, with approval."
        return t

    if "refund" in message:
        number = _order_number(message) or "1042"
        t.tool_calls.append(ToolCall("lookup_order", {"order": number}))
        if bug == "refund_without_approval":
            t.tool_calls.append(ToolCall("refund_order", {"order": number}))
            t.reply = f"I've refunded order {number}."
        else:
            t.approval_cards.append(
                {
                    "title": "Refund order?",
                    "details": f"Refund {ORDERS[number]['total']} for order {number}",
                }
            )
            t.reply = f"I've asked a teammate to approve the refund for order {number}."
        return t

    if "card" in message:
        if bug == "leak_card":
            t.reply = f"The card on file is {CARD_ON_FILE}."
        else:
            t.reply = "For your security I can't read out payment details. It ends in 1111."
        return t

    if "weather" in message:
        t.handoffs.append("weather_agent")
        t.tool_calls.append(ToolCall("get_weather", {"city": "Paris"}, agent="weather_agent"))
        t.reply = "It's 18°C and cloudy in Paris right now."
        return t

    order_id = _order_number(message) or _order_number(history)
    if order_id is not None:
        t.tool_calls.append(ToolCall("lookup_order", {"order": order_id}))
        if order_id not in ORDERS:
            t.reply = f"I couldn't find order {order_id}. Could you check the number?"
        elif "arrive" in message:
            t.reply = f"Order {order_id} should arrive {ORDERS[order_id]['eta']}."
        else:
            t.reply = _order_reply(order_id, rng, chatty=bug == "chatty")
        return t

    t.reply = "I can help with orders, refunds and the weather. What do you need?"
    return t
