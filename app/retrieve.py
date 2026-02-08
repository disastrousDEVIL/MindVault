"""Retrieve and prune facts for question answering."""

from typing import Dict, List

from app.memory import retrieve_facts


CONFIDENCE_THRESHOLD = 0.3
MAX_FACTS = 20


async def retrieve_context(question: str) -> List[Dict]:
    """
    Retrieve relevant atomic facts for a given question.
    Applies pruning and limits.
    """
    # Step 1: Get raw facts from memory
    raw_facts = await retrieve_facts(query=question, limit=MAX_FACTS * 2)

    if not raw_facts:
        return []

    # Step 2: Prune by confidence
    normalized = []
    for fact in raw_facts:
        if isinstance(fact, dict):
            normalized.append(fact)
        else:
            normalized.append({"text": str(fact), "confidence": 1.0})

    filtered = [
        fact
        for fact in normalized
        if float(fact.get("confidence", 1.0)) >= CONFIDENCE_THRESHOLD
    ]

    # Step 3: Deduplicate facts
    seen = set()
    deduped = []

    for fact in filtered:
        if all(field in fact for field in ("subject", "relation", "object")):
            key = (
                str(fact.get("subject", "")).lower(),
                str(fact.get("relation", "")).lower(),
                str(fact.get("object", "")).lower(),
            )
        else:
            key = str(fact.get("text", fact)).lower()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(fact)

        if len(deduped) >= MAX_FACTS:
            break

    return deduped
