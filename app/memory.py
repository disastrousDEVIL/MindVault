"""Cognee memory integration for fact storage and retrieval."""
# pylint: disable=import-error

from typing import Dict, List

import cognee
from cognee import SearchType


def init_memory() -> None:
    """
    Initialize Cognee.

    Cognee loads configuration from environment variables on import, so there is
    no explicit init step required. This remains for future startup hooks.
    """
    return None


def _format_fact(fact: Dict) -> str:
    """Render a fact dict into a single text string for Cognee ingestion."""
    subject = str(fact.get("subject", "")).strip()
    relation = str(fact.get("relation", "")).strip()
    obj = str(fact.get("object", "")).strip()
    confidence = fact.get("confidence")
    source_id = fact.get("source_document_id")

    core = " ".join(part for part in [subject, relation, obj] if part)
    meta = []
    if confidence is not None:
        meta.append(f"confidence={confidence}")
    if source_id:
        meta.append(f"source={source_id}")

    if meta:
        return f"{core} ({', '.join(meta)})"

    return core


async def store_facts(facts: List[Dict]) -> None:
    """
    Store atomic facts in Cognee.
    Each fact is assumed to be validated already.
    """
    if not facts:
        return

    texts = [_format_fact(fact) for fact in facts]
    await cognee.add(texts)
    await cognee.cognify()


async def retrieve_facts(query: str, limit: int = 20) -> List[Dict]:
    """
    Retrieve relevant facts from Cognee for a given query.
    """
    results = await cognee.search(
        query_text=query,
        query_type=SearchType.CHUNKS,
        top_k=limit,
    )

    facts = []
    for item in results:
        if hasattr(item, "search_result"):
            facts.append(item.search_result)
        elif isinstance(item, dict) and "search_result" in item:
            facts.append(item["search_result"])
        else:
            facts.append(item)

    return facts
