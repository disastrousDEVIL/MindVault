from typing import List, Dict

import cognee


def init_memory():
    """
    Initialize Cognee.
    This should be called once at app startup.
    """
    cognee.init()


def store_facts(facts: List[Dict]) -> None:
    """
    Store atomic facts in Cognee.
    Each fact is assumed to be validated already.
    """
    for fact in facts:
        cognee.add(
            content={
                "subject": fact["subject"],
                "relation": fact["relation"],
                "object": fact["object"],
                "confidence": fact["confidence"],
                "source_document_id": fact["source_document_id"],
            },
            content_type="atomic_fact",
        )


def retrieve_facts(query: str, limit: int = 20) -> List[Dict]:
    """
    Retrieve relevant facts from Cognee for a given query.
    """
    results = cognee.search(
        query=query,
        content_type="atomic_fact",
        limit=limit,
    )

    facts = []
    for item in results:
        facts.append(item["content"])

    return facts