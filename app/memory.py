"""Cognee memory integration for fact storage and retrieval."""
# pylint: disable=import-error

from pathlib import Path
from typing import Dict, List, Optional

import cognee
from cognee import SearchType
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.users.methods import get_default_user


DEFAULT_DATASET_NAME = "main_dataset"


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


async def _ensure_graph_context(dataset_name: Optional[str]) -> None:
    """
    Ensure the graph database context matches the dataset when access control is enabled.
    """
    user = await get_default_user()
    target_dataset = dataset_name or DEFAULT_DATASET_NAME
    await set_database_global_context_variables(target_dataset, user.id)


async def visualize_knowledge_graph(
    output_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Render the current Cognee knowledge graph to an interactive HTML file.
    Returns the path to the generated HTML file.
    """
    await _ensure_graph_context(dataset_name)

    if output_path:
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        await visualize_graph(str(target))
        return target

    await visualize_graph()
    return None


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
