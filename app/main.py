"""FastAPI entrypoint for the MindVault API."""
# pylint: disable=import-error

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.memory import init_memory, store_facts
from app.ingest import ingest_document
from app.retrieve import retrieve_context
from app.llm import llm_client


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize services on startup and clean up on shutdown."""
    init_memory()
    yield


app = FastAPI(title="MindVault", lifespan=lifespan)


# ---------- Schemas ----------

class IngestRequest(BaseModel):
    """Payload for document ingestion."""
    document_id: str
    title: str
    content: str


class QueryRequest(BaseModel):
    """Payload for answering a question."""
    question: str


# ---------- Routes ----------

@app.post("/ingest")
async def ingest(req: IngestRequest):
    """Ingest a document and store extracted facts."""
    try:
        facts = ingest_document(
            document_id=req.document_id,
            title=req.title,
            content=req.content,
        )
        await store_facts(facts)
        return {"status": "success", "facts_stored": len(facts)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/query")
async def query(req: QueryRequest):
    """Answer a question using stored facts."""
    facts = await retrieve_context(req.question)

    if not facts:
        return {
            "answer": "Not enough information in memory to answer this question.",
            "sources": [],
        }

    prompt = build_answer_prompt(req.question, facts)
    response = llm_client.generate_answer(prompt)

    return response


# ---------- Prompt builder ----------

def build_answer_prompt(question: str, facts: list) -> str:
    """Build the LLM prompt for answer generation."""
    return f"""
You are given a question and a list of facts.

Answer the question using ONLY the provided facts.
If the facts are insufficient, say so.

Facts:
{facts}

Question:
{question}

Return JSON in this format:
{{
  "answer": "string",
  "sources": ["document_id"]
}}
"""
