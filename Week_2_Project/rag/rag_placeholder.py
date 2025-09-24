"""Placeholder RAG tool for future integration.

Design goals for future implementation:
- Provide a retrieval interface over a vector store (FAISS, Chroma, etc.).
- Support document ingestion (PDF, text, web URLs) with chunking & embeddings.
- Provide a `retrieve` method returning context snippets.
- Optionally implement a separate tool for 'explain_with_context' that crafts an
  answer combining agent LLM reasoning + retrieved passages.

Current placeholder returns a static explanatory message so that the rest of the
agent architecture can already handle the presence (or absence) of this tool.
"""
from __future__ import annotations

from smolagents import Tool  # type: ignore


class RagKnowledgeTool(Tool):
    name = "rag_knowledge_tool"
    description = (
        "(Placeholder) Retrieve background knowledge on general topics. "
        "Future version will query a vector store."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Natural language query to retrieve context for.",
            "required": True,
        }
    }
    output_type = "string"

    def run(self, query: str) -> str:  # type: ignore[override]
        return (
            "RAG system not yet implemented. Planned enhancements: embedding docs, vector "
            "similarity search, and grounded answers. Your query was: " + query
        )


__all__ = ["RagKnowledgeTool"]
