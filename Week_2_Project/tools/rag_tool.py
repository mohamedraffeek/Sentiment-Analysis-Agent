"""Tool wrapping a RAG (Retrieval-Augmented Generation) system for document retrieval and knowledge-based question answering.

This integrates with LangChain StructuredTool interface so the agent can invoke it.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from rag.rag import RAG
import json


def _rag_call(inputs: str) -> str:
    try:
        data = json.loads(inputs)
        docs_query = data.get("docs_query", "")
        user_query = data.get("user_query", "")
        print(f"[Tool] rag_tool called with docs_query='{docs_query}' user_query='{user_query[:50]}...'")
        rag = RAG(docs_query=docs_query, user_query=user_query)
        relevant_docs = rag.get_relevant_documents()
        combined_content = "\n\n".join([doc.page_content for doc in relevant_docs])
        return combined_content
    except Exception as e:
        error_msg = f"Error in RAG tool: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg

RAGTool = StructuredTool.from_function(
    name="rag_tool",
    description=(
        "Use this tool to answer user queries with RAG (Retrieval-Augmented Generation). "
        "You must provide both: 'docs_query' (the topic area to fetch docs from Arxiv) "
        "and 'user_query' (the actual user question) in JSON format."
    ),
    func=_rag_call,
)

__all__ = ["RAGTool"]