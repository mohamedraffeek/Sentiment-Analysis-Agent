""" RAG module
An implementation of Retrieval-Augmented Generation (RAG) using LangChain and HuggingFace embeddings.
"""
from __future__ import annotations

import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class RAG():
    """RAG class to encapsulate RAG functionality."""

    def __init__(self, docs_query: str, user_query: str):
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        print(f"[RAG] Embedding the user query...")
        self.query_embedding = self.embed_query(user_query)
        print(f"[RAG] User query embedded.")
        print(f"[RAG] Loading documents")
        self.chunked_docs = self.get_documents(docs_query)
        print(f"[RAG] Documents loaded and chunked: {len(self.chunked_docs)} chunks.")
        print(f"[RAG] Embedding documents...")
        self.doc_embeddings = self.embed_documents(self.chunked_docs)
        print(f"[RAG] Document embeddings completed.")


    def get_relevant_documents(self):
        similarity_scores = np.array(self.query_embedding) @ np.array(self.doc_embeddings).T
        sorted_idx = similarity_scores.argsort()[::-1]
        top_k = 3
        return [self.chunked_docs[i] for i in sorted_idx[:top_k]]


    def embed_query(self, query: str):
        return self.embeddings.embed_query(query)


    def get_documents(self, query: str):
        docs = ArxivLoader(query=query, load_max_docs=3).load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=350, chunk_overlap=50)
        return text_splitter.split_documents(docs)  # return chunked docs


    def embed_documents(self, chunked_docs):
        doc_texts = [doc.page_content for doc in chunked_docs]
        return self.embeddings.embed_documents(doc_texts)
