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
        # Try GPU first, fallback to CPU
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"[RAG] Using GPU (CUDA) for embeddings")
            else:
                device = 'cpu'
                print(f"[RAG] CUDA not available, using CPU for embeddings")
        except ImportError:
            device = 'cpu'
            print(f"[RAG] PyTorch not available, using CPU for embeddings")
        
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': False}
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDINGS_MODEL_NAME, 
                model_kwargs=model_kwargs, 
                encode_kwargs=encode_kwargs
            )
            print(f"[RAG] Successfully initialized embeddings on {device}")
        except Exception as e:
            # Fallback to CPU if GPU initialization fails
            if device != 'cpu':
                print(f"[RAG] GPU initialization failed ({e}), falling back to CPU")
                model_kwargs = {'device': 'cpu'}
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDINGS_MODEL_NAME, 
                    model_kwargs=model_kwargs, 
                    encode_kwargs=encode_kwargs
                )
                print(f"[RAG] Successfully initialized embeddings on CPU (fallback)")
            else:
                raise e
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
