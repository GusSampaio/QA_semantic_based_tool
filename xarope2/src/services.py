"""External service factories for MongoDB, Hugging Face, and Mistral."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.collection import Collection

from src.settings import AppSettings, missing_required_env


def _raise_if_missing_env(settings: AppSettings) -> None:
    missing = missing_required_env(settings)
    if missing:
        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing)
            + ". Add them to .env or Streamlit deployment secrets."
        )


def build_mongo_collection(settings: AppSettings) -> Collection:
    _raise_if_missing_env(settings)
    client = MongoClient(settings.mongodb_uri)
    return client[settings.mongodb_database][settings.mongodb_collection]


def build_embeddings(settings: AppSettings) -> HuggingFaceEndpointEmbeddings:
    _raise_if_missing_env(settings)
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=settings.hf_token,
        model=settings.embedding_model,
        task="feature-extraction",
    )


def build_llm(settings: AppSettings) -> ChatMistralAI:
    _raise_if_missing_env(settings)
    return ChatMistralAI(
        model=settings.chat_model,
        api_key=settings.mistral_api_key,
        temperature=0.2,
    )


def build_vector_store(
    *,
    collection: Collection,
    embeddings: HuggingFaceEndpointEmbeddings,
    settings: AppSettings,
) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=settings.mongodb_vector_index,
    )
