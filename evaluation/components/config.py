# -*- coding: utf-8 -*-
"""
RAG系统配置类
支持检索和生成的分离配置
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """RAG配置 - 支持检索和生成的分离配置"""
    # 基础配置
    game_name: str = "dyinglight2"
    target_segment_id: Optional[int] = None
    include_timeless: bool = True

    # 路径配置
    corpus_dir: str = "data"
    qa_data_dir: str = "data"
    index_dir: str = "evaluation/index"

    # 结果路径配置
    retrieval_results_dir: str = "evaluation/retrieval_results"
    generation_results_dir: str = "evaluation/generation_results"

    # API配置
    api_key: str = "{your-api-key}"
    base_url: str = "{your-base-url}"

    # 检索配置
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 3

    # 生成配置
    llm_model: str = "gpt-4o"
    temperature: float = 0.7

    # 系统配置
    force_rebuild: bool = False
    verbose: bool = True


@dataclass
class RetrievalConfig:
    """专用于检索的配置"""
    # 基础配置
    game_name: str = "dyinglight2"
    target_segment_id: Optional[int] = None
    include_timeless: bool = True

    # 路径配置
    corpus_dir: str = "data"
    qa_data_dir: str = "data"
    index_dir: str = "evaluation/index"
    output_dir: str = "evaluation/retrieval_results"

    # API配置 - 统一使用 OpenAI 兼容接口
    api_key: str = "{your-api-key}"
    base_url: str = "{your-base-url}"

    # 检索配置
    retrieval_method: str = "vector"  # "vector" 或 "bm25"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 3

    # BM25参数
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # 系统配置
    force_rebuild: bool = False
    verbose: bool = True

    @classmethod
    def from_rag_config(cls, rag_config: RAGConfig) -> 'RetrievalConfig':
        """从RAGConfig创建RetrievalConfig"""
        return cls(
            game_name=rag_config.game_name,
            target_segment_id=rag_config.target_segment_id,
            include_timeless=rag_config.include_timeless,
            corpus_dir=rag_config.corpus_dir,
            qa_data_dir=rag_config.qa_data_dir,
            index_dir=rag_config.index_dir,
            output_dir=rag_config.retrieval_results_dir,
            api_key=rag_config.api_key,
            base_url=rag_config.base_url,
            embedding_model=rag_config.embedding_model,
            top_k=rag_config.top_k,
            force_rebuild=rag_config.force_rebuild,
            verbose=rag_config.verbose
        )


@dataclass
class GenerationConfig:
    """专用于生成的配置"""
    # API配置
    api_key: str = "{your-api-key}"
    base_url: str = "{your-base-url}"

    # 生成配置
    llm_model: str = "gpt-4o"
    temperature: float = 0.7

    # 路径配置
    output_dir: str = "evaluation/generation_results"

    # 并发配置
    concurrent_requests: int = 3

    # 系统配置
    verbose: bool = True

    @classmethod
    def from_rag_config(cls, rag_config: RAGConfig) -> 'GenerationConfig':
        """从RAGConfig创建GenerationConfig"""
        return cls(
            api_key=rag_config.api_key,
            base_url=rag_config.base_url,
            llm_model=rag_config.llm_model,
            temperature=rag_config.temperature,
            output_dir=rag_config.generation_results_dir,
            verbose=rag_config.verbose
        )
