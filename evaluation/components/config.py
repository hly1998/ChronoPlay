# -*- coding: utf-8 -*-
"""
简化版评估配置
"""

from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 基础配置
    game_name: str = "dune"

    # 路径配置
    corpus_dir: str = "data"
    csv_data_path: str = "data/data.csv"
    index_dir: str = "evaluation/index"
    output_dir: str = "evaluation/retrieval_results"

    # API配置
    api_key: str = "{your-api-key}"
    base_url: str = "{your-base-url}"

    # 检索配置
    retrieval_method: str = "vector"  # "vector" 或 "bm25"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5

    # BM25参数
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # 系统配置
    force_rebuild: bool = False
    include_timeless: bool = True
    verbose: bool = True


@dataclass
class GenerationConfig:
    """生成配置"""
    # API配置
    api_key: str = "{your-api-key}"
    base_url: str = "{your-base-url}"

    # 生成配置
    llm_model: str = "gpt-4o"
    temperature: float = 0.1

    # 路径配置
    output_dir: str = "evaluation/generation_results"

    # 并发配置
    concurrent_requests: int = 3

    # 系统配置
    verbose: bool = True
