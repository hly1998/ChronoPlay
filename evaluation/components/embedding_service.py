# -*- coding: utf-8 -*-
"""
嵌入服务 - 统一使用 OpenAI 兼容接口
"""

from typing import List
import time
import numpy as np
from tqdm import tqdm
from openai import OpenAI

from .config import RetrievalConfig


class EmbeddingService:
    """统一的嵌入服务 - 使用 OpenAI 兼容接口"""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        # 不同模型的维度映射
        self.model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "Pro/BAAI/bge-m3": 1024,
            "Qwen/Qwen3-Embedding-0.6B": 1024,
        }

    def get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """获取文本的embedding向量"""
        embeddings = []

        progress_bar = tqdm(range(0, len(texts), batch_size), desc="Embedding", disable=not self.config.verbose)

        for i in progress_bar:
            batch_texts = texts[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=batch_texts
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                # 添加延时避免触发速率限制
                time.sleep(0.05)

            except Exception as e:
                if self.config.verbose:
                    print(f"❌ 获取embedding失败: {e}")
                # 如果API调用失败，使用零向量作为占位符
                embeddings.extend([[0.0] * self.get_embedding_dim()] * len(batch_texts))

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """获取embedding向量维度"""
        return self.model_dims.get(self.config.embedding_model, 1536)


def create_embedding_service(config: RetrievalConfig) -> EmbeddingService:
    """创建嵌入服务实例"""
    return EmbeddingService(config)
