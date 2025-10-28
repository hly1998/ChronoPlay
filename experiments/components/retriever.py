# -*- coding: utf-8 -*-
"""
向量检索器组件
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import time

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from openai import OpenAI

from .config import RAGConfig, RetrievalConfig
from .embedding_service import create_embedding_service


class VectorRetriever:
    """简化的向量检索器"""

    def __init__(self, config):
        self.config = config
        self.index = None
        self.retriever = None

        # 支持两种配置类型
        if isinstance(config, RetrievalConfig):
            self._init_with_retrieval_config(config)
        else:
            self._init_with_rag_config(config)

    def _init_with_retrieval_config(self, config: RetrievalConfig):
        """使用RetrievalConfig初始化"""
        # 创建嵌入服务
        self.embedding_service = create_embedding_service(config)

        # 配置LlamaIndex的嵌入模型 - 统一使用 OpenAI 兼容接口
        Settings.embed_model = OpenAIEmbedding(
            api_key=config.api_key,
            api_base=config.base_url,
            model=config.embedding_model
        )

    def _init_with_rag_config(self, config: RAGConfig):
        """使用RAGConfig初始化（保持向后兼容）"""
        # 配置嵌入模型
        Settings.embed_model = OpenAIEmbedding(
            api_key=config.api_key,
            api_base=config.base_url,
            model=config.embedding_model
        )

        # 用于批量embeddings的OpenAI客户端
        self.openai_client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    def build_index(self, documents: List[Document]) -> bool:
        """构建或加载向量索引"""
        # 生成索引路径
        index_name = self._get_index_name()
        index_path = Path(self.config.index_dir) / self.config.game_name / index_name

        # 尝试加载已有索引
        if index_path.exists() and not self.config.force_rebuild:
            if self._load_existing_index(index_path):
                return True

        # 构建新索引
        return self._build_new_index(documents, index_path)

    def _get_index_name(self) -> str:
        """生成索引名称"""
        model_name = self.config.embedding_model.replace('-', '_').replace('/', '_')

        if self.config.target_segment_id:
            suffix = "_with_timeless" if self.config.include_timeless else ""
            return f"{model_name}/segment_{self.config.target_segment_id}{suffix}"
        return f"{model_name}/full_corpus"

    def _load_existing_index(self, index_path: Path) -> bool:
        """加载已有索引"""
        try:
            if self.config.verbose:
                print(f"📂 加载已有索引: {index_path.name}")

            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
            self.index = load_index_from_storage(storage_context)
            self.retriever = self.index.as_retriever(similarity_top_k=self.config.top_k)

            if self.config.verbose:
                print("✅ 索引加载成功")
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ 索引加载失败: {e}")
            return False

    def _build_new_index(self, documents: List[Document], index_path: Path) -> bool:
        """构建新索引"""
        try:
            if self.config.verbose:
                print(f"🔨 构建新索引，文档数量: {len(documents)}")

            self.index = VectorStoreIndex.from_documents(documents, show_progress=self.config.verbose)

            # 保存索引
            index_path.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(index_path))

            self.retriever = self.index.as_retriever(similarity_top_k=self.config.top_k)

            if self.config.verbose:
                print(f"✅ 索引构建完成: {index_path}")
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"❌ 索引构建失败: {e}")
            return False

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if not self.retriever:
            raise ValueError("检索器未初始化")

        try:
            nodes = self.retriever.retrieve(query)
            return [{
                'id': node.id_,
                'content': node.get_content(),
                'metadata': node.metadata,
                'score': getattr(node, 'score', 0.0)
            } for node in nodes]
        except Exception as e:
            if self.config.verbose:
                print(f"❌ 检索失败: {e}")
            return []

    def batch_get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        批量获取文本的embedding向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            embedding矩阵
        """
        # 如果是RetrievalConfig，使用新的嵌入服务
        if isinstance(self.config, RetrievalConfig) and hasattr(self, 'embedding_service'):
            return self.embedding_service.get_embeddings(texts, batch_size)

        # 向后兼容：使用原有的OpenAI客户端
        embeddings = []

        progress_bar = tqdm(range(0, len(texts), batch_size), desc="Embedding", disable=not self.config.verbose)

        for i in progress_bar:
            batch_texts = texts[i:i + batch_size]

            try:
                response = self.openai_client.embeddings.create(
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
                embeddings.extend([[0.0] * 1536] * len(batch_texts))

        return np.array(embeddings)

    def batch_retrieve(self, queries: List[str]) -> List[List[Dict[str, Any]]]:
        """
        批量检索相关文档

        Args:
            queries: 查询列表

        Returns:
            每个查询对应的检索结果列表
        """
        if not self.index:
            raise ValueError("索引未初始化")

        try:
            # 批量获取查询的embeddings
            query_embeddings = self.batch_get_embeddings(queries)

            # 执行批量检索
            all_results = []

            for i, (query, query_embedding) in enumerate(zip(queries, query_embeddings)):

                # 直接使用已计算的embedding向量进行检索，避免重复计算
                try:
                    # 直接使用vector_store查询，传入已计算的embedding
                    vector_store = self.index.vector_store

                    # 创建VectorStoreQuery对象
                    query_obj = VectorStoreQuery(
                        query_embedding=query_embedding.tolist(),
                        similarity_top_k=self.config.top_k
                    )

                    # 执行相似度搜索
                    query_result = vector_store.query(query_obj)

                    # 转换为标准格式
                    results = []
                    if hasattr(query_result, 'nodes') and query_result.nodes:
                        for node in query_result.nodes:
                            results.append({
                                'id': node.id_,
                                'content': node.get_content(),
                                'metadata': node.metadata,
                                'score': getattr(node, 'score', 0.0)
                            })
                    elif hasattr(query_result, 'similarities') and query_result.similarities:
                        # 如果没有nodes，使用similarities和ids
                        for j, (node_id, similarity) in enumerate(zip(query_result.ids, query_result.similarities)):
                            if j >= self.config.top_k:
                                break
                            # 从docstore获取节点信息
                            try:
                                node = self.index.docstore.get_node(node_id)
                                results.append({
                                    'id': node_id,
                                    'content': node.get_content(),
                                    'metadata': node.metadata,
                                    'score': float(similarity)
                                })
                            except Exception:
                                # 如果无法获取节点详情，至少返回基本信息
                                results.append({
                                    'id': node_id,
                                    'content': f"Document {node_id}",
                                    'metadata': {},
                                    'score': float(similarity)
                                })

                    all_results.append(results)

                except Exception:
                    # 返回空结果，避免触发单独的API调用
                    all_results.append([])

            return all_results

        except Exception as e:
            if self.config.verbose:
                print(f"❌ 批量检索失败，回退到单个检索: {e}")
            # 回退到单个检索
            return [self.retrieve(query) for query in queries]
