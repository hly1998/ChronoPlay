"""索引构建器 - 负责为分段构建和管理向量索引"""

import json
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.retrievers import VectorIndexRetriever


class IndexBuilder:
    """索引构建器类"""

    def __init__(self, corpus_path: Path, output_dir: Path, similarity_top_k: int = 3):
        """
        初始化索引构建器

        Args:
            corpus_path: 语料库路径
            output_dir: 输出目录
            similarity_top_k: 检索时返回的相似文档数量
        """
        self.corpus_path = corpus_path
        self.output_dir = output_dir
        self.similarity_top_k = similarity_top_k
        self.index = None
        self.retriever = None

    def load_corpus_for_segment(self, segment_id: int) -> List[Document]:
        """
        为指定分段加载语料库数据

        Args:
            segment_id: 分段ID

        Returns:
            Document对象列表
        """
        documents = []

        # 加载timeless数据
        timeless_corpus = self.corpus_path / "segment_timeless" / "corpus.jsonl"
        if timeless_corpus.exists():
            documents.extend(self._load_corpus_from_file(timeless_corpus, "timeless"))
            print(f"📚 加载timeless数据: {len(documents)} 个文档")

        # 加载指定分段数据
        segment_corpus = self.corpus_path / f"segment_{segment_id}" / "corpus.jsonl"
        if segment_corpus.exists():
            segment_docs = self._load_corpus_from_file(segment_corpus, f"segment_{segment_id}")
            documents.extend(segment_docs)
            print(f"📚 加载分段{segment_id}数据: {len(segment_docs)} 个文档")

        print(f"📚 分段 {segment_id} 语料加载完成: 总计 {len(documents)} 个文档")
        return documents

    def _load_corpus_from_file(self, file_path: Path, source: str) -> List[Document]:
        """从文件加载语料库数据"""
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())

                        # 构建基础metadata
                        processed_metadata = {
                            'id': data.get('id', f'{source}_doc_{line_num}'),
                            'source': source,
                        }

                        # 处理corpus文件中的entities字段（直接在根级别）
                        if 'entities' in data and isinstance(data['entities'], list):
                            # corpus文件中的entities是字符串列表，直接作为entity_texts
                            entity_texts = [entity.strip() for entity in data['entities'] if entity and entity.strip()]
                            processed_metadata['entity_texts'] = entity_texts

                        # 复制其他有用的字段到metadata
                        for key in ['title', 'segment_id', 'extracted_date']:
                            if key in data:
                                processed_metadata[key] = data[key]

                        # 如果存在metadata字段，也复制进来
                        if 'metadata' in data and isinstance(data['metadata'], dict):
                            for key, value in data['metadata'].items():
                                processed_metadata[key] = value

                        doc = Document(
                            text=data.get('contents', data.get('text', '')),
                            metadata=processed_metadata
                        )
                        doc.id_ = data.get('id', f'{source}_doc_{line_num}')
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        print(f"第{line_num}行JSON解析失败: {e}")
                        continue
        return documents

    def build_segment_index(self, segment_id: int, force_rebuild: bool = False) -> bool:
        """
        为指定分段构建索引

        Args:
            segment_id: 分段ID
            force_rebuild: 是否强制重建索引

        Returns:
            是否成功
        """
        print(f"=== 为分段 {segment_id} 构建索引 ===")

        # 设置分段专用的索引目录
        segment_index_dir = self.output_dir / f"segment_{segment_id}_index"

        # 检查是否需要重建索引
        if force_rebuild and segment_index_dir.exists():
            print("🔄 强制重建模式：删除现有索引目录...")
            import shutil
            shutil.rmtree(segment_index_dir)

        # 检查是否已有索引
        if not force_rebuild and segment_index_dir.exists():
            try:
                print(f"正在加载分段 {segment_id} 的已有索引...")
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(segment_index_dir))
                self.index = load_index_from_storage(storage_context)
                print(f"✅ 分段 {segment_id} 索引加载成功")

                # 创建检索器
                self.retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=self.similarity_top_k
                )
                return True

            except Exception as e:
                print(f"加载分段 {segment_id} 索引失败: {e}, 将重新构建索引")

        # 加载语料库数据
        documents = self.load_corpus_for_segment(segment_id)

        if not documents:
            print(f"❌ 分段 {segment_id} 没有语料库数据")
            return False

        try:
            # 构建向量索引
            print(f"正在为分段 {segment_id} 构建新索引...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )

            # 保存索引
            self.index.storage_context.persist(persist_dir=str(segment_index_dir))
            print(f"✅ 分段 {segment_id} 索引构建完成，保存到: {segment_index_dir}")

            # 创建检索器
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.similarity_top_k
            )

            return True

        except Exception as e:
            print(f"❌ 构建分段 {segment_id} 索引失败: {e}")
            return False

    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if not self.retriever:
            raise ValueError("检索器未初始化，请先构建索引")

        nodes = self.retriever.retrieve(query)
        documents = []
        for node in nodes:
            doc_data = {
                'id': node.id_,
                'content': node.get_content(),
                'metadata': node.metadata,
                'score': getattr(node, 'score', 0.0)
            }
            documents.append(doc_data)

        return documents

    def cleanup_index(self, segment_id: int):
        """清理索引文件"""
        segment_index_dir = self.output_dir / f"segment_{segment_id}_index"
        if segment_index_dir.exists():
            import shutil
            shutil.rmtree(segment_index_dir)
            print(f"✅ 删除索引目录: {segment_index_dir}")
