# -*- coding: utf-8 -*-
"""
独立检索Pipeline
将检索过程与生成过程解耦，支持一次检索，多次生成的实验模式
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from llama_index.core import Document

from .retriever import VectorRetriever
from .bm25_retriever import BM25Retriever


class RetrievalPipeline:
    """独立的检索Pipeline"""

    def __init__(self, config):
        self.config = config
        self.documents = []

        # 根据检索方法选择检索器
        if hasattr(config, 'retrieval_method') and config.retrieval_method == 'bm25':
            if hasattr(config, 'bm25_k1') and hasattr(config, 'bm25_b'):
                self.retriever = BM25Retriever(config, k1=config.bm25_k1, b=config.bm25_b)
            else:
                self.retriever = BM25Retriever(config)
        else:
            self.retriever = VectorRetriever(config)

    def load_documents(self) -> bool:
        """加载文档"""
        corpus_dir = Path(self.config.corpus_dir) / self.config.game_name / "corpus"
        if not corpus_dir.exists():
            if self.config.verbose:
                print(f"❌ 语料库目录不存在: {corpus_dir}")
            return False

        # 确定要加载的分段目录
        segment_dirs = self._get_segment_dirs(corpus_dir)

        # 加载文档
        self.documents = []
        for segment_dir in segment_dirs:
            self._load_segment_documents(segment_dir)

        if self.config.verbose:
            self._print_document_stats()

        return len(self.documents) > 0

    def _get_segment_dirs(self, corpus_dir: Path) -> List[Path]:
        """获取要加载的分段目录"""
        if self.config.target_segment_id:
            # 加载指定分段
            dirs = [corpus_dir / f"segment_{self.config.target_segment_id}"]
            if self.config.include_timeless:
                timeless_dir = corpus_dir / "segment_timeless"
                if timeless_dir.exists():
                    dirs.append(timeless_dir)
        else:
            # 加载所有分段
            dirs = [d for d in corpus_dir.iterdir() if d.is_dir() and d.name.startswith('segment_')]
            if self.config.include_timeless:
                timeless_dir = corpus_dir / "segment_timeless"
                if timeless_dir.exists() and timeless_dir not in dirs:
                    dirs.append(timeless_dir)
        return dirs

    def _load_segment_documents(self, segment_dir: Path):
        """加载单个分段的文档"""
        corpus_file = segment_dir / "corpus.jsonl"
        if not corpus_file.exists():
            return

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line.strip())

                    # 创建文档 - 处理metadata以避免过长问题
                    original_metadata = data.get('metadata', {})
                    metadata = {
                        'id': data.get('id', f'{segment_dir.name}_doc_{line_num}'),
                        'title': data.get('title', ''),
                        'source': data.get('source', ''),
                        'game': self.config.game_name,
                        'segment_id': segment_dir.name,
                    }

                    # 处理原始metadata，压缩entities信息
                    for key, value in original_metadata.items():
                        if key == 'entities' and isinstance(value, list):
                            # 只保留entities的text字段，大幅减少metadata长度
                            entity_texts = []
                            for entity in value:
                                if isinstance(entity, dict) and entity.get('text'):
                                    entity_texts.append(entity.get('text', ''))
                            metadata['entity_texts'] = entity_texts
                        else:
                            metadata[key] = value

                    doc = Document(text=data.get('contents', ''), metadata=metadata)
                    doc.id_ = metadata['id']
                    self.documents.append(doc)

                except json.JSONDecodeError as e:
                    if self.config.verbose:
                        print(f"⚠️ {segment_dir.name}第{line_num}行解析失败: {e}")

    def _print_document_stats(self):
        """打印文档统计信息"""
        segment_stats = {}
        for doc in self.documents:
            segment = doc.metadata.get('segment_id', 'unknown')
            segment_stats[segment] = segment_stats.get(segment, 0) + 1

        print(f"✅ 加载 {len(self.documents)} 个文档")

    def initialize(self) -> bool:
        """初始化检索pipeline"""
        try:
            # 加载文档
            if not self.load_documents():
                return False

            # 构建索引
            if not self.retriever.build_index(self.documents):
                return False

            return True
        except Exception as e:
            if self.config.verbose:
                print(f"❌ 初始化失败: {e}")
            return False

    def retrieve_single(self, question: str) -> Dict[str, Any]:
        """执行单个问题的检索"""
        start_time = time.time()

        if self.config.verbose:
            print(f"🔍 检索问题: {question}")

        # 检索文档
        docs = self.retriever.retrieve(question)

        if self.config.verbose:
            print(f"📄 检索到 {len(docs)} 个文档")
            for i, doc in enumerate(docs, 1):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Unknown')
                segment = metadata.get('segment_id', 'Unknown')
                score = doc.get('score', 0.0)
                print(f"   {i}. {title} ({segment}, {score:.3f})")

        # 构建检索结果
        result = {
            'question': question,
            'retrieved_docs': docs,
            'retrieved_doc_ids': [doc.get('id', '') for doc in docs],
            'retrieval_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'retrieval_config': self._get_retrieval_config()
        }

        if self.config.verbose:
            print(f"⏱️ 检索耗时: {result['retrieval_time']:.2f}秒")

        return result

    def batch_retrieve(self, questions: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """批量检索（优化版本：一次性获取所有embeddings）"""
        total = len(questions)
        results = []

        try:
            # 使用批量检索方法
            start_time = time.time()
            batch_results = self.retriever.batch_retrieve(questions)
            batch_time = time.time() - start_time

            if self.config.verbose:
                print(f"✅ 批量检索完成: {total} 个问题, 耗时 {batch_time:.2f}s")

            # 构建详细结果
            for i, (question, docs) in enumerate(zip(questions, batch_results)):

                result = {
                    'question': question,
                    'retrieved_docs': docs,
                    'retrieved_doc_ids': [doc.get('id', '') for doc in docs],
                    'retrieval_time': batch_time / total,  # 分摊时间
                    'timestamp': datetime.now().isoformat(),
                    'question_index': i,
                    'retrieval_config': {
                        **self._get_retrieval_config(),
                        'batch_mode': True
                    }
                }

                results.append(result)

                # 保存到文件
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ 批量检索失败，回退到逐个检索: {e}")

            # 回退到原来的逐个检索方式
            for i, question in enumerate(questions):

                try:
                    result = self.retrieve_single(question)
                    result['question_index'] = i
                    results.append(result)

                    # 保存到文件
                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')

                except Exception as e:
                    if self.config.verbose:
                        print(f"❌ 检索失败: {e}")
                    error_result = {
                        'question': question,
                        'question_index': i,
                        'retrieved_docs': [],
                        'retrieved_doc_ids': [],
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)

                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + '\n')

        if self.config.verbose:
            print(f"\n✅ 批量检索完成，处理 {len(results)} 个问题")

        return results

    def batch_retrieve_qa_pairs(self, qa_pairs: List[Dict[str, Any]], output_file: str = None) -> List[Dict[str, Any]]:
        """批量检索QA对（包含标准答案信息，便于后续评估）（优化版本：一次性获取所有embeddings）"""
        total = len(qa_pairs)
        results = []

        try:
            # 提取所有问题
            questions = [qa_pair['question'] for qa_pair in qa_pairs]

            # 使用批量检索方法
            start_time = time.time()
            batch_results = self.retriever.batch_retrieve(questions)
            batch_time = time.time() - start_time

            if self.config.verbose:
                print(f"✅ 批量检索完成: {total} 个QA对, 耗时 {batch_time:.2f}s")

            # 构建详细结果
            for i, (qa_pair, docs) in enumerate(zip(qa_pairs, batch_results)):

                question = qa_pair['question']

                result = {
                    'question': question,
                    'retrieved_docs': docs,
                    'retrieved_doc_ids': [doc.get('id', '') for doc in docs],
                    'retrieval_time': batch_time / total,  # 分摊时间
                    'timestamp': datetime.now().isoformat(),
                    'question_index': i,
                    'ground_truth_answer': qa_pair['ground_truth_answer'],
                    'ground_truth_doc_ids': qa_pair['ground_truth_doc_ids'],
                    'ground_truth_docs': qa_pair['ground_truth_docs'],
                    'original_qa_data': qa_pair.get('original_data', {}),
                    'retrieval_config': {
                        **self._get_retrieval_config(),
                        'batch_mode': True
                    }
                }

                results.append(result)

                # 保存到文件
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ 批量检索失败，回退到逐个检索: {e}")

            # 回退到原来的逐个检索方式
            for i, qa_pair in enumerate(qa_pairs):
                question = qa_pair['question']

                try:
                    # 执行检索
                    result = self.retrieve_single(question)

                    # 添加QA对相关信息（便于后续生成和评估）
                    result.update({
                        'question_index': i,
                        'ground_truth_answer': qa_pair['ground_truth_answer'],
                        'ground_truth_doc_ids': qa_pair['ground_truth_doc_ids'],
                        'ground_truth_docs': qa_pair['ground_truth_docs'],
                        'original_qa_data': qa_pair.get('original_data', {})
                    })

                    results.append(result)

                    # 保存到文件
                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')

                except Exception as e:
                    if self.config.verbose:
                        print(f"❌ 检索失败: {e}")
                    error_result = {
                        'question': question,
                        'question_index': i,
                        'retrieved_docs': [],
                        'retrieved_doc_ids': [],
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'ground_truth_answer': qa_pair['ground_truth_answer'],
                        'ground_truth_doc_ids': qa_pair['ground_truth_doc_ids']
                    }
                    results.append(error_result)

                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + '\n')

        return results

    def load_qa_pairs(self, segment_id: int = None) -> List[Dict[str, Any]]:
        """加载QA问答对（从生成的数据中提取）"""
        if segment_id is None:
            segment_id = self.config.target_segment_id

        if segment_id is None:
            raise ValueError("必须指定分段ID")

        # 使用新的 QA 数据路径: data/{游戏名称}/segments/segment_{id}
        qa_dir = Path(self.config.qa_data_dir) / self.config.game_name / "segments" / f"segment_{segment_id}"
        qa_filename = "generated_qa_pairs.jsonl"
        qa_file = qa_dir / qa_filename

        if not qa_file.exists():
            error_msg = f"❌ QA数据文件不存在: {qa_file}"
            if self.config.verbose:
                print(error_msg)
            raise FileNotFoundError(error_msg)

        qa_pairs = []
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            question = data.get('question', '').strip()
                            if question:
                                # 提取标准答案和标准文档ID
                                ground_truth_answer = data.get('answer', '')
                                ground_truth_docs = data.get('retrieved_docs', [])
                                ground_truth_doc_ids = [doc.get('id', '') for doc in ground_truth_docs]

                                qa_pairs.append({
                                    'question': question,
                                    'ground_truth_answer': ground_truth_answer,
                                    'ground_truth_doc_ids': ground_truth_doc_ids,
                                    'ground_truth_docs': ground_truth_docs,
                                    'original_data': data  # 保留完整原始数据
                                })
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            if self.config.verbose:
                print(f"❌ 加载QA数据失败: {e}")

        return qa_pairs

    def _get_retrieval_config(self) -> Dict[str, Any]:
        """获取检索配置信息"""
        config = {
            'game': self.config.game_name,
            'segment_id': self.config.target_segment_id,
            'top_k': self.config.top_k,
            'include_timeless': self.config.include_timeless
        }

        # 根据检索方法添加不同的配置信息
        if hasattr(self.config, 'retrieval_method') and self.config.retrieval_method == 'bm25':
            config.update({
                'retrieval_method': 'bm25',
                'bm25_k1': getattr(self.config, 'bm25_k1', 1.2),
                'bm25_b': getattr(self.config, 'bm25_b', 0.75)
            })
        else:
            config.update({
                'retrieval_method': 'vector',
                'embedding_model': getattr(self.config, 'embedding_model', 'unknown')
            })

        return config
