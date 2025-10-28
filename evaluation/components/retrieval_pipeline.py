# -*- coding: utf-8 -*-
"""
简化版检索Pipeline
专门为单游戏整体评估设计
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
    """简化的检索Pipeline - 支持多游戏混合检索"""

    def __init__(self, config):
        self.config = config
        self.game_retrievers = {}  # 每个游戏独立的检索器
        self.game_documents = {}   # 每个游戏的文档

    def load_documents_for_game(self, game_name: str) -> bool:
        """加载指定游戏的文档"""
        corpus_dir = Path(self.config.corpus_dir) / game_name / "corpus"
        if not corpus_dir.exists():
            if self.config.verbose:
                print(f"  ⚠️ 游戏 {game_name} 的语料库目录不存在")
            return False

        # 加载所有分段（包括timeless）
        segment_dirs = self._get_all_segment_dirs(corpus_dir)

        # 加载文档
        documents = []
        for segment_dir in segment_dirs:
            self._load_segment_documents(segment_dir, game_name, documents)

        self.game_documents[game_name] = documents
        return len(documents) > 0

    def _get_all_segment_dirs(self, corpus_dir: Path) -> List[Path]:
        """获取所有分段目录"""
        dirs = [d for d in corpus_dir.iterdir() if d.is_dir() and d.name.startswith('segment_')]

        # 加载 timeless 分段
        if self.config.include_timeless:
            timeless_dir = corpus_dir / "segment_timeless"
            if timeless_dir.exists() and timeless_dir not in dirs:
                dirs.append(timeless_dir)

        return sorted(dirs)

    def _load_segment_documents(self, segment_dir: Path, game_name: str, documents: List):
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

                    # 创建文档
                    original_metadata = data.get('metadata', {})
                    metadata = {
                        'id': data.get('id', f'{segment_dir.name}_doc_{line_num}'),
                        'title': data.get('title', ''),
                        'source': data.get('source', ''),
                        'game': game_name,
                        'segment_id': segment_dir.name,
                    }

                    # 处理原始metadata
                    for key, value in original_metadata.items():
                        if key == 'entities' and isinstance(value, list):
                            entity_texts = []
                            for entity in value:
                                if isinstance(entity, dict) and entity.get('text'):
                                    entity_texts.append(entity.get('text', ''))
                            metadata['entity_texts'] = entity_texts
                        else:
                            metadata[key] = value

                    doc = Document(text=data.get('contents', ''), metadata=metadata)
                    doc.id_ = metadata['id']
                    documents.append(doc)

                except json.JSONDecodeError as e:
                    if self.config.verbose:
                        print(f"⚠️ {game_name}/{segment_dir.name}第{line_num}行解析失败: {e}")

    def initialize_for_games(self, game_names: List[str]) -> bool:
        """为多个游戏初始化检索器"""
        try:
            total_docs = 0
            for game_name in game_names:
                # 加载文档
                if not self.load_documents_for_game(game_name):
                    continue

                documents = self.game_documents[game_name]
                total_docs += len(documents)

                # 为每个游戏创建独立的检索器
                if hasattr(self.config, 'retrieval_method') and self.config.retrieval_method == 'bm25':

                    retriever = BM25Retriever(self.config, k1=self.config.bm25_k1, b=self.config.bm25_b)
                else:

                    retriever = VectorRetriever(self.config)

                # 临时修改config的game_name以构建正确的索引路径
                original_game = self.config.game_name
                self.config.game_name = game_name

                # 构建索引
                if not retriever.build_index(documents):
                    if self.config.verbose:
                        print(f"  ⚠️ 游戏 {game_name} 索引构建失败")
                    self.config.game_name = original_game
                    continue

                self.config.game_name = original_game
                self.game_retrievers[game_name] = retriever

            if self.config.verbose:
                print(f"  加载文档: {total_docs} 个 (跨 {len(self.game_retrievers)} 个游戏)")

            return len(self.game_retrievers) > 0
        except Exception as e:
            if self.config.verbose:
                print(f"❌ 初始化失败: {e}")
            return False

    def retrieve_single(self, question: str, game_name: str) -> Dict[str, Any]:
        """执行单个问题的检索"""
        start_time = time.time()

        # 根据游戏名获取对应的检索器
        if game_name not in self.game_retrievers:
            return {
                'question': question,
                'retrieved_docs': [],
                'retrieved_doc_ids': [],
                'retrieval_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'error': f'游戏 {game_name} 的检索器未初始化',
                'retrieval_config': self._get_retrieval_config()
            }

        retriever = self.game_retrievers[game_name]

        # 检索文档
        docs = retriever.retrieve(question)

        # 构建检索结果
        result = {
            'question': question,
            'game_name': game_name,
            'retrieved_docs': docs,
            'retrieved_doc_ids': [doc.get('id', '') for doc in docs],
            'retrieval_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'retrieval_config': self._get_retrieval_config()
        }

        return result

    def batch_retrieve_qa_pairs(self, qa_pairs: List[Dict[str, Any]], output_file: str = None) -> List[Dict[str, Any]]:
        """批量检索QA对 - 支持多游戏混合"""
        total = len(qa_pairs)
        results = []
        start_time = time.time()

        # 按游戏分组
        game_qa_groups = {}
        for i, qa_pair in enumerate(qa_pairs):
            game_name = qa_pair.get('original_data', {}).get('game_name', 'unknown')
            if game_name not in game_qa_groups:
                game_qa_groups[game_name] = []
            game_qa_groups[game_name].append((i, qa_pair))

        # 对每个游戏进行批量检索
        all_results = [None] * total  # 保持原始顺序

        for game_name, game_qa_list in game_qa_groups.items():
            if game_name not in self.game_retrievers:
                # 游戏检索器不存在
                for idx, qa_pair in game_qa_list:
                    error_result = {
                        'question': qa_pair['question'],
                        'game_name': game_name,
                        'question_index': idx,
                        'retrieved_docs': [],
                        'retrieved_doc_ids': [],
                        'error': f'游戏 {game_name} 的检索器未初始化',
                        'timestamp': datetime.now().isoformat(),
                        'ground_truth_answer': qa_pair['ground_truth_answer'],
                        'ground_truth_doc_ids': qa_pair['ground_truth_doc_ids'],
                        'ground_truth_docs': qa_pair['ground_truth_docs'],
                        'original_qa_data': qa_pair.get('original_data', {})
                    }
                    all_results[idx] = error_result
                continue

            retriever = self.game_retrievers[game_name]
            questions = [qa_pair['question'] for _, qa_pair in game_qa_list]

            try:
                # 批量检索
                batch_docs = retriever.batch_retrieve(questions)

                # 构建结果
                for (idx, qa_pair), docs in zip(game_qa_list, batch_docs):
                    result = {
                        'question': qa_pair['question'],
                        'game_name': game_name,
                        'retrieved_docs': docs,
                        'retrieved_doc_ids': [doc.get('id', '') for doc in docs],
                        'retrieval_time': 0,  # 稍后计算
                        'timestamp': datetime.now().isoformat(),
                        'question_index': idx,
                        'ground_truth_answer': qa_pair['ground_truth_answer'],
                        'ground_truth_doc_ids': qa_pair['ground_truth_doc_ids'],
                        'ground_truth_docs': qa_pair['ground_truth_docs'],
                        'original_qa_data': qa_pair.get('original_data', {}),
                        'retrieval_config': self._get_retrieval_config()
                    }
                    all_results[idx] = result

            except Exception:
                # 批量失败，逐个检索
                for idx, qa_pair in game_qa_list:
                    try:
                        result = self.retrieve_single(qa_pair['question'], game_name)
                        result.update({
                            'question_index': idx,
                            'ground_truth_answer': qa_pair['ground_truth_answer'],
                            'ground_truth_doc_ids': qa_pair['ground_truth_doc_ids'],
                            'ground_truth_docs': qa_pair['ground_truth_docs'],
                            'original_qa_data': qa_pair.get('original_data', {})
                        })
                        all_results[idx] = result
                    except Exception as e2:
                        error_result = {
                            'question': qa_pair['question'],
                            'game_name': game_name,
                            'question_index': idx,
                            'retrieved_docs': [],
                            'retrieved_doc_ids': [],
                            'error': str(e2),
                            'timestamp': datetime.now().isoformat(),
                            'ground_truth_answer': qa_pair['ground_truth_answer'],
                            'ground_truth_doc_ids': qa_pair['ground_truth_doc_ids'],
                            'ground_truth_docs': qa_pair['ground_truth_docs'],
                            'original_qa_data': qa_pair.get('original_data', {})
                        }
                        all_results[idx] = error_result

        # 计算平均时间并保存结果
        batch_time = time.time() - start_time
        avg_time = batch_time / total

        for result in all_results:
            if result:
                result['retrieval_time'] = avg_time
                results.append(result)

                # 保存到文件
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

        if self.config.verbose:
            print(f"  批量检索完成，耗时 {batch_time:.2f}s")

        return results

    def _get_retrieval_config(self) -> Dict[str, Any]:
        """获取检索配置信息"""
        config = {
            'top_k': self.config.top_k,
            'include_timeless': self.config.include_timeless,
            'multi_game': True
        }

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
