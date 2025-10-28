# -*- coding: utf-8 -*-
"""
RAG Pipeline主组件
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from llama_index.core import Document

from .config import RAGConfig
from .retriever import VectorRetriever
from .generator import TextGenerator


class RAGPipeline:
    """简化的RAG Pipeline"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = VectorRetriever(config)
        self.generator = TextGenerator(config)
        self.documents = []

        if config.verbose:
            print("🚀 初始化RAG Pipeline")
            print(f"   游戏: {config.game_name}")
            if config.target_segment_id:
                print(f"   目标分段: {config.target_segment_id}")
            print(f"   包含时间无关数据: {'是' if config.include_timeless else '否'}")

    def load_documents(self) -> bool:
        """加载文档"""
        if self.config.verbose:
            print("📚 加载语料库...")

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

        if self.config.verbose:
            print(f"📄 加载: {segment_dir.name}")

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

        print(f"✅ 加载文档总数: {len(self.documents)}")
        print("📊 分段统计:")
        for segment, count in sorted(segment_stats.items()):
            label = " (时间无关)" if segment == 'segment_timeless' or segment == -1 else ""
            print(f"   {segment}: {count}{label}")

        # 时间无关数据状态提示
        has_timeless = 'segment_timeless' in segment_stats or -1 in segment_stats
        if self.config.include_timeless and has_timeless:
            print("🕐 已包含时间无关数据")
        elif self.config.include_timeless and not has_timeless:
            print("⚠️ 未找到时间无关数据")
        elif not self.config.include_timeless and has_timeless:
            print("ℹ️ 意外加载了时间无关数据")

    def initialize(self) -> bool:
        """初始化pipeline"""
        try:
            # 加载文档
            if not self.load_documents():
                return False

            # 构建索引
            if not self.retriever.build_index(self.documents):
                return False

            if self.config.verbose:
                print("✅ RAG Pipeline初始化完成")
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"❌ 初始化失败: {e}")
            return False

    def query(self, question: str) -> Dict[str, Any]:
        """执行查询"""
        start_time = time.time()

        if self.config.verbose:
            print(f"🔍 查询: {question}")

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

        # 生成回答
        answer = self.generator.generate(question, docs)

        # 构建结果
        result = {
            'question': question,
            'answer': answer,
            'retrieved_docs': docs,
            'retrieved_doc_ids': [doc.get('id', '') for doc in docs],
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'game': self.config.game_name,
                'embedding_model': self.config.embedding_model,
                'llm_model': self.config.llm_model,
                'segment_id': self.config.target_segment_id
            }
        }

        if self.config.verbose:
            print(f"💡 回答: {answer}")
            print(f"⏱️ 耗时: {result['processing_time']:.2f}秒")

        return result

    def batch_query(self, questions: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """批量查询"""
        total = len(questions)
        results = []

        if self.config.verbose:
            print(f"📝 批量查询开始，共 {total} 个问题")

        for i, question in enumerate(questions):
            if self.config.verbose:
                print(f"\n--- 问题 {i+1}/{total} ---")
                print(f"问题: {question[:60]}...")

            try:
                result = self.query(question)
                result['question_index'] = i
                results.append(result)

                # 保存到文件
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                if self.config.verbose:
                    print(f"❌ 查询失败: {e}")
                error_result = {
                    'question': question,
                    'question_index': i,
                    'answer': f"查询失败: {str(e)}",
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)

                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + '\n')

        if self.config.verbose:
            print(f"\n✅ 批量查询完成，处理 {len(results)} 个问题")

        return results

    def batch_evaluate(self, qa_pairs: List[Dict[str, Any]], output_file: str = None) -> List[Dict[str, Any]]:
        """批量评估（使用QA对，包含标准答案和文档ID）"""
        total = len(qa_pairs)
        results = []

        if self.config.verbose:
            print(f"📝 批量评估开始，共 {total} 个问答对")

        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair['question']
            ground_truth_answer = qa_pair['ground_truth_answer']
            ground_truth_doc_ids = qa_pair['ground_truth_doc_ids']

            if self.config.verbose:
                print(f"\n--- 问答对 {i+1}/{total} ---")
                print(f"问题: {question[:60]}...")

            try:
                # 执行RAG查询
                result = self.query(question)

                # 添加评估相关信息
                result.update({
                    'question_index': i,
                    'ground_truth_answer': ground_truth_answer,
                    'ground_truth_doc_ids': ground_truth_doc_ids,
                    'ground_truth_docs': qa_pair['ground_truth_docs'],
                    # 保留原有的retrieved_doc_ids（RAG检索到的）
                    # result中已经包含了retrieved_doc_ids
                })

                results.append(result)

                # 保存到文件
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                if self.config.verbose:
                    print(f"❌ 评估失败: {e}")
                error_result = {
                    'question': question,
                    'question_index': i,
                    'answer': f"评估失败: {str(e)}",
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'ground_truth_answer': ground_truth_answer,
                    'ground_truth_doc_ids': ground_truth_doc_ids,
                    'retrieved_doc_ids': []
                }
                results.append(error_result)

                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + '\n')

        if self.config.verbose:
            print(f"\n✅ 批量评估完成，处理 {len(results)} 个问答对")

        return results

    def load_qa_pairs(self, segment_id: int = None) -> List[Dict[str, Any]]:
        """加载QA问答对（包含标准答案和标准文档ID）"""
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

        if self.config.verbose:
            print(f"📝 使用QA数据: {qa_file}")

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
                                    'original_data': data  # 保留完整原始数据用于其他分析
                                })
                        except json.JSONDecodeError:
                            continue

            if self.config.verbose:
                print(f"✅ 加载了 {len(qa_pairs)} 个QA问答对")

        except Exception as e:
            if self.config.verbose:
                print(f"❌ 加载QA数据失败: {e}")

        return qa_pairs

    def load_qa_questions(self, segment_id: int = None) -> List[str]:
        """加载QA问题（向后兼容）"""
        qa_pairs = self.load_qa_pairs(segment_id)
        return [pair['question'] for pair in qa_pairs]
