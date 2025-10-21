# -*- coding: utf-8 -*-
"""
BM25检索器组件
基于词频-逆文档频率的传统检索方法
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import math
import re
import jieba
from tqdm import tqdm

from llama_index.core import Document


class BM25Retriever:
    """BM25检索器实现"""

    def __init__(self, config, k1: float = 1.2, b: float = 0.75):
        """
        初始化BM25检索器

        Args:
            config: RAGConfig或RetrievalConfig
            k1: BM25参数k1，控制词频饱和度
            b: BM25参数b，控制文档长度归一化程度
        """
        self.config = config
        self.k1 = k1
        self.b = b

        # BM25计算相关
        self.documents = []
        self.doc_freqs = []  # 每个文档的词频字典
        self.idf = {}  # 词汇的IDF值
        self.doc_lens = []  # 每个文档的长度
        self.avgdl = 0.0  # 平均文档长度
        self.N = 0  # 文档总数

    def _tokenize(self, text: str) -> List[str]:
        """
        分词函数，支持中英文混合

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        if not text:
            return []

        # 清理文本
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        text = text.lower()

        # 使用jieba分词处理中文，保留英文单词
        tokens = []
        words = jieba.lcut(text)

        for word in words:
            word = word.strip()
            if len(word) > 1:  # 过滤单字符
                tokens.append(word)

        return tokens

    def _compute_idf(self):
        """计算所有词汇的IDF值"""
        df = defaultdict(int)  # 文档频率

        # 统计每个词出现在多少个文档中
        for doc_freq in self.doc_freqs:
            for word in doc_freq.keys():
                df[word] += 1

        # 计算IDF
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log(self.N / freq)

    def build_index(self, documents: List[Document]) -> bool:
        """
        构建BM25索引

        Args:
            documents: 文档列表

        Returns:
            是否构建成功
        """
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
        if self.config.target_segment_id:
            suffix = "_with_timeless" if self.config.include_timeless else ""
            return f"bm25/segment_{self.config.target_segment_id}{suffix}"
        return "bm25/full_corpus"

    def _load_existing_index(self, index_path: Path) -> bool:
        """加载已有索引"""
        try:
            if self.config.verbose:
                print(f"📂 加载已有BM25索引: {index_path.name}")

            # 加载索引文件
            with open(index_path / "bm25_index.pkl", 'rb') as f:
                index_data = pickle.load(f)

            self.documents = index_data['documents']
            self.doc_freqs = index_data['doc_freqs']
            self.idf = index_data['idf']
            self.doc_lens = index_data['doc_lens']
            self.avgdl = index_data['avgdl']
            self.N = index_data['N']

            if self.config.verbose:
                print(f"✅ BM25索引加载成功: {self.N} 文档")
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ BM25索引加载失败: {e}")
            return False

    def _build_new_index(self, documents: List[Document], index_path: Path) -> bool:
        """构建新的BM25索引"""
        try:
            self.documents = documents
            self.N = len(documents)
            self.doc_freqs = []
            self.doc_lens = []

            # 为每个文档计算词频
            for doc in tqdm(documents, desc="BM25索引", disable=not self.config.verbose):
                # 分词
                tokens = self._tokenize(doc.text)

                # 计算词频
                word_freq = Counter(tokens)
                self.doc_freqs.append(dict(word_freq))
                self.doc_lens.append(len(tokens))

            # 计算平均文档长度
            self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

            # 计算IDF
            self._compute_idf()

            # 保存索引
            index_path.mkdir(parents=True, exist_ok=True)

            index_data = {
                'documents': self.documents,
                'doc_freqs': self.doc_freqs,
                'idf': self.idf,
                'doc_lens': self.doc_lens,
                'avgdl': self.avgdl,
                'N': self.N
            }

            with open(index_path / "bm25_index.pkl", 'wb') as f:
                pickle.dump(index_data, f)

            # 保存元数据
            metadata = {
                'k1': self.k1,
                'b': self.b,
                'num_documents': self.N,
                'num_vocab': len(self.idf),
                'avg_doc_len': self.avgdl
            }

            with open(index_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            if self.config.verbose:
                print(f"✅ BM25索引构建完成: {self.N} 文档")

            return True
        except Exception as e:
            if self.config.verbose:
                print(f"❌ BM25索引构建失败: {e}")
            return False

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        计算单个文档的BM25得分

        Args:
            query_tokens: 查询分词结果
            doc_idx: 文档索引

        Returns:
            BM25得分
        """
        score = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        for token in query_tokens:
            if token in doc_freq and token in self.idf:
                tf = doc_freq[token]
                idf = self.idf[token]

                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator

        return score

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索相关文档

        Args:
            query: 查询字符串
            top_k: 返回top-k结果，默认使用配置中的值

        Returns:
            检索结果列表
        """
        if not self.documents:
            raise ValueError("BM25索引未初始化")

        if top_k is None:
            top_k = self.config.top_k

        # 查询分词
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 计算所有文档的BM25得分
        scores = []
        for i in range(self.N):
            score = self._bm25_score(query_tokens, i)
            scores.append((score, i))

        # 按得分排序，取top-k
        scores.sort(reverse=True)
        top_scores = scores[:top_k]

        # 构建结果
        results = []
        for score, doc_idx in top_scores:
            doc = self.documents[doc_idx]
            results.append({
                'id': doc.id_,
                'content': doc.text,
                'metadata': doc.metadata,
                'score': score
            })

        return results

    def batch_retrieve(self, queries: List[str], top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        批量检索

        Args:
            queries: 查询列表
            top_k: 返回top-k结果，默认使用配置中的值

        Returns:
            每个查询对应的检索结果列表
        """
        if top_k is None:
            top_k = self.config.top_k

        results = []

        for i, query in enumerate(tqdm(queries, desc="BM25检索", disable=not self.config.verbose)):

            try:
                query_results = self.retrieve(query, top_k)
                results.append(query_results)
            except Exception:
                results.append([])

        return results
