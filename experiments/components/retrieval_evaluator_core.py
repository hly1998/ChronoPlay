# -*- coding: utf-8 -*-
"""
检索评估核心逻辑
包含检索指标计算、性能分析等核心功能
"""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


class RetrievalMetricsCalculator:
    """检索指标计算器"""

    @staticmethod
    def calculate_ndcg(retrieved_ids: List[str], relevant_ids_set: set, k: int) -> float:
        """计算NDCG@K"""
        if not retrieved_ids or not relevant_ids_set:
            return 0.0

        # 计算DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids_set:
                dcg += 1.0 / np.log2(i + 2)

        # 计算IDCG@K
        num_relevant = len(relevant_ids_set)
        idcg = 0.0
        for i in range(min(k, num_relevant)):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def evaluate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> Dict[str, float]:
        """计算单个查询在k值下的各项指标"""
        retrieved_ids_at_k = retrieved_ids[:k]
        retrieved_ids_set = set(retrieved_ids_at_k)
        relevant_ids_set = set(relevant_ids)

        if not relevant_ids:
            return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'ndcg': 0.0}

        intersection = retrieved_ids_set & relevant_ids_set

        # Recall@K
        recall = len(intersection) / len(relevant_ids)

        # Precision@K
        precision = len(intersection) / len(retrieved_ids_at_k) if retrieved_ids_at_k else 0.0

        # F1@K
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # NDCG@K
        ndcg = RetrievalMetricsCalculator.calculate_ndcg(retrieved_ids_at_k, relevant_ids_set, k)

        return {'recall': recall, 'precision': precision, 'f1': f1, 'ndcg': ndcg}


class RetrievalEvaluator:
    """检索评估器核心类"""

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5]
        self.calculator = RetrievalMetricsCalculator()

    def evaluate_metrics(self, retrieval_data: List[Dict]) -> Dict[str, Any]:
        """评估检索指标：Recall@K, F1@K, NDCG@K"""
        metrics_by_k = {k: {'recall_scores': [], 'f1_scores': [], 'ndcg_scores': []}
                        for k in self.k_values}
        valid_queries = 0

        for item in retrieval_data:
            retrieved_doc_ids = item.get('retrieved_doc_ids', [])
            ground_truth_doc_ids = item.get('ground_truth_doc_ids', [])

            # 过滤空ID
            retrieved_ids = [doc_id for doc_id in retrieved_doc_ids if doc_id]
            relevant_ids = [doc_id for doc_id in ground_truth_doc_ids if doc_id]

            if not relevant_ids:
                continue

            valid_queries += 1

            # 对每个k值计算指标
            for k in self.k_values:
                metrics = self.calculator.evaluate_at_k(retrieved_ids, relevant_ids, k)
                metrics_by_k[k]['recall_scores'].append(metrics['recall'])
                metrics_by_k[k]['f1_scores'].append(metrics['f1'])
                metrics_by_k[k]['ndcg_scores'].append(metrics['ndcg'])

        # 汇总结果
        results = {
            'total_queries': len(retrieval_data),
            'valid_queries': valid_queries,
            'coverage': valid_queries / len(retrieval_data) if retrieval_data else 0.0
        }

        for k in self.k_values:
            results[f'recall_at_{k}'] = np.mean(
                metrics_by_k[k]['recall_scores']) if metrics_by_k[k]['recall_scores'] else 0.0
            results[f'f1_at_{k}'] = np.mean(metrics_by_k[k]['f1_scores']) if metrics_by_k[k]['f1_scores'] else 0.0
            results[f'ndcg_at_{k}'] = np.mean(metrics_by_k[k]['ndcg_scores']) if metrics_by_k[k]['ndcg_scores'] else 0.0

        return results

    def analyze_performance(self, retrieval_data: List[Dict]) -> Dict[str, Any]:
        """分析检索性能（按不同维度分组）"""
        type_performance = defaultdict(lambda: defaultdict(list))
        difficulty_performance = defaultdict(lambda: defaultdict(list))
        segment_performance = defaultdict(lambda: defaultdict(list))
        retrieval_times = []

        for item in retrieval_data:
            metadata = item.get('metadata', {})
            question_type = metadata.get('question_type', 'unknown')
            difficulty = metadata.get('difficulty', 'unknown')
            segment_id = metadata.get('segment_id', 'unknown')
            retrieval_time = item.get('retrieval_time', 0)

            retrieved_doc_ids = item.get('retrieved_doc_ids', [])
            ground_truth_doc_ids = item.get('ground_truth_doc_ids', [])

            retrieved_ids = [doc_id for doc_id in retrieved_doc_ids if doc_id]
            relevant_ids = [doc_id for doc_id in ground_truth_doc_ids if doc_id]

            if relevant_ids:
                for k in self.k_values:
                    metrics = self.calculator.evaluate_at_k(retrieved_ids, relevant_ids, k)
                    performance = {'recall': metrics['recall'], 'f1': metrics['f1'], 'ndcg': metrics['ndcg']}
                    type_performance[question_type][k].append(performance)
                    difficulty_performance[difficulty][k].append(performance)
                    segment_performance[str(segment_id)][k].append(performance)

            if retrieval_time > 0:
                retrieval_times.append(retrieval_time)

        return {
            'by_question_type': self._aggregate_multi_k(type_performance),
            'by_difficulty': self._aggregate_multi_k(difficulty_performance),
            'by_segment': self._aggregate_multi_k(segment_performance),
            'timing_analysis': {
                'avg_retrieval_time': np.mean(retrieval_times) if retrieval_times else 0.0,
                'median_retrieval_time': np.median(retrieval_times) if retrieval_times else 0.0,
                'total_samples': len(retrieval_times)
            }
        }

    def _aggregate_multi_k(self, performance_dict: Dict) -> Dict:
        """聚合多k值性能指标"""
        aggregated = {}
        for category, k_data in performance_dict.items():
            if not k_data:
                continue

            aggregated[category] = {}
            for k in self.k_values:
                samples = k_data.get(k, [])
                if samples:
                    aggregated[category][f'k_{k}'] = {
                        'count': len(samples),
                        'avg_recall': float(np.mean([s['recall'] for s in samples])),
                        'avg_f1': float(np.mean([s['f1'] for s in samples])),
                        'avg_ndcg': float(np.mean([s['ndcg'] for s in samples]))
                    }

            if aggregated[category]:
                aggregated[category]['total_count'] = len(k_data.get(self.k_values[0], []))

        return aggregated
