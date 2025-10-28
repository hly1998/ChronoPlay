# -*- coding: utf-8 -*-
"""
生成评估核心组件
"""

import json
import time
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from .game_evaluator import GameCorrectnessEvaluator


class GenerationEvaluatorCore:
    """生成评估核心"""

    def __init__(self, config: Dict):
        self.config = config
        self.game_name = config.get('game_name', 'dyinglight2')

        # 初始化评估器
        self.evaluator = GameCorrectnessEvaluator(
            api_key=config['api_key'],
            base_url=config.get('openai_api', 'https://api.openai.com/v1'),
            model=config.get('model_name', 'gpt-4o'),
            temperature=config.get('temperature', 0.01)
        )

        self.default_metrics = ['correctness', 'faithfulness']

    def load_generation_results(self, filepath: str) -> List[Dict]:
        """加载生成结果"""
        results = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    results.append(self._prepare_item(item))
        return results

    def _prepare_item(self, item: Dict) -> Dict:
        """准备评估数据项"""
        # 提取contexts
        contexts = item.get('contexts', [])
        if not contexts and item.get('retrieved_docs'):
            contexts = [
                doc.get('content', '') if isinstance(doc, dict) else str(doc)
                for doc in item['retrieved_docs']
            ]

        return {
            'question': item.get('question', ''),
            'rag_answer': item.get('answer', item.get('rag_answer', '')),
            'ground_truth_answer': item.get('ground_truth_answer', ''),
            'contexts': contexts,
            'question_index': item.get('question_index', -1)
        }

    def evaluate_metrics(
            self, data: List[Dict], metrics: List[str] = None,
            test_mode: bool = False) -> Dict[str, Any]:
        """评估指标"""
        if metrics is None:
            metrics = self.default_metrics

        if test_mode:
            data = data[:min(5, len(data))]
            print(f"🔍 测试模式：评估 {len(data)} 个样本")

        detailed_results = []

        for i, item in enumerate(tqdm(data, desc="评估中", unit="项")):
            question = item['question']
            rag_answer = item['rag_answer']
            ground_truth = item['ground_truth_answer']

            result = {'index': i, 'correctness_score': 0.0, 'faithfulness_score': 0.0}

            if not rag_answer or not ground_truth:
                detailed_results.append(result)
                continue

            if 'correctness' in metrics:
                result['correctness_score'] = self.evaluator.evaluate_single(
                    question, ground_truth, rag_answer
                )

            if 'faithfulness' in metrics and item['contexts']:
                result['faithfulness_score'] = self.evaluator.evaluate_faithfulness(
                    question, item['contexts'], rag_answer
                )

            detailed_results.append(result)
            time.sleep(0.1)

        return self._calculate_statistics(detailed_results, metrics)

    def _calculate_statistics(
            self, detailed_results: List[Dict], metrics: List[str]) -> Dict[str, Any]:
        """计算统计信息"""
        overall = {}

        if 'correctness' in metrics:
            scores = [r['correctness_score'] for r in detailed_results]
            overall['correctness'] = {
                'mean': float(np.mean(scores)),
                'count': len(scores)
            }

        if 'faithfulness' in metrics:
            scores = [r['faithfulness_score'] for r in detailed_results]
            overall['faithfulness'] = {
                'mean': float(np.mean(scores)),
                'count': len(scores)
            }

        return {'overall': overall}

    def evaluate_basic_metrics(self, data: List[Dict]) -> Dict[str, Any]:
        """评估基础指标"""
        valid_answers = sum(1 for item in data if item.get('rag_answer', '').strip())
        total_length = sum(len(item.get('rag_answer', '')) for item in data if item.get('rag_answer'))

        return {
            'total_questions': len(data),
            'valid_answers': valid_answers,
            'answer_rate': valid_answers / len(data) if data else 0.0,
            'avg_answer_length': total_length / valid_answers if valid_answers > 0 else 0.0
        }


def print_results_summary(results: Dict):
    """打印结果摘要"""
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)

    if 'llm_metrics' in results and 'overall' in results['llm_metrics']:
        overall = results['llm_metrics']['overall']
        if 'correctness' in overall:
            correctness = overall['correctness']
            print(f"\n正确性: {correctness['mean']:.4f} (样本数: {correctness['count']})")
        if 'faithfulness' in overall:
            faithfulness = overall['faithfulness']
            print(f"忠实度: {faithfulness['mean']:.4f} (样本数: {faithfulness['count']})")

    if 'basic_metrics' in results:
        basic = results['basic_metrics']
        print(f"\n总问题数: {basic.get('total_questions', 0)}")
        print(f"有效答案数: {basic.get('valid_answers', 0)}")
        print(f"答案率: {basic.get('answer_rate', 0):.4f}")
