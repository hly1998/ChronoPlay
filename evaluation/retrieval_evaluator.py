# -*- coding: utf-8 -*-
"""
简化版检索评估器
专门用于评估整个游戏的检索效果
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

from components.retrieval_evaluator_core import RetrievalEvaluator


def load_retrieval_results(file_path: str) -> List[Dict]:
    """加载检索结果数据"""
    retrieval_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line.strip())
            original_qa_data = item.get('original_qa_data', {})

            # 提取 ground truth 文档IDs
            ground_truth_docs = original_qa_data.get('retrieved_docs', [])
            ground_truth_doc_ids = []
            for doc in ground_truth_docs:
                if isinstance(doc, dict):
                    if 'metadata' in doc and 'id' in doc['metadata']:
                        ground_truth_doc_ids.append(doc['metadata']['id'])
                    elif 'id' in doc:
                        ground_truth_doc_ids.append(doc['id'])

            # 提取检索结果文档IDs
            retrieved_docs = item.get('retrieved_docs', [])
            retrieved_doc_ids = []
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    if 'metadata' in doc and 'id' in doc['metadata']:
                        retrieved_doc_ids.append(doc['metadata']['id'])
                    elif 'id' in doc:
                        retrieved_doc_ids.append(doc['id'])

            # 提取配置信息
            retrieval_config = item.get('retrieval_config', {})
            retrieval_method = retrieval_config.get('retrieval_method', 'vector')
            embedding_model = retrieval_config.get('embedding_model', 'unknown')
            top_k = retrieval_config.get('top_k', 5)

            eval_item = {
                'question': item.get('question', ''),
                'retrieved_docs': retrieved_docs,
                'retrieved_doc_ids': retrieved_doc_ids,
                'ground_truth_doc_ids': ground_truth_doc_ids,
                'ground_truth_docs': ground_truth_docs,
                'retrieval_time': item.get('retrieval_time', 0),
                'config': retrieval_config,
                'metadata': {
                    'question_type': original_qa_data.get('task_type', 'unknown'),
                    'question_topic': original_qa_data.get('question_topic', 'unknown'),
                    'game_name': original_qa_data.get('game_name', 'unknown'),
                    'k': top_k,
                    'retrieval_method': retrieval_method,
                    'embedding_model': embedding_model if retrieval_method != 'bm25' else "BM25",
                }
            }
            retrieval_data.append(eval_item)

    return retrieval_data


def evaluate_file(file_path: str, output_dir: str = None):
    """评估单个检索结果文件"""
    # 加载数据
    retrieval_data = load_retrieval_results(file_path)
    if not retrieval_data:
        print("❌ 未找到有效数据")
        return

    print(f"\n[检索评估] 样本数: {len(retrieval_data)}")

    # 创建评估器（只计算@3指标）
    evaluator = RetrievalEvaluator(k_values=[3])

    # 评估指标
    metrics = evaluator.evaluate_metrics(retrieval_data)

    # 打印核心指标 (使用 @3)
    recall_3 = metrics.get('recall_at_3', 0.0)
    f1_3 = metrics.get('f1_at_3', 0.0)
    ndcg_3 = metrics.get('ndcg_at_3', 0.0)
    print(f"  Recall@3: {recall_3:.4f} | F1@3: {f1_3:.4f} | NDCG@3: {ndcg_3:.4f}")

    # 性能分析
    performance_analysis = evaluator.analyze_performance(retrieval_data)

    # 打印按游戏划分的指标
    by_game = performance_analysis.get('by_game', {})
    avg_time = performance_analysis.get('avg_retrieval_time', 0.0)

    if by_game:
        print("\n  按游戏划分:")
        for game_name in sorted(by_game.keys()):
            game_metrics = by_game[game_name]
            count = game_metrics['count']
            recall = game_metrics['recall_at_3']
            f1 = game_metrics['f1_at_3']
            ndcg = game_metrics['ndcg_at_3']
            print(f"    {game_name} ({count}样本): Recall@3={recall:.4f} | F1@3={f1:.4f} | NDCG@3={ndcg:.4f}")

    print(f"  平均检索时间: {avg_time:.4f}s")

    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_stem = Path(file_path).stem
        result_file = output_path / f"{file_stem}_evaluation.json"

        result = {
            'file': str(file_path),
            'total_samples': len(retrieval_data),
            'overall_metrics': {
                'recall_at_3': metrics.get('recall_at_3', 0.0),
                'f1_at_3': metrics.get('f1_at_3', 0.0),
                'ndcg_at_3': metrics.get('ndcg_at_3', 0.0)
            },
            'by_game': performance_analysis.get('by_game', {}),
            'avg_retrieval_time': performance_analysis.get('avg_retrieval_time', 0.0)
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  详细结果: {result_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版检索评估器')
    parser.add_argument('--file', required=True, help='检索结果文件路径')
    parser.add_argument('--output_dir', default='evaluation/eval_results',
                        help='评估结果输出目录')

    args = parser.parse_args()

    evaluate_file(args.file, args.output_dir)


if __name__ == '__main__':
    main()
