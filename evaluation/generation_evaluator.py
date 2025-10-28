# -*- coding: utf-8 -*-
"""
简化版生成评估器
专门用于评估整个游戏的生成效果
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_generation_results(file_path: str) -> List[Dict]:
    """加载生成结果数据"""
    generation_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line.strip())

            # 提取contexts
            contexts = item.get('contexts', [])
            if not contexts and item.get('retrieved_docs'):
                contexts = [
                    doc.get('content', '') if isinstance(doc, dict) else str(doc)
                    for doc in item['retrieved_docs']
                ]

            # 提取游戏名称
            game_name = 'unknown'
            if 'original_qa_data' in item:
                game_name = item['original_qa_data'].get('game_name', 'unknown')
            elif 'game_name' in item:
                game_name = item['game_name']
            elif 'retrieved_docs' in item and item['retrieved_docs']:
                # 从retrieved_docs的metadata中提取game字段
                for doc in item['retrieved_docs']:
                    if isinstance(doc, dict) and 'metadata' in doc:
                        doc_game = doc['metadata'].get('game')
                        if doc_game:
                            game_name = doc_game
                            break

            # 准备评估数据项
            eval_item = {
                'question': item.get('question', ''),
                'rag_answer': item.get('answer', item.get('rag_answer', '')),
                'ground_truth_answer': item.get('ground_truth_answer', ''),
                'contexts': contexts,
                'question_index': item.get('question_index', -1),
                'game_name': game_name
            }

            generation_data.append(eval_item)

    return generation_data


def evaluate_file(file_path: str, api_key: str, base_url: str, output_dir: str = None, concurrent_requests: int = 5):
    """评估单个生成结果文件"""
    # 加载数据
    generation_data = load_generation_results(file_path)
    if not generation_data:
        print("❌ 未找到有效数据")
        return

    print(f"\n[生成评估] 样本数: {len(generation_data)}, 并发数: {concurrent_requests}")

    # 创建评估器配置
    config = {
        'api_key': api_key,
        'openai_api': base_url,
        'model_name': 'gpt-4o',
        'temperature': 0.01
    }

    from components.generation_evaluator_core import GenerationEvaluatorCore
    evaluator = GenerationEvaluatorCore(config)

    # 评估指标（并发执行）
    metrics_result = evaluator.evaluate_metrics(generation_data, concurrent_requests=concurrent_requests)

    # 提取核心指标
    overall = metrics_result.get('overall', {})
    correctness = overall.get('correctness', {}).get('mean', 0.0)
    faithfulness = overall.get('faithfulness', {}).get('mean', 0.0)

    # 打印核心指标
    print(f"  Correctness: {correctness:.4f} | Faithfulness: {faithfulness:.4f}")

    # 打印按游戏划分的指标
    by_game = metrics_result.get('by_game', {})
    if by_game:
        print("\n  按游戏划分:")
        for game_name in sorted(by_game.keys()):
            game_metrics = by_game[game_name]
            count = game_metrics.get('count', 0)
            game_correctness = game_metrics.get('correctness', 0.0)
            game_faithfulness = game_metrics.get('faithfulness', 0.0)
            print(f"    {game_name} ({count}样本): Correctness={game_correctness:.4f} | Faithfulness={game_faithfulness:.4f}")

    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_stem = Path(file_path).stem
        result_file = output_path / f"{file_stem}_evaluation.json"

        result = {
            'file': str(file_path),
            'total_samples': len(generation_data),
            'overall_metrics': {
                'correctness': correctness,
                'faithfulness': faithfulness
            },
            'by_game': by_game
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  详细结果: {result_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版生成评估器')
    parser.add_argument('--file', required=True, help='生成结果文件路径')
    parser.add_argument('--api_key', default='{your-api-key}', help='API Key')
    parser.add_argument('--base_url', default='{your-base-url}', help='API Base URL')
    parser.add_argument('--concurrent_requests', type=int, default=5, help='并发请求数')
    parser.add_argument('--output_dir', default='evaluation/eval_results',
                        help='评估结果输出目录')

    args = parser.parse_args()

    evaluate_file(args.file, args.api_key, args.base_url, args.output_dir, args.concurrent_requests)


if __name__ == '__main__':
    main()
