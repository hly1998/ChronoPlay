# -*- coding: utf-8 -*-
"""
ChronoPlay检索评估器
专门用于评估RAG系统的检索效果 - 仅基础指标
"""

import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any
from pathlib import Path

from components.retrieval_evaluator_core import RetrievalEvaluator


class ChronoPlayRetrievalEvaluator:
    """ChronoPlay检索评估器 - 仅基础指标评估"""

    def __init__(self, config: Dict):
        self.config = config
        self.game_name = config.get('game_name', 'dyinglight2')
        self.target_segment_id = config.get('target_segment_id', None)
        self.evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

    def load_retrieval_results(self, retrieval_results_path: str) -> List[Dict]:
        """加载检索结果数据"""
        retrieval_data = []
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line.strip())
                original_qa_data = item.get('original_qa_data', {})

                # 提取 ground truth 文档IDs
                ground_truth_docs = original_qa_data.get('retrieved_docs', [])
                ground_truth_doc_ids = [
                    doc['metadata']['id'] for doc in ground_truth_docs
                    if isinstance(doc, dict) and 'metadata' in doc and 'id' in doc['metadata']
                ]

                # 提取检索结果文档IDs
                retrieved_docs = item.get('retrieved_docs', [])
                retrieved_doc_ids = [
                    doc['metadata']['id'] for doc in retrieved_docs
                    if isinstance(doc, dict) and 'metadata' in doc and 'id' in doc['metadata']
                ]

                # 提取模型信息
                retrieval_config = item.get('retrieval_config', {})
                retrieval_method = retrieval_config.get('retrieval_method', 'vector')
                embedding_model = retrieval_config.get('embedding_model', 'unknown')
                top_k = retrieval_config.get('top_k', 5)

                # 生成模型标识
                model_key = f"bm25_k{top_k}" if retrieval_method == 'bm25' else f"{embedding_model}_k{top_k}"

                eval_item = {
                    'question': item.get('question', ''),
                    'retrieved_docs': retrieved_docs,
                    'retrieved_doc_ids': retrieved_doc_ids,
                    'ground_truth_doc_ids': ground_truth_doc_ids,
                    'ground_truth_docs': ground_truth_docs,
                    'retrieval_time': item.get('retrieval_time', 0),
                    'config': retrieval_config,
                    'metadata': {
                        'question_type': original_qa_data.get('question_type', 'unknown'),
                        'difficulty': original_qa_data.get('difficulty', 'unknown'),
                        'segment_id': retrieval_config.get('segment_id'),
                        'k': top_k,
                        'retrieval_method': retrieval_method,
                        'embedding_model': embedding_model if retrieval_method != 'bm25' else "BM25",
                        'embedding_service': retrieval_config.get('embedding_service', 'unknown'),
                        'model_key': model_key
                    }
                }
                retrieval_data.append(eval_item)

        return retrieval_data

    def evaluate_retrieval_metrics(self, retrieval_data: List[Dict]) -> Dict[str, Any]:
        """评估检索指标：Recall@K, F1@K, NDCG@K (k=1,3,5)"""
        return self.evaluator.evaluate_metrics(retrieval_data)

    def analyze_performance(self, retrieval_data: List[Dict]) -> Dict[str, Any]:
        """分析检索性能（按不同维度分组）"""
        return self.evaluator.analyze_performance(retrieval_data)

    def evaluate_segments(self, retrieval_results_dir: str, output_dir: str = None,
                          segment_ids: List[int] = None) -> Dict[str, Any]:
        """对指定分段进行检索评估 - 按模型分组"""
        # 确定要评估的分段
        if segment_ids is None:
            # 自动发现可用的分段
            results_path = Path(retrieval_results_dir)
            segment_ids = []

            # 检索结果文件模式
            patterns = [
                f"retrieval_{self.game_name}_segment_*_*.jsonl"
            ]

            for pattern in patterns:
                for file in results_path.glob(pattern):
                    try:
                        # 从文件名提取分段ID
                        parts = file.stem.split('_')
                        for i, part in enumerate(parts):
                            if part == 'segment' and i + 1 < len(parts):
                                segment_id = int(parts[i + 1])
                                if segment_id not in segment_ids:
                                    segment_ids.append(segment_id)
                                break
                    except (ValueError, IndexError):
                        continue
            segment_ids.sort()

        if self.target_segment_id:
            segment_ids = [self.target_segment_id]

        # 按模型分组的结果
        model_results = {}
        all_segments_summary = {}

        print(f"🔍 评估 {len(segment_ids)} 个分段: {segment_ids}")

        for segment_id in tqdm(segment_ids, desc="评估分段"):
            # 查找检索结果文件
            retrieval_files = list(Path(retrieval_results_dir).glob(
                f"retrieval_{self.game_name}_segment_{segment_id}_*.jsonl"))

            if not retrieval_files:
                all_segments_summary[segment_id] = {
                    'segment_id': segment_id,
                    'error': 'No retrieval results file found'
                }
                continue

            # 按模型分组处理文件
            segment_model_results = {}

            for retrieval_file in retrieval_files:
                # 加载分段数据
                try:
                    retrieval_data = self.load_retrieval_results(str(retrieval_file))
                    if not retrieval_data:
                        continue

                    # 按模型分组数据
                    model_groups = defaultdict(list)
                    for item in retrieval_data:
                        model_key = item['metadata']['model_key']
                        model_groups[model_key].append(item)

                    # 对每个模型组进行评估
                    for model_key, model_data in model_groups.items():

                        # 执行检索评估
                        results = {}
                        results['metrics'] = self.evaluate_retrieval_metrics(model_data)
                        results['analysis'] = self.analyze_performance(model_data)
                        results['model_info'] = {
                            'retrieval_method': model_data[0]['metadata']['retrieval_method'],
                            'embedding_model': model_data[0]['metadata']['embedding_model'],
                            'embedding_service': model_data[0]['metadata']['embedding_service'],
                            'top_k': model_data[0]['metadata']['k'],
                            'model_key': model_key
                        }

                        # 保存模型结果
                        if output_dir:
                            model_output_dir = Path(output_dir) / model_key
                            model_output_dir.mkdir(parents=True, exist_ok=True)
                            output_file = model_output_dir / \
                                f"{self.game_name}_segment_{segment_id}_retrieval_evaluation.json"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, ensure_ascii=False, indent=2)

                        # 记录模型结果
                        if model_key not in model_results:
                            model_results[model_key] = {}

                        model_results[model_key][segment_id] = {
                            'segment_id': segment_id,
                            'data_count': len(model_data),
                            'metrics': results.get('metrics', {}),
                            'analysis': results.get('analysis', {}),
                            'model_info': results.get('model_info', {}),
                            'output_file': str(output_file) if output_dir else None
                        }

                        segment_model_results[model_key] = {
                            'data_count': len(model_data),
                            'metrics': results.get('metrics', {}),
                            'model_info': results.get('model_info', {})
                        }

                except Exception:
                    pass

            # 记录分段总结
            if segment_model_results:
                all_segments_summary[segment_id] = {
                    'segment_id': segment_id,
                    'models': segment_model_results,
                    'total_models': len(segment_model_results)
                }
            else:
                all_segments_summary[segment_id] = {
                    'segment_id': segment_id,
                    'error': 'No valid retrieval data found'
                }

        # 保存每个模型的总结报告
        if output_dir:
            for model_key, model_data in model_results.items():
                model_summary_file = Path(output_dir) / model_key / \
                    f"{self.game_name}_retrieval_evaluation_summary.json"
                model_summary_file.parent.mkdir(parents=True, exist_ok=True)
                with open(model_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(model_data, f, ensure_ascii=False, indent=2)

            # 保存总体总结报告
            overall_summary_file = Path(output_dir) / f"{self.game_name}_retrieval_evaluation_overall_summary.json"
            with open(overall_summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_segments_summary, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存到: {output_dir}")

        # 打印总结
        self._print_model_summary(model_results, all_segments_summary)

        return model_results

    def _print_model_summary(self, model_results: Dict, all_segments_summary: Dict):
        """打印按模型分组的检索评估总结"""
        print("\n" + "=" * 80)
        print(f"{self.game_name.upper()} 检索评估总结")
        print("=" * 80)

        total_segments = len(all_segments_summary)
        successful_segments = len([s for s in all_segments_summary.values() if 'error' not in s])
        print(f"\n📊 评估了 {len(model_results)} 个模型, {successful_segments}/{total_segments} 个分段")

        # 按模型打印结果
        for model_key, model_data in model_results.items():
            print(f"\n🔧 {model_key}")

            # 汇总指标
            model_metrics = defaultdict(list)
            total_data = 0
            for segment_data in model_data.values():
                if 'error' not in segment_data:
                    total_data += segment_data.get('data_count', 0)
                    metrics = segment_data.get('metrics', {})
                    for metric, score in metrics.items():
                        if isinstance(score, (int, float)):
                            model_metrics[metric].append(score)

            print(f"   数据量: {total_data} | 分段: {len(model_data)}")

            # 打印平均指标
            for k in [1, 3, 5]:
                metrics_str = []
                for metric_type in ['recall', 'f1', 'ndcg']:
                    key = f'{metric_type}_at_{k}'
                    if key in model_metrics and model_metrics[key]:
                        avg = np.mean(model_metrics[key])
                        metrics_str.append(f"{metric_type.upper()}@{k}={avg:.3f}")
                if metrics_str:
                    print(f"   K={k}: {' | '.join(metrics_str)}")

        print("\n✅ 评估完成")

    def run_single_evaluation(self, retrieval_results_path: str, output_path: str = None) -> Dict:
        """运行单文件检索评估"""
        retrieval_data = self.load_retrieval_results(retrieval_results_path)
        if not retrieval_data:
            return {}

        results = {
            'metrics': self.evaluate_retrieval_metrics(retrieval_data),
            'analysis': self.analyze_performance(retrieval_data)
        }

        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存: {output_path}")

        # 打印总结
        self._print_single_summary(results)
        return results

    def _print_single_summary(self, results: Dict):
        """打印单文件评估结果总结"""
        print("\n" + "=" * 60)
        print("检索评估结果")
        print("=" * 60)

        metrics = results.get('metrics', {})
        print(f"\n📊 评估了 {metrics.get('valid_queries', 0)}/{metrics.get('total_queries', 0)} 个查询")

        for k in [1, 3, 5]:
            metrics_str = []
            for metric_type in ['recall', 'f1', 'ndcg']:
                key = f'{metric_type}_at_{k}'
                if key in metrics:
                    metrics_str.append(f"{metric_type.upper()}@{k}={metrics[key]:.3f}")
            if metrics_str:
                print(f"   K={k}: {' | '.join(metrics_str)}")

        timing = results.get('analysis', {}).get('timing_analysis', {})
        if timing.get('total_samples', 0) > 0:
            print(f"\n⏱️  平均检索时间: {timing['avg_retrieval_time']:.3f}s")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='ChronoPlay检索效果评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 自动评估所有分段的检索效果
  python retrieval_evaluator.py

  # 评估特定分段
  python retrieval_evaluator.py --segment_id 1

  # 评估特定游戏的所有分段
  python retrieval_evaluator.py --game dyinglight2

  # 单文件评估
  python retrieval_evaluator.py --retrieval_results ./results/retrieval_dyinglight2_segment_1_*.jsonl
        """
    )

    # 基础配置
    parser.add_argument('--game', type=str, default='dyinglight2',
                        help='游戏名称 (默认: dyinglight2)')
    parser.add_argument('--segment_id', type=int,
                        help='目标分段ID (不指定则评估所有可用分段)')

    # 输入输出配置
    parser.add_argument('--retrieval_results', type=str,
                        help='单个检索结果文件路径 (如果指定则进入单文件模式)')
    parser.add_argument('--results_dir', type=str,
                        default='evaluation/retrieval_results',
                        help='检索结果目录 (默认: ./retrieval_results)')
    parser.add_argument('--output_dir', type=str,
                        default='evaluation/retrieval_evaluation',
                        help='评估结果输出目录 (默认: ./retrieval_evaluation)')

    args = parser.parse_args()

    print(f"🔍 ChronoPlay检索评估器 - {args.game}")

    if args.retrieval_results:
        mode = "single_file"
        retrieval_file = args.retrieval_results
        if not os.path.exists(retrieval_file):
            print(f"❌ 文件不存在: {retrieval_file}")
            return
    else:
        mode = "batch"
        results_dir = args.results_dir
        if not os.path.exists(results_dir):
            print(f"❌ 目录不存在: {results_dir}")
            return

    # 创建评估器
    config = {
        'game_name': args.game,
        'target_segment_id': args.segment_id
    }
    evaluator = ChronoPlayRetrievalEvaluator(config)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if mode == "single_file":
            output_file = os.path.join(args.output_dir, f"{args.game}_single_retrieval_evaluation.json")
            evaluator.run_single_evaluation(retrieval_file, output_file)
        else:
            evaluator.evaluate_segments(results_dir, args.output_dir)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == '__main__':
    main()
