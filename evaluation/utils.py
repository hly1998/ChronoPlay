# -*- coding: utf-8 -*-
"""
ChronoPlay评估工具函数
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(results: Dict[str, Any], output_path: str):
    """保存评估结果"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL格式文件"""
    data = []

    if not os.path.exists(file_path):
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"⚠️ {file_path}第{line_num}行JSON解析失败: {e}")
                    continue

    return data


def save_jsonl_file(data: List[Dict[str, Any]], file_path: str):
    """保存JSONL格式文件"""
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_available_segments(game_name: str, data_dir: str) -> List[int]:
    """获取可用的分段ID列表"""
    segments = []
    # 新路径: data/{游戏名称}/segments/segment_{id}
    data_path = Path(data_dir) / game_name / "segments"

    if not data_path.exists():
        return segments

    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('segment_'):
            try:
                segment_id = int(item.name.split('_')[1])
                segments.append(segment_id)
            except (ValueError, IndexError):
                continue

    return sorted(segments)


def calculate_metrics_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """计算指标统计信息"""
    stats = {}

    for category, metrics in results.items():
        if isinstance(metrics, dict):
            category_stats = {}

            for metric, values in metrics.items():
                if isinstance(values, dict) and 'mean' in values:
                    category_stats[metric] = values
                elif isinstance(values, (list, np.ndarray)):
                    valid_values = [v for v in values if v >= 0]
                    if valid_values:
                        category_stats[metric] = {
                            'mean': float(np.mean(valid_values)),
                            'std': float(np.std(valid_values)),
                            'min': float(np.min(valid_values)),
                            'max': float(np.max(valid_values)),
                            'count': len(valid_values),
                            'total': len(values)
                        }

            if category_stats:
                stats[category] = category_stats

    return stats


def generate_evaluation_report(results_summary: Dict[str, Any],
                               game_name: str,
                               output_path: str = None) -> str:
    """生成评估报告"""
    report_lines = []

    # 报告标题
    report_lines.append(f"# {game_name.upper()} RAG系统评估报告")
    report_lines.append("")

    # 总体统计
    total_segments = len(results_summary)
    successful_segments = sum(1 for r in results_summary.values() if 'error' not in r)
    total_data_count = sum(r.get('data_count', 0) for r in results_summary.values() if 'error' not in r)

    report_lines.append("## 总体统计")
    report_lines.append(f"- 处理分段数: {successful_segments}/{total_segments}")
    report_lines.append(f"- 总评估数据: {total_data_count} 条")
    report_lines.append(f"- 成功率: {successful_segments/total_segments*100:.1f}%")
    report_lines.append("")

    # 分段详情
    report_lines.append("## 分段详情")
    for segment_id, summary in sorted(results_summary.items()):
        data_count = summary.get('data_count', 0)

        if 'error' in summary:
            report_lines.append(f"- 分段 {segment_id}: ❌ {summary['error']}")
        else:
            report_lines.append(f"- 分段 {segment_id}: ✅ {data_count} 条数据")

    report_lines.append("")

    # 收集指标数据
    retrieval_metrics = defaultdict(list)
    generation_metrics = defaultdict(list)

    for summary in results_summary.values():
        if 'error' not in summary:
            # 收集检索指标
            retrieval_results = summary.get('retrieval_results', {})
            for metric, score in retrieval_results.items():
                if isinstance(score, (int, float)):
                    retrieval_metrics[metric].append(score)

            # 收集生成指标
            generation_results = summary.get('generation_results', {})
            for metric, result in generation_results.items():
                if isinstance(result, dict) and 'mean' in result:
                    generation_metrics[metric].append(result['mean'])

    # 检索指标报告
    if retrieval_metrics:
        report_lines.append("## 检索指标平均值")
        for metric, scores in retrieval_metrics.items():
            avg_score = np.mean(scores) if scores else 0.0
            report_lines.append(f"- {metric}: {avg_score:.4f} (来自 {len(scores)} 个分段)")
        report_lines.append("")

    # 生成指标报告
    if generation_metrics:
        report_lines.append("## 生成指标平均值")
        for metric, scores in generation_metrics.items():
            avg_score = np.mean(scores) if scores else 0.0
            report_lines.append(f"- {metric}: {avg_score:.4f} (来自 {len(scores)} 个分段)")
        report_lines.append("")

    # 分段性能对比
    if len(results_summary) > 1:
        report_lines.append("## 分段性能对比")
        report_lines.append("### 检索性能")
        for metric in ['precision', 'recall', 'f1', 'map']:
            if metric in retrieval_metrics:
                report_lines.append(f"#### {metric.upper()}")
                segment_scores = {}
                for segment_id, summary in results_summary.items():
                    if 'error' not in summary:
                        retrieval_results = summary.get('retrieval_results', {})
                        score = retrieval_results.get(metric, 0.0)
                        if isinstance(score, (int, float)):
                            segment_scores[segment_id] = score

                for segment_id, score in sorted(segment_scores.items()):
                    report_lines.append(f"- 分段 {segment_id}: {score:.4f}")
                report_lines.append("")

        report_lines.append("### 生成性能")
        for metric in ['accuracy', 'completeness', 'factuality', 'game_relevance']:
            if metric in generation_metrics:
                report_lines.append(f"#### {metric}")
                segment_scores = {}
                for segment_id, summary in results_summary.items():
                    if 'error' not in summary:
                        generation_results = summary.get('generation_results', {})
                        result = generation_results.get(metric, {})
                        if isinstance(result, dict) and 'mean' in result:
                            segment_scores[segment_id] = result['mean']

                for segment_id, score in sorted(segment_scores.items()):
                    report_lines.append(f"- 分段 {segment_id}: {score:.4f}")
                report_lines.append("")

    # 生成报告文本
    report_text = "\n".join(report_lines)

    # 保存报告
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"📊 评估报告已保存到: {output_path}")

    return report_text


def clean_response_text(text: str) -> str:
    """清理响应文本"""
    if not text:
        return ""

    # 移除多余的空白字符
    text = " ".join(text.split())

    # 移除可能的前缀和后缀
    prefixes_to_remove = [
        "回答：", "答案：", "Answer:", "Response:",
        "根据提供的资料", "基于参考资料"
    ]

    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    return text


def validate_evaluation_data(rag_results: List[Dict], ground_truth: List[Dict]) -> bool:
    """验证评估数据的完整性"""
    if not rag_results:
        print("❌ RAG结果数据为空")
        return False

    if not ground_truth:
        print("❌ 真实标签数据为空")
        return False

    # 检查必需字段
    required_rag_fields = ['question', 'answer']
    required_gt_fields = ['question', 'answer']

    for i, item in enumerate(rag_results[:5]):  # 只检查前5条
        for field in required_rag_fields:
            if field not in item:
                print(f"❌ RAG结果第{i+1}条缺少字段: {field}")
                return False

    for i, item in enumerate(ground_truth[:5]):  # 只检查前5条
        for field in required_gt_fields:
            if field not in item:
                print(f"❌ 真实标签第{i+1}条缺少字段: {field}")
                return False

    print(f"✅ 数据验证通过: RAG结果 {len(rag_results)} 条, 真实标签 {len(ground_truth)} 条")
    return True


def format_evaluation_summary(results: Dict[str, Any]) -> str:
    """格式化评估结果摘要"""
    summary_lines = []

    if 'retrieval' in results:
        summary_lines.append("📊 检索评估结果:")
        for metric, score in results['retrieval'].items():
            if isinstance(score, (int, float)):
                summary_lines.append(f"  {metric}: {score:.4f}")

    if 'generation' in results:
        summary_lines.append("📝 生成评估结果:")
        for metric, result in results['generation'].items():
            if isinstance(result, dict) and 'mean' in result:
                summary_lines.append(f"  {metric}: {result['mean']:.4f} ({result['count']}/{result['total']})")

    return "\n".join(summary_lines)


def get_segment_info(game_name: str, segment_id: int, corpus_dir: str) -> Dict[str, Any]:
    """获取分段信息"""
    segment_info_file = Path(corpus_dir) / game_name / f"segment_{segment_id}" / "segment_info.json"

    if not segment_info_file.exists():
        return {
            'segment_id': segment_id,
            'start_date': 'unknown',
            'end_date': 'unknown',
            'document_count': 0
        }

    try:
        with open(segment_info_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 加载分段 {segment_id} 信息失败: {e}")
        return {
            'segment_id': segment_id,
            'start_date': 'unknown',
            'end_date': 'unknown',
            'document_count': 0
        }
