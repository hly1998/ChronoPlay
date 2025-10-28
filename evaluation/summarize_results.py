#!/usr/bin/env python3
"""
评估结果汇总脚本
将检索和生成的评估结果合并成一个综合报告
"""

import json
from pathlib import Path
from typing import Dict, List, Any


# 模型名称映射（用于生成友好的显示名称）
MODEL_DISPLAY_NAMES = {
    "gpt_4o": "GPT-4o",
    "gpt_4.1": "GPT-4.1",
    "gpt_3.5_turbo": "GPT-3.5-Turbo",
    "gpt_5_chat_latest": "GPT-5-Chat",
    "claude_3_5_sonnet_20240620": "Claude-3.5-Sonnet",
    "claude_3_7_sonnet_20250219": "Claude-3.7-Sonnet",
    "claude_sonnet_4_20250514": "Claude-Sonnet-4",
    "claude_sonnet_4_5_20250929": "Claude-Sonnet-4.5",
    "gemini_2.5_flash": "Gemini-2.5-Flash",
    "gemini_2.5_pro": "Gemini-2.5-Pro",
    "deepseek_v3_1_250821": "DeepSeek-V3.1",
    "deepseek_v3_2_exp": "DeepSeek-V3.2-Exp",
    "llama_4_scout_17b_16e_instruct": "Llama-4-Scout",
    "qwen2.5_72b_instruct": "Qwen2.5-72B",
    "qwen3_max": "Qwen3-Max",
    "glm_4.5": "GLM-4.5",
    "kimi_k2_250905": "Kimi-K2",
    "grok_3": "Grok-3",
    "o1": "O1",
    "o3": "O3"
}

RETRIEVER_DISPLAY_NAMES = {
    "text_embedding_3_small": "Dense retrieval using text-embedding-3-small",
    "bm25": "Sparse retrieval using BM25"
}

GAME_NAME_MAPPING = {
    "dune": "dune",
    "dyinglight2": "dying_light_2",
    "pubgm": "pubg_mobile"
}


def load_json_file(filepath: Path) -> Dict:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_filename(filename: str) -> Dict[str, str]:
    """
    解析文件名以提取retriever和model信息
    例如：generation_retrieval_text_embedding_3_small_k3_gpt_4o_evaluation.json
    或：retrieval_text_embedding_3_small_k3_evaluation.json
    """
    parts = filename.replace('_evaluation.json', '').split('_')

    if filename.startswith('generation_retrieval'):
        # generation_retrieval_{retriever}_k3_{model}_evaluation.json
        # 需要找到k3的位置
        k3_idx = parts.index('k3')
        retriever = '_'.join(parts[2:k3_idx])
        model = '_'.join(parts[k3_idx + 1:])
        return {
            'type': 'generation',
            'retriever': retriever,
            'model': model
        }
    elif filename.startswith('retrieval'):
        # retrieval_{retriever}_k3_evaluation.json
        k3_idx = parts.index('k3')
        retriever = '_'.join(parts[1:k3_idx])
        return {
            'type': 'retrieval',
            'retriever': retriever
        }

    return {}


def merge_results(retrieval_data: Dict, generation_data: Dict, game_key: str) -> Dict[str, float]:
    """
    合并检索和生成结果
    """
    merged = {
        'topk': 3,
        'recall': round(retrieval_data.get('recall_at_3', 0), 3),
        'f1': round(retrieval_data.get('f1_at_3', 0), 3),
        'ndcg': round(retrieval_data.get('ndcg_at_3', 0), 3),
        'correctness': round(generation_data.get('correctness', 0), 3),
        'faithfulness': round(generation_data.get('faithfulness', 0), 3)
    }
    return merged


def process_results(eval_results_dir: Path) -> List[Dict[str, Any]]:
    """
    处理所有评估结果文件
    """
    # 首先加载所有检索结果
    retrieval_results = {}

    for filepath in eval_results_dir.glob('retrieval_*.json'):
        parsed = parse_filename(filepath.name)
        if parsed.get('type') == 'retrieval':
            data = load_json_file(filepath)
            retriever = parsed['retriever']
            retrieval_results[retriever] = data

    # 然后处理所有生成结果
    systems = []

    for filepath in sorted(eval_results_dir.glob('generation_*.json')):
        parsed = parse_filename(filepath.name)
        if parsed.get('type') == 'generation':
            retriever = parsed['retriever']
            model = parsed['model']

            # 加载生成数据
            generation_data = load_json_file(filepath)

            # 获取对应的检索数据
            if retriever not in retrieval_results:
                print(f"警告：找不到 {retriever} 的检索结果，跳过 {filepath.name}")
                continue

            retrieval_data = retrieval_results[retriever]

            # 构建系统信息
            model_display_name = MODEL_DISPLAY_NAMES.get(model, model.replace('_', '-').title())
            retriever_display_name = retriever.replace('_', '-')
            retriever_description = RETRIEVER_DISPLAY_NAMES.get(retriever, f"Retrieval using {retriever}")

            system = {
                'system_name': f'{retriever_display_name}+{model_display_name}',
                'description': f'{retriever_description} with {model_display_name} for generation',
                'games': {},
                'average': {}
            }

            # 处理每个游戏的结果
            for game_key in ['dune', 'dyinglight2', 'pubgm']:
                if game_key in retrieval_data['by_game'] and game_key in generation_data['by_game']:
                    game_display_name = GAME_NAME_MAPPING[game_key]
                    system['games'][game_display_name] = merge_results(
                        retrieval_data['by_game'][game_key],
                        generation_data['by_game'][game_key],
                        game_key
                    )

            # 计算平均值
            system['average'] = merge_results(
                retrieval_data['overall_metrics'],
                generation_data['overall_metrics'],
                'overall'
            )

            systems.append(system)

    return systems


def main():
    """主函数"""
    eval_results_dir = Path('/Users/liyanghe/Documents/chronoplay/evaluation/evaluation/eval_results')
    output_file = Path('/Users/liyanghe/Documents/chronoplay/evaluation/evaluation/summary_results.json')

    print(f"正在从 {eval_results_dir} 读取评估结果...")

    # 处理所有结果
    systems = process_results(eval_results_dir)

    # 按系统名称排序
    systems.sort(key=lambda x: x['system_name'])

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(systems, f, indent=4, ensure_ascii=False)

    print(f"✓ 已生成汇总结果: {output_file}")
    print(f"✓ 共处理了 {len(systems)} 个系统配置")


if __name__ == '__main__':
    main()
