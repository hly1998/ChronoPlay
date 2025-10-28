# -*- coding: utf-8 -*-
"""
生成评估器
"""

import json
import argparse
from pathlib import Path

from components.generation_evaluator_core import GenerationEvaluatorCore, print_results_summary


def run_evaluation(generation_file: str, output_file: str, config: dict,
                   metrics: list = None, test_mode: bool = False):
    """运行评估"""
    if metrics is None:
        metrics = ['correctness', 'faithfulness']

    # 创建评估器
    evaluator = GenerationEvaluatorCore(config)

    # 加载数据
    print(f"加载: {generation_file}")
    data = evaluator.load_generation_results(generation_file)
    print(f"已加载 {len(data)} 条数据")

    # 评估
    results = {
        'llm_metrics': evaluator.evaluate_metrics(data, metrics, test_mode),
        'basic_metrics': evaluator.evaluate_basic_metrics(data)
    }

    # 保存结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_results_summary(results)
    print(f"\n保存至: {output_file}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成评估器')

    # 输入输出
    parser.add_argument('--generation_results', required=True, help='生成结果文件')
    parser.add_argument('--output', help='评估结果输出路径')

    # 游戏配置
    parser.add_argument('--game', default='dyinglight2', help='游戏名称')
    parser.add_argument('--test_mode', action='store_true', help='测试模式(只评估5个样本)')

    # 评估配置
    parser.add_argument(
        '--metrics', nargs='+', default=['correctness', 'faithfulness'],
        choices=['correctness', 'faithfulness'], help='评估指标')

    # API配置
    parser.add_argument('--model_name', default='gpt-4o', help='评估模型')
    parser.add_argument(
        '--api_key',
        default='{your-api-key}',
        help='API密钥'
    )
    parser.add_argument('--openai_api', default='{your-base-url}', help='API地址')

    args = parser.parse_args()

    # 验证输入文件
    if not Path(args.generation_results).exists():
        print(f"❌ 文件不存在: {args.generation_results}")
        return

    # 确定输出路径
    if args.output:
        output_file = args.output
    else:
        # 自动生成输出路径
        input_file = Path(args.generation_results)
        output_dir = Path("evaluation/generation_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 提取模型名
        model_name = args.model_name.replace('-', '_').replace('.', '_')
        output_file = output_dir / f"{input_file.stem}_{model_name}_eval.json"

    # 配置
    config = {
        'game_name': args.game,
        'model_name': args.model_name,
        'api_key': args.api_key,
        'openai_api': args.openai_api,
        'temperature': 0.01
    }

    print("=" * 50)
    print("生成评估")
    print("=" * 50)
    print(f"游戏: {args.game}")
    print(f"输入: {args.generation_results}")
    print(f"输出: {output_file}")
    print(f"模型: {args.model_name}")
    print(f"指标: {', '.join(args.metrics)}")
    print("=" * 50)

    # 执行评估
    run_evaluation(
        generation_file=args.generation_results,
        output_file=output_file,
        config=config,
        metrics=args.metrics,
        test_mode=args.test_mode
    )


if __name__ == '__main__':
    main()
