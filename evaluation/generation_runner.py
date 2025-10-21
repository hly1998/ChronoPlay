# -*- coding: utf-8 -*-
"""
独立生成运行器
基于检索结果文件生成答案
"""

import argparse
import json
from pathlib import Path

from components import GenerationConfig, GenerationPipeline


def main():
    """生成主函数"""
    parser = argparse.ArgumentParser(description='独立生成运行器')

    # 基础参数
    parser.add_argument('--retrieval_file', required=True, help='检索结果文件路径')
    parser.add_argument('--output_file', help='输出文件路径')

    # 生成配置
    parser.add_argument('--api_key', default='{your-api-key}')
    parser.add_argument('--base_url', default='{your-base-url}')
    parser.add_argument('--llm_model', default='gpt-4o', help='生成模型')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--concurrent_requests', type=int, default=3, help='并发数')

    args = parser.parse_args()

    # 检查检索结果文件
    retrieval_file = Path(args.retrieval_file)
    if not retrieval_file.exists():
        print(f"❌ 文件不存在: {retrieval_file}")
        return

    # 加载检索结果
    retrieval_results = []
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                retrieval_results.append(json.loads(line.strip()))

    if not retrieval_results:
        print("❌ 检索结果文件为空")
        return

    print(f"📊 加载 {len(retrieval_results)} 个检索结果")
    print(f"🤖 模型: {args.llm_model}, 温度: {args.temperature}, 并发: {args.concurrent_requests}")

    # 创建生成配置
    config = GenerationConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        llm_model=args.llm_model,
        temperature=args.temperature,
        concurrent_requests=args.concurrent_requests,
        verbose=False
    )

    # 初始化生成pipeline
    pipeline = GenerationPipeline(config)

    # 确定输出文件
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # 自动生成文件名
        model_name = args.llm_model.replace('-', '_').replace('/', '_')
        filename = f"generation_{retrieval_file.stem}_{model_name}.jsonl"

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename

    print(f"💾 输出: {output_file}")

    # 执行生成
    results = pipeline.batch_generate(retrieval_results, str(output_file))

    # 统计
    success_count = len([r for r in results if 'error' not in r])
    error_count = len(results) - success_count
    print(f"✅ 完成: {success_count} 成功, {error_count} 失败")


if __name__ == '__main__':
    main()
