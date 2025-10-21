# -*- coding: utf-8 -*-
"""
独立检索运行器
只负责检索和索引构建，输出检索结果文件
"""

import argparse
from pathlib import Path

from components import RetrievalConfig, RetrievalPipeline


def main():
    """检索主函数"""
    parser = argparse.ArgumentParser(description='独立检索运行器')

    # 基础参数
    parser.add_argument('--game', default='dyinglight2', help='游戏名称')
    parser.add_argument('--segment_id', type=int, required=True, help='分段ID')
    parser.add_argument('--output_file', help='输出文件路径 (可选，会自动生成)')

    # 检索配置
    parser.add_argument('--retrieval_method', default='vector', choices=['vector', 'bm25'],
                        help='检索方法: vector (向量检索) 或 bm25 (BM25检索)')
    parser.add_argument('--api_key', default='{your-api-key}')
    parser.add_argument('--base_url', default='{your-base-url}')
    parser.add_argument('--embedding_model', default='text-embedding-3-small')
    parser.add_argument('--top_k', type=int, default=5)

    # BM25专用参数
    parser.add_argument('--bm25_k1', type=float, default=1.2, help='BM25参数k1 (词频饱和度)')
    parser.add_argument('--bm25_b', type=float, default=0.75, help='BM25参数b (文档长度归一化)')

    # 控制参数
    parser.add_argument('--force_rebuild', action='store_true', help='强制重建索引')
    parser.add_argument('--include_timeless', action='store_true', default=True, help='包含时间无关数据')
    parser.add_argument('--verbose', action='store_true', default=True, help='详细输出')

    args = parser.parse_args()

    # 创建检索配置
    config = RetrievalConfig(
        game_name=args.game,
        target_segment_id=args.segment_id,
        retrieval_method=args.retrieval_method,
        api_key=args.api_key,
        base_url=args.base_url,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        force_rebuild=args.force_rebuild,
        include_timeless=args.include_timeless,
        verbose=args.verbose
    )

    print("🔍 开始检索任务")
    print(f"   游戏: {args.game} | 分段: {args.segment_id}")
    if args.retrieval_method == 'vector':
        print(f"   方法: Vector | 模型: {args.embedding_model} | Top-K: {args.top_k}")
    elif args.retrieval_method == 'bm25':
        print(f"   方法: BM25 | k1={args.bm25_k1}, b={args.bm25_b} | Top-K: {args.top_k}")

    # 初始化检索pipeline
    pipeline = RetrievalPipeline(config)
    if not pipeline.initialize():
        print("❌ 检索初始化失败")
        return

    # 加载QA数据
    qa_pairs = pipeline.load_qa_pairs(args.segment_id)
    if not qa_pairs:
        print("❌ 未找到QA数据")
        return

    print(f"📊 加载了 {len(qa_pairs)} 个QA对")

    # 确定输出文件
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # 自动生成文件名
        if args.retrieval_method == 'bm25':
            method_name = "bm25"
        else:
            model_name = args.embedding_model.replace('-', '_').replace('/', '_')
            method_name = model_name

        filename = f"retrieval_{args.game}_segment_{args.segment_id}_{method_name}_k{args.top_k}.jsonl"

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename

    print(f"💾 输出文件: {output_file}")

    # 执行检索
    results = pipeline.batch_retrieve_qa_pairs(qa_pairs, str(output_file))

    print("✅ 检索完成!")
    print(f"📊 处理了 {len(results)} 个问题")
    print(f"📄 检索结果保存至: {output_file}")

    # 输出统计信息
    success_count = len([r for r in results if 'error' not in r])
    error_count = len(results) - success_count
    print(f"📈 成功: {success_count}, 失败: {error_count}")

    if error_count > 0:
        print("⚠️ 部分检索失败，请检查日志")


if __name__ == '__main__':
    main()
