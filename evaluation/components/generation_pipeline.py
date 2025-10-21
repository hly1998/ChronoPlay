# -*- coding: utf-8 -*-
"""
独立生成Pipeline
专门负责基于检索结果生成答案，支持多种生成模型的实验对比
"""

import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from .config import GenerationConfig
from .generator import TextGenerator


class GenerationPipeline:
    """独立的生成Pipeline"""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.generator = TextGenerator(config)

    def generate_single(self, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """基于单个检索结果生成答案"""
        start_time = time.time()

        question = retrieval_result['question']
        retrieved_docs = retrieval_result['retrieved_docs']

        # 生成回答
        answer = self.generator.generate(question, retrieved_docs)

        # 构建生成结果
        result = {
            'question': question,
            'answer': answer,
            'generation_time': time.time() - start_time,
            'generation_config': {
                'llm_model': self.config.llm_model,
                'temperature': self.config.temperature
            },
            'retrieved_docs': retrieved_docs,
            'retrieved_doc_ids': retrieval_result['retrieved_doc_ids'],
            'retrieval_time': retrieval_result.get('retrieval_time', 0),
            'total_processing_time': retrieval_result.get('retrieval_time', 0) + (time.time() - start_time)
        }

        # 保留评估信息
        if 'ground_truth_answer' in retrieval_result:
            result['ground_truth_answer'] = retrieval_result['ground_truth_answer']
            result['ground_truth_doc_ids'] = retrieval_result['ground_truth_doc_ids']

        if 'question_index' in retrieval_result:
            result['question_index'] = retrieval_result['question_index']

        return result

    async def generate_single_async(self, retrieval_result: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """基于单个检索结果异步生成答案"""
        start_time = time.time()

        question = retrieval_result['question']
        retrieved_docs = retrieval_result['retrieved_docs']

        # 异步生成回答
        answer = await self.generator.generate_async(question, retrieved_docs)

        # 构建生成结果
        result = {
            'question': question,
            'answer': answer,
            'generation_time': time.time() - start_time,
            'generation_config': {
                'llm_model': self.config.llm_model,
                'temperature': self.config.temperature
            },
            'retrieved_docs': retrieved_docs,
            'retrieved_doc_ids': retrieval_result['retrieved_doc_ids'],
            'retrieval_time': retrieval_result.get('retrieval_time', 0),
            'total_processing_time': retrieval_result.get('retrieval_time', 0) + (time.time() - start_time)
        }

        # 保留评估信息
        if 'ground_truth_answer' in retrieval_result:
            result['ground_truth_answer'] = retrieval_result['ground_truth_answer']
            result['ground_truth_doc_ids'] = retrieval_result['ground_truth_doc_ids']

        if 'question_index' in retrieval_result:
            result['question_index'] = retrieval_result['question_index']

        return result

    def batch_generate(self, retrieval_results: List[Dict[str, Any]], output_file: str = None) -> List[Dict[str, Any]]:
        """批量生成答案 - 使用并行模式"""
        # 运行异步并行生成
        return asyncio.run(self.batch_generate_parallel(retrieval_results, output_file))

    async def batch_generate_parallel(
            self, retrieval_results: List[Dict[str, Any]], output_file: str = None) -> List[Dict[str, Any]]:
        """并行批量生成答案"""
        total = len(retrieval_results)
        all_results = []

        print(f"⚡ 开始生成 {total} 个结果，并发: {self.config.concurrent_requests}")

        # 分批处理
        for batch_start in range(0, total, self.config.concurrent_requests):
            batch_end = min(batch_start + self.config.concurrent_requests, total)
            batch_retrieval_results = retrieval_results[batch_start:batch_end]

            print(f"处理 {batch_start+1}-{batch_end}/{total}")

            # 创建并发任务
            tasks = [
                self._generate_single_with_error_handling(r, batch_start + i)
                for i, r in enumerate(batch_retrieval_results)
            ]

            # 并发执行
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # 处理异常
                    error_result = {
                        'question': batch_retrieval_results[i].get('question', ''),
                        'answer': f"生成失败: {str(result)}",
                        'error': str(result),
                        'retrieved_docs': batch_retrieval_results[i].get('retrieved_docs', []),
                        'retrieved_doc_ids': batch_retrieval_results[i].get('retrieved_doc_ids', [])
                    }

                    if 'question_index' in batch_retrieval_results[i]:
                        error_result['question_index'] = batch_retrieval_results[i]['question_index']
                    if 'ground_truth_answer' in batch_retrieval_results[i]:
                        error_result['ground_truth_answer'] = batch_retrieval_results[i]['ground_truth_answer']
                        error_result['ground_truth_doc_ids'] = batch_retrieval_results[i]['ground_truth_doc_ids']

                    all_results.append(error_result)

                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                else:
                    all_results.append(result)

                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        return all_results

    async def _generate_single_with_error_handling(
            self, retrieval_result: Dict[str, Any], index: int) -> Dict[str, Any]:
        """带错误处理的单次异步生成"""
        return await self.generate_single_async(retrieval_result, index)

    def generate_from_file(self, retrieval_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """从检索结果文件生成答案"""
        # 读取检索结果
        retrieval_results = []
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    retrieval_results.append(json.loads(line.strip()))

        # 批量生成
        return self.batch_generate(retrieval_results, output_file)

    @staticmethod
    def load_retrieval_results(retrieval_file: str) -> List[Dict[str, Any]]:
        """加载检索结果文件"""
        results = []
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))
        return results

    def create_generation_experiment(self,
                                     retrieval_file: str,
                                     experiment_name: str = None,
                                     output_dir: str = "./results") -> str:
        """创建生成实验，返回输出文件路径"""
        if experiment_name is None:
            model_name = self.config.llm_model.replace('/', '_').replace('-', '_')
            experiment_name = f"generation_{model_name}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{experiment_name}.jsonl"

        # 执行生成
        self.generate_from_file(retrieval_file, str(output_file))

        return str(output_file)
