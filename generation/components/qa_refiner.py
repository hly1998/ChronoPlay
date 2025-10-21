"""
QA问答对优化器组件 - 用于生成流程中的QA优化
从question_refine.py中提取核心功能，集成到生成流程中
"""

import json
import time
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
from prompt import qa_refiner_system_prompt, qa_refiner_user_prompt


class QARefiner:
    """QA问答对优化器组件"""

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str,
                 base_path: str):
        """
        初始化QA优化器

        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model_name: 使用的模型名称
            base_path: 基础路径
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.base_path = base_path

    def get_refine_prompt(self) -> Tuple[str, str]:
        """
        获取优化问答对的系统和用户prompt（从prompt.py导入）

        Returns:
            Tuple[str, str]: (系统prompt, 用户prompt模板)
        """
        return qa_refiner_system_prompt, qa_refiner_user_prompt

    def extract_retrieved_docs_content(self, qa_item: Dict[str, Any]) -> str:
        """
        提取retrieved_docs中的内容

        Args:
            qa_item: QA数据项

        Returns:
            str: 格式化的文档内容
        """
        retrieved_docs = qa_item.get('retrieved_docs', [])
        if not retrieved_docs:
            return "No reference documents available."

        docs_content = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get('content', '').strip()
            if content:
                docs_content.append(f"Document {i}:\n{content}")

        return "\n\n".join(docs_content) if docs_content else "No reference documents available."

    def refine_qa_pair(self, qa_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        优化单个QA问答对

        Args:
            qa_item: 原始QA数据项

        Returns:
            Optional[Dict[str, Any]]: 优化后的QA数据项，如果优化失败返回None
        """
        try:
            # 提取原始问答内容进行长度检查
            original_question = qa_item.get('question', '')
            original_answer = qa_item.get('answer', '')

            # 长度检查
            if len(original_question) > 250:
                print(f"    ⚠️  问题过长({len(original_question)}字符 > 250)，跳过优化")
                return None

            if len(original_answer) > 500:
                print(f"    ⚠️  答案过长({len(original_answer)}字符 > 500)，跳过优化")
                return None

            system_prompt, user_prompt_template = self.get_refine_prompt()

            game_name = qa_item.get('game_name', 'Unknown')
            question_topic = qa_item.get('question_topic', 'Unknown')
            task_type = qa_item.get('task_type', 'Unknown')

            # 提取retrieved_docs内容
            retrieved_docs_content = self.extract_retrieved_docs_content(qa_item)

            # 构造用户prompt
            user_prompt = user_prompt_template.format(
                question=original_question,
                answer=original_answer,
                retrieved_docs_content=retrieved_docs_content,
                game_name=game_name,
                question_type=question_topic,
                task_type=task_type
            )

            print("    🔄 正在优化问答对...")

            # 调用大模型进行优化
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=8000,
                temperature=0.3,  # 较低的temperature确保稳定性
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content.strip()

            # 解析响应
            try:
                # 清理响应文本（移除可能的markdown标记）
                cleaned_text = response_text
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

                optimization_result = json.loads(cleaned_text)

                # 创建优化后的QA项，保留原始数据的其他字段
                refined_qa_item = qa_item.copy()
                refined_qa_item.update({
                    'question': optimization_result.get('optimized_question', original_question),
                    'answer': optimization_result.get('optimized_answer', original_answer),
                    'optimization_notes': optimization_result.get('optimization_notes', ''),
                    'original_question': original_question,  # 保存原始问题
                    'original_answer': original_answer,      # 保存原始答案
                    'optimized': True,                       # 标记已优化
                    'optimization_timestamp': time.time()    # 优化时间戳
                })

                print("    ✅ 优化完成")
                return refined_qa_item

            except json.JSONDecodeError as e:
                print(f"    ❌ JSON解析失败: {e}")
                return None

        except Exception as e:
            print(f"    ❌ 优化失败: {e}")
            return None
