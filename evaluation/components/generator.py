# -*- coding: utf-8 -*-
"""
文本生成器组件
"""

import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from openai import OpenAI


class TextGenerator:
    """简化的文本生成器"""

    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.async_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    def generate(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """生成回答"""
        docs = docs[:3]

        # 构建文档内容
        docs_text = ""
        for i, doc in enumerate(docs, 1):
            title = doc.get('metadata', {}).get('title', '无标题')
            docs_text += f"=== 参考资料 {i} ===\n标题: {title}\n内容: {doc['content']}\n\n"

        user_prompt = f"""Context information is below:
{docs_text}
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:
"""

        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature
        )
        return response.choices[0].message.content.strip()

    async def generate_async(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """异步生成回答"""
        docs = docs[:3]

        # 构建文档内容
        docs_text = ""
        for i, doc in enumerate(docs, 1):
            title = doc.get('metadata', {}).get('title', '无标题')
            docs_text += f"=== 参考资料 {i} ===\n标题: {title}\n内容: {doc['content']}\n\n"

        user_prompt = f"""Context information is below:
{docs_text}
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:
"""

        # 重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"生成失败: {str(e)}"
                await asyncio.sleep(1 * (attempt + 1))
