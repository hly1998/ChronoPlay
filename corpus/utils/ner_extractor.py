"""游戏实体识别模块"""

import json
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


class GameNERExtractor:
    """游戏相关实体识别器，使用SELF-ICL技术提升准确性"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 use_self_icl: bool = True,
                 num_pseudo_examples: int = 3,
                 max_workers: int = 3):
        """
        初始化NER提取器

        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL（用于自定义端点）
            model: 使用的模型名称
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            use_self_icl: 是否使用SELF-ICL技术
            num_pseudo_examples: 生成的伪示例数量
            max_workers: 并发处理的最大线程数
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_self_icl = use_self_icl
        self.num_pseudo_examples = num_pseudo_examples
        self.max_workers = max_workers

        # 游戏实体类型定义
        self.entity_types = {
            "CHARACTER": "Characters, NPCs, players, protagonists, companions",
            "ITEM": "Items, equipment, weapons, tools, consumables, gear",
            "LOCATION": "Locations, places, areas, maps, regions, zones",
            "QUEST": "Quests, missions, tasks, objectives, storylines",
            "SKILL": "Skills, abilities, talents, powers, upgrades",
            "FACTION": "Factions, organizations, groups, guilds, clans",
            "MONSTER": "Monsters, enemies, bosses, creatures, opponents",
            "MECHANIC": "Game mechanics, gameplay systems, features, rules",
            "EVENT": "Events, story events, plot points, scenarios"
        }

    def _generate_pseudo_inputs(self, text: str) -> List[str]:
        """
        SELF-ICL步骤1: 生成伪输入
        根据给定的测试输入生成相似的伪输入示例
        """
        if not self.client:
            return []

        prompt = f"""You are tasked with generating pseudo-inputs for game-related Named Entity Recognition (NER).

Given the following game-related text, generate {self.num_pseudo_examples} similar but different game-related text examples that would be suitable for entity recognition. The generated texts should:
1. Be similar in style and domain to the input text
2. Contain various types of game entities
3. Be realistic and coherent
4. Have different specific entities but similar context patterns

Original text:
{text}...

Return the result in the following JSON format only, no other text:
{{
    "pseudo_inputs": [
        "example text 1",
        "example text 2",
        "example text 3"
    ]
}}

Generate exactly {self.num_pseudo_examples} pseudo-input examples."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in generating game-related text examples for NER training. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            response_text = response.choices[0].message.content.strip()
            result = self._parse_json_response(response_text)

            if isinstance(result, dict) and "pseudo_inputs" in result:
                pseudo_inputs = result["pseudo_inputs"]
                if isinstance(pseudo_inputs, list):
                    return pseudo_inputs[:self.num_pseudo_examples]

            return []

        except Exception:
            return []

    def _predict_pseudo_labels(self, pseudo_inputs: List[str]) -> List[List[Dict]]:
        """
        SELF-ICL步骤2: 预测伪标签
        使用零样本提示为伪输入预测实体标签
        """
        pseudo_labels = []

        for pseudo_input in pseudo_inputs:
            entities = self._extract_entities_zero_shot(pseudo_input)
            pseudo_labels.append(entities)

        return pseudo_labels

    def _extract_entities_zero_shot(self, text: str) -> List[Dict]:
        """
        使用零样本提示提取实体（用于生成伪标签）
        """
        entity_desc = "\n".join([f"- {k}: {v}" for k, v in self.entity_types.items()])

        prompt = f"""Extract game-related entities from the following text.

Entity Types:
{entity_desc}

Text: {text}

Return the result in the following JSON format only, no other text:
{{
    "entities": [
        {{
            "text": "entity text",
            "type": "ENTITY_TYPE",
            "context": "brief context"
        }}
    ]
}}

If no entities are found, return: {{"entities": []}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a game entity extraction expert. Return only valid JSON in the specified format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )

            response_text = response.choices[0].message.content.strip()
            result = self._parse_json_response(response_text)

            if isinstance(result, dict) and "entities" in result:
                entities = result["entities"]
                if isinstance(entities, list):
                    return entities

            return []

        except Exception:
            return []

    def _parse_json_response(self, response_text: str) -> Dict:
        """通用JSON解析函数，增强容错性"""
        try:
            # 尝试直接解析
            result = json.loads(response_text)
            if isinstance(result, dict):
                return result
            return {}
        except json.JSONDecodeError:
            # 尝试多种方式提取JSON
            import re

            # 1. 尝试提取```json代码块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass

            # 2. 尝试提取花括号内容
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass

            # 3. 尝试简单的键值对提取（作为最后的备份）
            entities_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', response_text, re.DOTALL)
            if entities_match:
                try:
                    entities_str = f'[{entities_match.group(1)}]'
                    entities = json.loads(entities_str)
                    return {"entities": entities}
                except json.JSONDecodeError:
                    pass

            return {}

    def _parse_json_entities(self, response_text: str) -> List[str]:
        """解析JSON格式的实体响应，返回简化的实体名称列表"""
        result = self._parse_json_response(response_text)
        if "entities" in result:
            entities = result["entities"]
            # 如果是字符串列表，直接返回
            if isinstance(entities, list) and all(isinstance(e, str) for e in entities):
                return entities
            # 如果是字典列表（旧格式），提取text字段
            elif isinstance(entities, list) and all(isinstance(e, dict) for e in entities):
                return [e.get("text", "") for e in entities if e.get("text")]
        return []

    def _create_icl_prompt(self, test_text: str, pseudo_examples: List[tuple]) -> str:
        """
        SELF-ICL步骤3: 创建包含演示示例的ICL提示

        Args:
            test_text: 测试文本
            pseudo_examples: (伪输入, 伪标签) 对的列表
        """
        entity_desc = "\n".join([f"- {k}: {v}" for k, v in self.entity_types.items()])

        # 构建演示示例
        demonstrations = []
        for i, (pseudo_input, pseudo_labels) in enumerate(pseudo_examples):
            demo_entities = []
            for entity in pseudo_labels:
                if isinstance(entity, dict) and 'text' in entity and 'type' in entity:
                    demo_entities.append({
                        "text": entity['text'],
                        "type": entity['type'],
                        "context": entity.get('context', '')
                    })

            demo_json = json.dumps({"entities": demo_entities}, ensure_ascii=False, indent=2)
            demonstrations.append(f"Example {i+1}:\nInput: {pseudo_input}\nOutput: {demo_json}")

        demonstrations_text = "\n\n".join(demonstrations)

        prompt = f"""You are a specialized assistant for identifying game-related entities. Learn from the following examples and then extract entities from the test input.

Entity Type Definitions:
{entity_desc}

Here are some examples:

{demonstrations_text}

Now, extract entities from the following test input.

Test Input: {test_text}

Return the result in the following JSON format only, no other text:
{{
    "entities": [
        {{
            "text": "entity text",
            "type": "ENTITY_TYPE",
            "context": "brief context"
        }}
    ]
}}

If no entities are found, return: {{"entities": []}}
"""
        return prompt

    def _create_ner_prompt(self, text: str) -> str:
        """创建NER识别的提示词"""
        entity_desc = "\n".join([f"- {k}: {v}" for k, v in self.entity_types.items()])

        prompt = f"""You are a specialized assistant for identifying game-related entities. Please identify all game-related entities from the following text and classify them according to the specified types.

Entity Type Definitions:
{entity_desc}

Text Content:
{text}

Return the result in the following JSON format only, no other text:
{{
    "entities": [
        "entity_name_1",
        "entity_name_2"
    ]
}}

Instructions:
1. Only identify clear game-related entities
2. DO NOT extract game titles or game names (e.g., "PUBG Mobile", "Dune", "Dying Light 2")
3. Focus on in-game entities like characters, items, locations, etc.
4. Return only the entity names as strings in a simple list
5. If no entities are found, return: {{"entities": []}}
"""
        return prompt

    def extract_entities(self, text: str) -> List[str]:
        """
        从文本中提取游戏相关实体，支持SELF-ICL技术

        Args:
            text: 输入文本

        Returns:
            提取的实体列表
        """
        if not self.client:
            return []

        if not text.strip():
            return []

        # 限制文本长度以避免token限制
        max_chars = 3000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        # 根据配置选择使用SELF-ICL或传统方法
        if self.use_self_icl:
            return self._extract_entities_with_self_icl(text)
        else:
            return self._extract_entities_traditional(text)

    def _extract_entities_with_self_icl(self, text: str) -> List[str]:
        """
        使用SELF-ICL技术提取实体的完整流程
        """
        # 创建总体进度条
        with tqdm(total=3, desc="SELF-ICL提取", leave=False) as pbar:
            for attempt in range(self.max_retries):
                try:
                    # 步骤1: 生成伪输入
                    pbar.set_description("步骤1: 生成伪输入")
                    pseudo_inputs = self._generate_pseudo_inputs(text)

                    if not pseudo_inputs:
                        return self._extract_entities_traditional(text)
                    pbar.update(1)

                    # 步骤2: 预测伪标签
                    pbar.set_description(f"步骤2: 预测{len(pseudo_inputs)}个标签")
                    pseudo_labels = self._predict_pseudo_labels(pseudo_inputs)

                    # 创建伪示例对
                    pseudo_examples = []
                    for pseudo_input, labels in zip(pseudo_inputs, pseudo_labels):
                        if labels:  # 只包含有实体的示例
                            pseudo_examples.append((pseudo_input, labels))

                    if not pseudo_examples:
                        return self._extract_entities_traditional(text)
                    pbar.update(1)

                    # 步骤3: 执行ICL
                    pbar.set_description(f"步骤3: 使用{len(pseudo_examples)}个示例")
                    icl_prompt = self._create_icl_prompt(text, pseudo_examples)

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a professional game entity recognition assistant. Learn from the examples and extract entities accurately."},
                            {"role": "user", "content": icl_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1500
                    )

                    response_text = response.choices[0].message.content.strip()
                    entities = self._parse_json_entities(response_text)
                    pbar.update(1)
                    return entities

                except Exception:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        return self._extract_entities_traditional(text)

    def _extract_entities_traditional(self, text: str) -> List[str]:
        """
        传统的实体提取方法（作为回退选项）
        """
        prompt = self._create_ner_prompt(text)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional game entity recognition assistant. Extract game-related entities accurately and follow the specified format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )

                response_text = response.choices[0].message.content.strip()
                entities = self._parse_json_entities(response_text)
                return entities

            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return []

    def batch_extract_entities(self, text_chunks: List[str], show_progress: bool = True) -> List[List[str]]:
        """
        批量提取实体（支持并发处理）

        Args:
            text_chunks: 文本块列表
            show_progress: 是否显示进度

        Returns:
            每个文本块对应的实体列表
        """
        if not text_chunks:
            return []

        # 如果只有一个文本块或没有配置客户端，使用串行处理
        if len(text_chunks) == 1 or not self.client:
            return self._batch_extract_serial(text_chunks, show_progress)

        # 使用并发处理
        return self._batch_extract_concurrent(text_chunks, show_progress)

    def _batch_extract_serial(self, text_chunks: List[str], show_progress: bool = True) -> List[List[Dict]]:
        """串行批量提取实体（原始方法）"""
        results = []

        if show_progress:
            from tqdm import tqdm
            text_chunks = tqdm(text_chunks, desc="NER提取")

        for text in text_chunks:
            entities = self.extract_entities(text)
            results.append(entities)

            # 添加短暂延迟以避免API限流
            if self.client:
                time.sleep(0.1)

        return results

    def _batch_extract_concurrent(self, text_chunks: List[str], show_progress: bool = True) -> List[List[Dict]]:
        """并发批量提取实体"""
        results = [None] * len(text_chunks)  # 预分配结果列表以保持顺序
        # 创建进度条
        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(text_chunks), desc="NER并发提取")

        def extract_with_index(args):
            """带索引的提取函数"""
            index, text = args
            entities = self.extract_entities(text)
            return index, entities

        # 使用ThreadPoolExecutor进行并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务，包含索引以保持顺序
            indexed_chunks = [(i, text) for i, text in enumerate(text_chunks)]
            future_to_index = {
                executor.submit(extract_with_index, chunk): chunk[0]
                for chunk in indexed_chunks
            }

            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_index):
                try:
                    index, entities = future.result()
                    results[index] = entities
                    completed_count += 1

                    if show_progress:
                        pbar.update(1)
                        pbar.set_postfix({"完成": f"{completed_count}/{len(text_chunks)}"})
                except Exception:
                    index = future_to_index[future]
                    results[index] = []  # 失败时返回空列表

        if show_progress:
            pbar.close()

        return results

    def get_entity_statistics(self, all_entities: List[List[str]]) -> Dict:
        """
        获取实体统计信息

        Args:
            all_entities: 所有文本块的实体列表

        Returns:
            统计信息字典
        """
        stats = {
            "total_chunks": len(all_entities),
            "total_entities": 0,
            "unique_entities": set(),
            "chunks_with_entities": 0
        }

        for entities in all_entities:
            if entities:
                stats["chunks_with_entities"] += 1
                stats["total_entities"] += len(entities)

                for entity in entities:
                    if entity.strip():
                        stats["unique_entities"].add(entity.strip().lower())

        stats["unique_entity_count"] = len(stats["unique_entities"])
        stats["unique_entities"] = list(stats["unique_entities"])  # 转换为列表以便JSON序列化

        return stats
