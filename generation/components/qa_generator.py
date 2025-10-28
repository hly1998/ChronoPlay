"""QA对生成器 - 负责生成具体的问答对"""

import json
import random
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
from prompt import system_prompt, user_prompt
# 任务描述将从question_types_map中获取
from utils.utils import clean_response_text, get_question_type_description
from utils.time_extractor import format_qa_time_field
# RoleMatcher将在需要时动态导入，避免sklearn依赖问题


class QAGenerator:
    """QA对生成器类"""

    def __init__(self,
                 client: OpenAI,
                 model_name: str,
                 question_topics_map: Dict[str, Any],
                 question_types_map: Dict[str, Any],
                 role_matcher: Optional[Any] = None):
        """
        初始化QA生成器

        Args:
            client: OpenAI客户端
            model_name: 模型名称
            question_topics_map: 问题主题映射
            question_types_map: 问题类型映射
            role_matcher: 角色匹配器（可选）
        """
        self.client = client
        self.model_name = model_name
        self.question_topics_map = question_topics_map
        self.question_types_map = question_types_map
        # 从question_types_map获取任务描述和任务列表
        self.task_description = question_types_map
        self.task_list = list(question_types_map.keys())
        self.role_matcher = role_matcher

    def _select_task_type_by_probability(self) -> str:
        """
        按指定概率选择任务类型

        Returns:
            str: 选择的任务类型
        """
        # 定义任务类型和对应的概率
        task_type_probabilities = {
            "Extraction-based QA": 0.7,        # 60%
            "Comparative QA": 0.1,             # 20%
            "Multi-hop Reasoning QA": 0.2      # 20%
        }

        available_tasks = []
        probabilities = []

        for task_type, prob in task_type_probabilities.items():
            if task_type in self.task_list:
                available_tasks.append(task_type)
                probabilities.append(prob)

        # 如果没有匹配的任务类型，回退到随机选择（排除黑名单）
        if not available_tasks:
            fallback_tasks = [t for t in self.task_list]
            return random.choice(fallback_tasks) if fallback_tasks else random.choice(self.task_list)

        # 归一化概率（确保总和为1）
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]

        # 使用numpy的random.choice进行概率采样
        # 如果没有numpy，使用自定义的概率采样
        try:
            import numpy as np
            return np.random.choice(available_tasks, p=probabilities)
        except ImportError:
            # 自定义概率采样
            rand_val = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    return available_tasks[i]
            # 备用：返回最后一个任务类型
            return available_tasks[-1]

    def generate_hypothetical_qa_from_template(self, template_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        根据问题模板生成假设的问题和答案

        Args:
            template_data: 模板数据

        Returns:
            (假设问题, 假设答案)
        """
        if not self.client:
            return "", ""

        try:
            template = template_data.get('template', '')
            question_topic = template_data.get('question_topic', 'UNCATEGORIZED')
            game_name = template_data.get('game_name', '游戏名称')
            # 构建prompt，要求模型根据模板生成具体的问题和对应的假设答案
            prompt = f"""
                Based on the following question template, generate a specific question and corresponding hypothetical answer. Please ensure that placeholders in the template are replaced with appropriate content. Pay special attention to the game placeholder [GAME_NAME], please use the correct game name {game_name}.

                Question Template: {template}
                Question Topic: {question_topic}

                Please output in the following format:
                Question: [Generated specific question]
                Answer: [Corresponding hypothetical answer]

                Requirements:
                1. Questions should be specific and clear, conforming to gaming players' expression habits
                2. Answers should be reasonable and helpful
                3. If there are placeholders in the template (such as [GAME_NAME], etc.), please replace them with appropriate content
                4. Answer length should be moderate, both useful and not too lengthy
            """

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional gaming Q&A assistant capable of generating specific questions and reasonable hypothetical questions and answers based on question templates."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )

            response_text = response.choices[0].message.content.strip()

            # 解析响应，提取问题和答案
            lines = response_text.split('\n')
            question = ""
            answer = ""

            for line in lines:
                line = line.strip()
                if line.startswith('Question：') or line.startswith('Question:'):
                    question = line.replace('Question：', '').replace('Question:', '').strip()
                elif line.startswith('Answer：') or line.startswith('Answer:'):
                    answer = line.replace('Answer：', '').replace('Answer:', '').strip()

            # 如果解析失败，尝试简单的分割
            if not question or not answer:
                parts = response_text.split('\n', 1)
                if len(parts) >= 2:
                    question = parts[0].strip()
                    answer = parts[1].strip()
                else:
                    # 最后的备选方案：使用模板作为问题，响应作为答案
                    question = template
                    answer = response_text

            return question, answer

        except Exception as e:
            print(f"Failed to generate hypothetical Q&A pair: {e}")
            return "", ""

    def generate_qa_pair(self,
                         template_data: Dict[str, Any],
                         retrieved_docs: List[Dict[str, Any]],
                         selected_task_type: str = None,
                         segment_id: int = None) -> List[Dict[str, Any]]:
        """
        根据问题模板生成新的问答对

        Args:
            template_data: 问题模板数据
            retrieved_docs: 检索到的文档列表
            selected_task_type: 选择的任务类型

        Returns:
            生成的问答对
        """

        # 构建文档内容和实体信息
        docs_content = ""
        all_entities = []
        for i, doc in enumerate(retrieved_docs, 1):
            if isinstance(doc, dict) and 'content' in doc:
                content = doc['content']
                if content:  # 只有当内容不为空时才添加
                    docs_content += f"Reference Material {i}:\n{content}\n\n"

                    # 从corpus.jsonl文件中直接获取预处理的实体信息
                    # 由于索引中只保留了entity_texts以减少metadata长度，我们需要重构实体格式
                    if 'metadata' in doc:
                        metadata = doc['metadata']
                        if 'entity_texts' in metadata:
                            # 索引中的压缩格式：只有实体文本
                            entity_texts = metadata['entity_texts']
                            for entity_text in entity_texts:
                                if entity_text.strip():
                                    # 重构为标准实体格式
                                    entity_obj = {
                                        'text': entity_text,
                                        'type': 'UNKNOWN',  # 索引中没有保存类型信息
                                        'context': ''  # 索引中没有保存上下文信息
                                    }
                                    all_entities.append(entity_obj)
                        elif 'entities' in metadata:
                            # 完整格式（如果有的话）
                            all_entities.extend(metadata['entities'])

        # 如果所有文档内容都为空，记录警告
        if not docs_content.strip():
            print("    ⚠️  Warning: Retrieved document content is empty, may affect QA quality")

        # 实体信息将在后处理阶段从corpus文件中直接获取，无需在prompt中提及

        # 获取问题主题的详细描述
        question_topic = template_data.get('question_topic', 'UNCATEGORIZED')
        question_topic_description = get_question_type_description(
            question_topic, self.question_topics_map)

        # 获取任务描述（如果没有指定任务类型，则按概率选择）
        if selected_task_type is None:
            selected_task_type = self._select_task_type_by_probability()

        task_desc = self.task_description.get(selected_task_type, "")

        # 获取匹配的角色信息
        role_data = None
        role_context = ""
        if self.role_matcher:
            role_data = self.role_matcher.find_matching_role(template_data)
            if role_data:
                role_context = self.role_matcher.get_role_context(role_data)

        # 构建用户提示词，包含角色信息和模板信息
        user_prompt_str = self._build_role_aware_prompt(
            question_topic_description, selected_task_type, task_desc,
            docs_content, role_context, template_data
        )

        try:
            # 注释掉冗长的输出，只保留关键信息
            # print("user_prompt_str: ", user_prompt_str)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt_str
                    }
                ],
                max_tokens=2500,
                temperature=0.5,
                response_format={"type": "json_object"}
            )

            # 解析响应
            response_text = response.choices[0].message.content.strip()
            # print("response_text: ", response_text)
            cleaned_text = clean_response_text(response_text)
            qa_data = json.loads(cleaned_text)

            # 现在qa_data是单个对象而不是列表，需要转换为列表格式以保持兼容性
            if isinstance(qa_data, dict):
                qa_list = [qa_data]
            else:
                qa_list = qa_data  # 备用情况，如果仍然是列表

            # 添加模板信息和角色信息，并确保实体信息正确提取
            for qa in qa_list:
                # 移除不需要保存的思考过程字段
                qa.pop('###THOUGHT_PROCESS###', None)

                # 如果QA对中没有实体信息，从可用实体中提取相关的
                if 'entities' not in qa or not qa['entities']:
                    qa['entities'] = self._extract_relevant_entities(qa, all_entities)

                # 从检索到的文档中提取时间信息，覆盖大模型生成的时间
                extracted_time = format_qa_time_field(retrieved_docs, segment_id)
                qa['time'] = extracted_time

                qa.update({
                    'template_id': template_data.get('id'),
                    'template': template_data.get('template'),
                    'question_topic': question_topic,
                    'task_type': selected_task_type,
                    'retrieved_docs': retrieved_docs,
                    'source_template': template_data,
                    'role_data': role_data if role_data else None
                })

            return qa_list

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"生成QA对失败: {e}")
            return []

    def _build_role_aware_prompt(self,
                                 question_topic_description: str,
                                 selected_task_type: str,
                                 task_desc: str,
                                 docs_content: str,
                                 role_context: str,
                                 template_data: Dict[str, Any]) -> str:
        """
        构建包含角色信息和模板信息的提示词

        Args:
            question_topic_description: 问题主题描述
            selected_task_type: 选择的任务类型
            task_desc: 任务描述
            docs_content: 文档内容
            role_context: 角色上下文
            template_data: 问题模板数据

        Returns:
            构建好的提示词
        """
        # 构建角色上下文部分，如果有角色信息则格式化，否则为空
        if role_context:
            formatted_role_context = f"""
                ## Player Role Background
                {role_context}

                **Important Note**: Please generate questions completely from the perspective of the above player role, ensuring questions reflect the role's:
                - Language expression style and habits
                - Gaming experience level and focus points
                - Personal characteristics and preferences
                - Mindset and motivation when asking questions
            """
        else:
            formatted_role_context = ""

        # 构建模板上下文部分
        template_context = self._build_template_context(template_data)

        # 使用统一的user_prompt模板，根据是否有角色信息和模板信息来填充不同的内容
        return user_prompt.format(
            topic_description=question_topic_description,
            task_name=selected_task_type,
            task_require=task_desc,
            role_context=formatted_role_context,
            template_context=template_context,
            doc_str=docs_content
        )

    def _build_template_context(self, template_data: Dict[str, Any]) -> str:
        """
        构建问题模板上下文信息

        Args:
            template_data: 问题模板数据

        Returns:
            格式化的模板上下文字符串
        """
        template_text = template_data.get('template', '')
        description = template_data.get('description', '')
        placeholders = template_data.get('placeholders', [])

        if not template_text:
            return ""

        template_context = f"""
## Question Template Guidance
### Template Example
{template_text}

### Template Description
{description if description else 'Generate questions based on the structure and style of this template'}

**Generation Guidelines**:
1. **Reference Template Structure**: Generated questions should follow the expression and structural characteristics of the above template
2. **Maintain Style Consistency**: The language style of questions should be consistent with the template, reflecting the same expression habits
3. **Appropriate Variation**: While maintaining the template spirit, make reasonable changes and expansions based on specific document content
4. **Player Perspective**: Ensure generated questions are raised from a real player's perspective, not a simple copy of the template"""

        if placeholders:
            placeholder_text = ", ".join(placeholders)
            template_context += f"""
5. **Placeholder Understanding**: The template contains placeholders ({placeholder_text}), please understand their meaning and use them flexibly when generating questions"""

        return template_context.strip()

    def _extract_relevant_entities(self, qa_pair: Dict[str, Any],
                                   all_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从corpus.jsonl文件预处理的实体中筛选与QA对相关的实体

        注意：实体信息来源于corpus文件中的metadata.entities字段，
        这些实体已经预先提取并存储，无需通过LLM进行实时提取

        Args:
            qa_pair: QA对
            all_entities: 从corpus文件中获取的所有实体列表

        Returns:
            与当前QA对相关的实体列表
        """
        if not all_entities:
            return []

        relevant_entities = []
        question_text = qa_pair.get('question', '').lower()
        answer_text = qa_pair.get('answer', '').lower()
        combined_text = question_text + ' ' + answer_text

        for entity in all_entities:
            entity_text = entity.get('text', '').lower()
            # 检查实体文本是否在问题或答案中出现
            if entity_text and entity_text in combined_text:
                relevant_entities.append(entity)

        return relevant_entities
