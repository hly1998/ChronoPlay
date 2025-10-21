#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA生成工具的辅助函数库
包含文本处理、JSON解析等功能性函数
"""

import re
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path


def clean_response_text(text: str) -> str:
    """
    清理响应文本，移除markdown标记等

    Args:
        text: 原始响应文本

    Returns:
        清理后的文本
    """
    # 首先尝试提取被<json></json>包围的内容
    json_block_pattern = r'<json>\s*([\s\S]*?)\s*</json>'
    json_block_match = re.search(json_block_pattern, text, re.DOTALL)

    if json_block_match:
        # 提取到被<json></json>包围的内容
        extracted_content = json_block_match.group(1).strip()
        return extracted_content

    # 尝试提取被```json ```包围的内容
    json_code_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_code_block_match = re.search(json_code_block_pattern, text, re.DOTALL)

    if json_code_block_match:
        # 提取到被```json ```包围的内容
        extracted_content = json_code_block_match.group(1).strip()
        return extracted_content

    # 尝试提取任何```包围的内容（作为备用）
    code_block_pattern = r'```\s*([\s\S]*?)\s*```'
    code_block_match = re.search(code_block_pattern, text, re.DOTALL)

    if code_block_match:
        extracted_content = code_block_match.group(1).strip()
        # 检查是否看起来像JSON
        if extracted_content.startswith('{') and extracted_content.endswith('}'):
            return extracted_content

    # 如果没有找到任何代码块，尝试直接查找JSON对象
    json_object_pattern = r'\{[\s\S]*\}'
    json_object_match = re.search(json_object_pattern, text, re.DOTALL)

    if json_object_match:
        return json_object_match.group().strip()

    # 如果都没有找到，返回空字符串
    return ""


def fix_truncated_json(json_text: str) -> str:
    """
    修复可能被截断的JSON

    Args:
        json_text: 可能被截断的JSON文本

    Returns:
        修复后的JSON文本
    """
    # 如果JSON以...结尾，尝试修复
    if json_text.rstrip().endswith('...'):
        # 移除...
        json_text = json_text.rstrip()[:-3]

        # 尝试补全常见的截断情况
        # 如果在字符串中间截断，补全引号
        if json_text.count('"') % 2 != 0:
            json_text += '"'

        # 补全可能缺失的}
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        if open_braces > close_braces:
            json_text += '}' * (open_braces - close_braces)

        # 补全可能缺失的]
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')
        if open_brackets > close_brackets:
            json_text += ']' * (open_brackets - close_brackets)

    return json_text


def parse_openai_response(response_text: str, question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    解析OpenAI响应（用于模板生成）

    Args:
        response_text: OpenAI API响应文本
        question_data: 问题数据

    Returns:
        解析后的模板列表
    """
    templates = []

    try:
        # 清理响应文本
        cleaned_text = clean_response_text(response_text)

        # 尝试直接解析JSON
        response_json = json.loads(cleaned_text)

        if "templates" in response_json:
            for template_data in response_json["templates"]:
                template = {
                    'id': str(uuid.uuid4()),
                    'source_question_id': question_data['id'],
                    'template': template_data.get('template', ''),
                    'placeholders': template_data.get('placeholders', []),
                    'description': template_data.get('description', ''),
                    'question_type': question_data['question_type'],
                    'topic': question_data['topic'],
                    'created_time': datetime.now().isoformat(),
                    'generation_method': 'openai'
                }
                templates.append(template)

    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        print(f"原始响应前200字符: {response_text[:200]}")

        # 尝试多种解析方式
        templates = try_alternative_parsing(response_text, question_data)

        if not templates:
            # 创建一个基础模版作为备用
            template = {
                'id': str(uuid.uuid4()),
                'source_question_id': question_data['id'],
                'template': question_data['refined_question'],
                'placeholders': [],
                'description': f"基于{question_data['question_type']}的基础模版（解析失败备用）",
                'question_type': question_data['question_type'],
                'topic': question_data['topic'],
                'created_time': datetime.now().isoformat(),
                'generation_method': 'fallback'
            }
            templates.append(template)

    return templates


def try_alternative_parsing(response_text: str, question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    尝试多种方式解析响应

    Args:
        response_text: 响应文本
        question_data: 问题数据

    Returns:
        解析后的模板列表
    """
    templates = []

    # 方法1: 查找JSON块
    try:
        # 查找{}包围的JSON内容
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group()
            # 清理可能的截断问题
            json_text = fix_truncated_json(json_text)
            response_json = json.loads(json_text)

            if "templates" in response_json:
                for template_data in response_json["templates"]:
                    template = {
                        'id': str(uuid.uuid4()),
                        'source_question_id': question_data['id'],
                        'template': template_data.get('template', ''),
                        'placeholders': template_data.get('placeholders', []),
                        'description': template_data.get('description', ''),
                        'question_type': question_data['question_type'],
                        'topic': question_data['topic'],
                        'created_time': datetime.now().isoformat(),
                        'generation_method': 'openai_alt1'
                    }
                    templates.append(template)

            print("✅ 使用替代解析方法1成功")
            return templates

    except Exception as e:
        print(f"替代解析方法1失败: {e}")

    # 方法2: 处理单个问答对（不是templates格式）
    try:
        cleaned_text = clean_response_text(response_text)
        # 修复可能的截断
        cleaned_text = fix_truncated_json(cleaned_text)

        response_json = json.loads(cleaned_text)

        # 如果直接是问答对格式
        if "question" in response_json and "answer" in response_json:
            template = {
                'id': str(uuid.uuid4()),
                'source_question_id': question_data['id'],
                'template': response_json.get('question', ''),
                'placeholders': response_json.get('placeholders', []),
                'description': response_json.get('description', '基于单个QA对生成的模版'),
                'question_type': question_data['question_type'],
                'topic': question_data['topic'],
                'created_time': datetime.now().isoformat(),
                'generation_method': 'openai_alt2'
            }
            templates.append(template)

            print("✅ 使用替代解析方法2成功")
            return templates

    except Exception as e:
        print(f"替代解析方法2失败: {e}")

    # 方法3: 提取问题文本（最基础的方法）
    try:
        # 使用正则表达式提取问题
        question_pattern = r'"question":\s*"([^"]+)"'
        question_match = re.search(question_pattern, response_text)

        if question_match:
            extracted_question = question_match.group(1)
            template = {
                'id': str(uuid.uuid4()),
                'source_question_id': question_data['id'],
                'template': extracted_question,
                'placeholders': [],
                'description': "从响应文本中提取的问题",
                'question_type': question_data['question_type'],
                'topic': question_data['topic'],
                'created_time': datetime.now().isoformat(),
                'generation_method': 'openai_regex'
            }
            templates.append(template)

            print("✅ 使用正则提取方法成功")
            return templates

    except Exception as e:
        print(f"正则提取方法失败: {e}")

    print("❌ 所有解析方法都失败")
    return templates


def parse_qa_response_alternative(response_text: str, original_question: str,
                                  question_type: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    QA对响应的备用解析方法

    Args:
        response_text: 响应文本
        original_question: 原始问题
        question_type: 问题类型
        retrieved_docs: 检索到的文档列表

    Returns:
        解析后的QA对数据
    """

    # 方法1: 清理后重新尝试JSON解析
    try:
        cleaned_text = clean_response_text(response_text)
        cleaned_text = fix_truncated_json(cleaned_text)

        qa_data = json.loads(cleaned_text)

        # 验证必要字段存在
        if qa_data.get('question') and qa_data.get('answer'):
            qa_data.update({
                'id': str(uuid.uuid4()),
                'original_question': original_question,
                'question_type': question_type,
                'retrieved_docs': retrieved_docs,
                'generated_time': datetime.now().isoformat(),
                'generation_method': 'retrieval_augmented_alt1'
            })
            print("✅ 备用解析方法1成功")
            return qa_data

    except Exception as e:
        print(f"备用解析方法1失败: {e}")

    # 方法2: 正则表达式提取
    try:
        # 提取问题
        question_pattern = r'"question":\s*"([^"]*(?:\\.[^"]*)*)"'
        question_match = re.search(question_pattern, response_text, re.DOTALL)

        # 提取答案（可能跨多行）
        answer_pattern = r'"answer":\s*"([^"]*(?:\\.[^"]*)*(?:\.\.\.)?)(?:"|$)'
        answer_match = re.search(answer_pattern, response_text, re.DOTALL)

        # 提取引用（可选）
        references_pattern = r'"references":\s*\[(.*?)\]'
        references_match = re.search(references_pattern, response_text, re.DOTALL)

        if question_match and answer_match:
            question = question_match.group(1)
            answer = answer_match.group(1)

            # 处理可能的转义字符
            question = question.replace('\\"', '"').replace('\\n', '\n')
            answer = answer.replace('\\"', '"').replace('\\n', '\n')

            # 如果答案以...结尾，说明被截断了，尝试补全
            if answer.endswith('...'):
                answer = answer[:-3] + "（答案可能被截断）"

            # 处理引用
            references = []
            if references_match:
                refs_text = references_match.group(1)
                # 简单的引用提取
                ref_items = re.findall(r'"([^"]+)"', refs_text)
                references = ref_items

            qa_data = {
                'id': str(uuid.uuid4()),
                'question': question,
                'answer': answer,
                'references': references,
                'original_question': original_question,
                'question_type': question_type,
                'retrieved_docs': retrieved_docs,
                'generated_time': datetime.now().isoformat(),
                'generation_method': 'retrieval_augmented_regex'
            }

            print("✅ 备用解析方法2（正则提取）成功")
            return qa_data

    except Exception as e:
        print(f"备用解析方法2失败: {e}")

    # 方法3: 基础文本提取（最后的备用方案）
    try:
        # 如果所有方法都失败，至少提取一些基本信息
        # 查找看起来像问题的文本
        lines = response_text.split('\n')
        potential_question = ""
        potential_answer = ""

        for line in lines:
            line = line.strip()
            if line.endswith('？') or line.endswith('?'):
                potential_question = line
            elif len(line) > 50 and ('是的' in line or '不是' in line or '可以' in line or '根据' in line):
                potential_answer = line[:200]  # 限制长度
                break

        if potential_question:
            qa_data = {
                'id': str(uuid.uuid4()),
                'question': potential_question,
                'answer': potential_answer or "无法解析完整答案",
                'references': [],
                'original_question': original_question,
                'question_type': question_type,
                'retrieved_docs': retrieved_docs,
                'generated_time': datetime.now().isoformat(),
                'generation_method': 'retrieval_augmented_text_extract'
            }

            print("✅ 备用解析方法3（文本提取）成功")
            return qa_data

    except Exception as e:
        print(f"备用解析方法3失败: {e}")

    print("❌ 所有备用解析方法都失败")
    return {}


def validate_qa_pair(qa_data: Dict[str, Any]) -> bool:
    """
    验证QA对数据的完整性

    Args:
        qa_data: QA对数据

    Returns:
        是否有效
    """
    required_fields = ['question', 'answer']
    for field in required_fields:
        if not qa_data.get(field):
            return False

    # 检查问题和答案的长度
    if len(qa_data['question'].strip()) < 5:
        return False

    if len(qa_data['answer'].strip()) < 10:
        return False

    return True


def format_progress_message(current: int, total: int, prefix: str = "进度") -> str:
    """
    格式化进度消息

    Args:
        current: 当前进度
        total: 总数
        prefix: 前缀文本

    Returns:
        格式化后的进度消息
    """
    percentage = (current / total * 100) if total > 0 else 0
    return f"{prefix}: {current}/{total} ({percentage:.1f}%)"


def load_question_types_mapping(output_dir: Path) -> Dict[str, Any]:
    """加载问题类型映射"""
    mapping_file = output_dir / "question_types.json"

    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"✅ 成功加载问题类型映射: {len(mapping)} 个主分类")
        return mapping
    except Exception as e:
        print(f"⚠️  加载问题类型映射失败: {e}")
        print("   将使用原始问题类型")
        return {}


def get_question_type_description(question_type: str, question_types_map: Dict[str, Any]) -> str:
    """获取问题类型的详细描述"""
    if not question_types_map:
        return question_type

    # 在映射中查找匹配的类型
    for main_category, main_info in question_types_map.items():
        if main_category == question_type:
            return main_info.get('description', question_type)

        # 检查子分类
        subcategories = main_info.get('subcategories', {})
        for sub_category, sub_info in subcategories.items():
            if sub_category == question_type:
                return sub_info.get('description', question_type)

            # 检查代码匹配
            if sub_info.get('code') == question_type:
                return sub_info.get('description', question_type)

    # 如果没有找到匹配，返回原始类型
    return question_type
