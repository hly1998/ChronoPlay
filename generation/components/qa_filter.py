# QA数据质量过滤组件
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加父目录到路径以便导入utils
sys.path.append(str(Path(__file__).parent.parent))

# 这些模块导入必须在path设置之后，因为它们依赖于项目内部模块
from openai import OpenAI  # noqa: E402
from utils.utils import (  # noqa: E402
    clean_response_text,
    get_question_type_description
)
from prompt import data_filter_system, data_filter_user  # noqa: E402
# 从新的question_type.json文件加载任务描述


class QAFilter:
    """QA数据质量过滤器组件"""

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str = "gpt-4o",
                 question_topics_map: Optional[Dict[str, Any]] = None,
                 question_types_map: Optional[Dict[str, Any]] = None):
        """
        初始化QA过滤器

        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model_name: 使用的模型名称
            question_topics_map: 问题主题映射
            question_types_map: 问题类型映射
        """
        self.model_name = model_name

        # 加载问题主题映射和问题类型映射
        base_path = Path(__file__).parent.parent.parent

        if question_topics_map:
            self.question_topics_map = question_topics_map
        else:
            question_topics_file = base_path / "global_vars" / "question_topics.json"
            with open(question_topics_file, 'r', encoding='utf-8') as f:
                self.question_topics_map = json.load(f)

        if question_types_map:
            self.question_types_map = question_types_map
        else:
            question_types_file = base_path / "global_vars" / "question_type.json"
            with open(question_types_file, 'r', encoding='utf-8') as f:
                self.question_types_map = json.load(f)

        # task_description就是question_types_map
        self.task_description = self.question_types_map

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        print("🔧 QA过滤器组件初始化完成")
        print(f"   模型: {self.model_name}")
        print("   使用检索片段进行质量评估")

    def get_related_documents(self, qa_item: Dict[str, Any]) -> str:
        """获取与QA相关的检索文档片段"""
        retrieved_docs = qa_item.get('retrieved_docs', [])
        doc_content = ""

        for i, doc in enumerate(retrieved_docs, 1):
            # 直接使用检索片段内容
            content = doc.get('content', '')
            doc_content += f"文档{i}:\n{content}\n\n"

        return doc_content.strip()

    def evaluate_qa_quality(self, qa_item: Dict[str, Any]) -> int:
        """使用大模型评估单条QA数据的质量，返回质量评分"""
        try:
            # 获取相关文档
            doc_content = self.get_related_documents(qa_item)

            # 获取问题主题描述
            question_topic = qa_item.get('question_topic', '')
            question_topic_desc = get_question_type_description(question_topic, self.question_topics_map)

            # 获取任务描述
            task_type = qa_item.get('task_type', '')
            task_desc = self.task_description.get(task_type, '')

            # 构建评估用的数据格式
            gen_data = [{
                "question": qa_item.get('question', ''),
                "answer": qa_item.get('answer', ''),
                "time": qa_item.get('time', ''),
                "relevant_passage": qa_item.get('references', [])
            }]

            # 如果answer是字符串，转换为列表
            if isinstance(gen_data[0]["answer"], str):
                gen_data[0]["answer"] = [gen_data[0]["answer"]]

            # 构建用户提示
            user_prompt = data_filter_user.format(
                doc_str=doc_content,
                topic_name=question_topic_desc,
                task_name=task_type,
                task_require=task_desc,
                gen_datas=json.dumps(gen_data, ensure_ascii=False, indent=2)
            )

            print("  🔍 调用大模型进行QA质量评估...")
            print(f"    问题主题: {question_topic} -> {question_topic_desc}")
            print(f"    任务类型: {task_type}")
            print(f"    文档数量: {len(qa_item.get('retrieved_docs', []))}")

            # 调用大模型进行评估
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": data_filter_system},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,  # 减少token数量，因为不需要修正结果
                temperature=0.1
            )

            response_text = response.choices[0].message.content.strip()
            print(f"  📝 原始响应长度: {len(response_text)}")

            cleaned_text = clean_response_text(response_text)
            print(f"  🧹 清理后响应长度: {len(cleaned_text)}")

            if not cleaned_text:
                print("  ⚠️  清理后的响应为空，使用原始响应")
                cleaned_text = response_text

            # 解析评估结果，只关注evaluation字段
            print(f"  🔍 准备解析的文本: {cleaned_text[:200]}...")
            eval_result = json.loads(cleaned_text)
            print(f"  📊 解析结果类型: {type(eval_result)}")

            # 确保eval_result是字典类型
            if isinstance(eval_result, dict):
                quality_score = eval_result.get('evaluation', 0)
                print(f"  ✅ JSON解析成功，质量评分: {quality_score}")
            elif isinstance(eval_result, str):
                print(f"  ⚠️  解析结果是字符串: {eval_result[:100]}...")
                quality_score = 0
            else:
                print(f"  ⚠️  解析结果类型未知: {type(eval_result)} - {eval_result}")
                quality_score = 0

            return quality_score

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON解析失败: {e}")
            print(f"  📝 要解析的文本: {cleaned_text[:500] if 'cleaned_text' in locals() else '无'}")
            return 0
        except Exception as e:
            print(f"  ❌ 评估失败: {e}")
            return 0

    def filter_qa_pair(self, qa_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        过滤单个QA对，只保留高质量数据

        Returns:
            如果QA对质量为高质量(评分=2)，返回原始QA对；否则返回None
        """
        # 评估数据质量
        quality_score = self.evaluate_qa_quality(qa_item)

        print(f"    质量评分: {quality_score}")

        # 降低质量阈值：保留评分为1和2的数据，只丢弃评分为0的数据
        if quality_score >= 1:
            print("    ✅ 质量合格数据，保留")
            return qa_item
        else:
            # 只丢弃最低质量数据(评分=0)
            print(f"    ❌ 低质量数据(评分={quality_score})，丢弃")
            return None

    def filter_qa_batch(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量过滤QA对，保留质量合格的数据

        Args:
            qa_pairs: QA对列表

        Returns:
            过滤后的QA对列表（包含质量评分>=1的数据）
        """
        filtered_qa_pairs = []
        stats = {'total': 0, 'qualified': 0, 'discarded': 0}

        for qa_item in qa_pairs:
            stats['total'] += 1
            print(f"  过滤第 {stats['total']}/{len(qa_pairs)} 个QA对...")
            print(f"    问题: {qa_item.get('question', '')[:50]}...")

            filtered_qa = self.filter_qa_pair(qa_item)

            if filtered_qa:
                filtered_qa_pairs.append(filtered_qa)
                stats['qualified'] += 1
            else:
                stats['discarded'] += 1

        print(f"  📊 批量过滤统计: 总数({stats['total']}) 合格({stats['qualified']}) 丢弃({stats['discarded']})")

        return filtered_qa_pairs
