"""问题采样器 - 负责从问题模板中按分布采样"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


class QuestionSampler:
    """问题采样器类"""

    def __init__(self, template_file: Path):
        """
        初始化问题采样器

        Args:
            template_file: 问题模板文件路径
        """
        self.template_file = template_file
        self.templates = self._load_question_templates()

    def _load_question_templates(self) -> List[Dict[str, Any]]:
        """加载问题模板数据"""
        if not self.template_file.exists():
            print(f"❌ 问题模板文件不存在: {self.template_file}")
            print("程序无法继续运行，请检查文件路径是否正确")
            exit(1)

        print(f"正在加载问题模板: {self.template_file}")

        templates = []
        try:
            with open(self.template_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            templates.append(data)
                        except json.JSONDecodeError as e:
                            print(f"第{line_num}行JSON解析失败: {e}")
                            continue

            print(f"✅ 成功加载 {len(templates)} 个问题模板")
            return templates

        except Exception as e:
            print(f"❌ 加载问题模板失败: {e}")
            print("程序无法继续运行")
            exit(1)

    def sample_templates_by_distribution(self,
                                         segment: Dict[str, Any],
                                         target_sample_size: int = 150,
                                         test_mode: bool = False,
                                         test_size: int = 10,
                                         inherited_qas_by_type: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        根据分段的问题类型分布采样问题模板

        Args:
            segment: 分段信息
            target_sample_size: 目标采样总数量（仅指新生成的，不包括继承的）
            test_mode: 是否为测试模式
            test_size: 测试模式下的样本数量
            inherited_qas_by_type: 已继承的QA对按类型分布

        Returns:
            采样的问题模板列表
        """
        segment_id = segment['segment_id']
        type_distribution = segment['type_distribution']

        if inherited_qas_by_type is None:
            inherited_qas_by_type = {}

        print(f"=== 为分段 {segment_id} 采样问题模板（新生成目标: {target_sample_size}） ===")

        if inherited_qas_by_type:
            total_inherited = sum(inherited_qas_by_type.values())
            print(f"已继承QA对: {total_inherited} 个")
            for qtype, count in sorted(inherited_qas_by_type.items(), key=lambda x: x[1], reverse=True):
                print(f"  {qtype}: {count} 个")

        # 计算原始分段的总问题数和各类型比例
        original_total = sum(type_distribution.values())
        type_proportions = {qtype: count / original_total for qtype, count in type_distribution.items()}

        print(f"\n原始分段统计: 总问题数 {original_total}")
        print("问题类型比例:")
        for qtype, proportion in sorted(type_proportions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {qtype}: {proportion:.1%} ({type_distribution[qtype]}个)")

        # 按问题类型分组模板
        templates_by_type = {}
        for template in self.templates:
            qtype = template.get('question_topic', 'UNCATEGORIZED')
            if qtype not in templates_by_type:
                templates_by_type[qtype] = []
            templates_by_type[qtype].append(template)

        print("\n问题模板库统计:")
        for qtype, templates in templates_by_type.items():
            print(f"  {qtype}: {len(templates)} 个模板")

        # 计算每个类型还需要生成的数量
        # 理想情况下每个类型应该有的总数量（基于原始比例）
        total_target_with_inherited = target_sample_size + sum(inherited_qas_by_type.values())
        ideal_counts = {}
        for qtype, proportion in type_proportions.items():
            ideal_counts[qtype] = int(total_target_with_inherited * proportion)

        # 计算每个类型还需要生成的数量
        allocated_counts = {}
        remaining_target = target_sample_size

        print("\n计算各类型需要生成的数量:")
        sorted_types_by_priority = sorted(type_proportions.items(), key=lambda x: x[1], reverse=True)

        for qtype, proportion in sorted_types_by_priority:
            if qtype in templates_by_type and remaining_target > 0:
                ideal_total = ideal_counts.get(qtype, 0)
                already_inherited = inherited_qas_by_type.get(qtype, 0)
                need_to_generate = max(0, ideal_total - already_inherited)

                # 限制不能超过剩余目标
                need_to_generate = min(need_to_generate, remaining_target)

                if need_to_generate > 0:
                    allocated_counts[qtype] = need_to_generate
                    remaining_target -= need_to_generate

                print(f"  {qtype}: 理想总数{ideal_total}, 已继承{already_inherited}, 需生成{need_to_generate}")

        # 如果还有剩余目标，按比例分配给最大的几个类型
        if remaining_target > 0:
            sorted_types = sorted(type_proportions.items(), key=lambda x: x[1], reverse=True)
            while remaining_target > 0 and any(qtype in templates_by_type for qtype, _ in sorted_types):
                for qtype, _ in sorted_types:
                    if remaining_target <= 0:
                        break
                    if qtype in templates_by_type:
                        if qtype not in allocated_counts:
                            allocated_counts[qtype] = 0
                        allocated_counts[qtype] += 1
                        remaining_target -= 1

        print(f"\n目标分配方案（需生成 {sum(allocated_counts.values())} 个模板）:")
        for qtype in sorted(allocated_counts.keys()):
            original_count = type_distribution.get(qtype, 0)
            inherited_count = inherited_qas_by_type.get(qtype, 0)
            allocated = allocated_counts[qtype]
            available = len(templates_by_type.get(qtype, []))
            total_final = inherited_count + allocated
            print(f"  {qtype}: 需生成{allocated}个 (继承{inherited_count} + 生成{allocated} = 总计{total_final}, 原始{original_count}, 可用模板{available})")

        # 按分段分布采样
        sampled_templates = []
        actual_sampled = {}

        for qtype, target_count in allocated_counts.items():
            available_templates = templates_by_type.get(qtype, [])

            if not available_templates:
                print(f"⚠️  类型 {qtype} 没有可用模板，跳过")
                continue

            # 确定实际采样数量
            sample_count = min(target_count, len(available_templates))

            # 随机采样
            if sample_count >= len(available_templates):
                sampled = available_templates.copy()
            else:
                sampled = random.sample(available_templates, sample_count)

            sampled_templates.extend(sampled)
            actual_sampled[qtype] = len(sampled)

        # 测试模式下的额外限制
        if test_mode:
            if len(sampled_templates) > test_size:
                sampled_templates = random.sample(sampled_templates, test_size)
                print(f"🧪 测试模式限制: 最终采样 {len(sampled_templates)} 个模板")

        print("\n实际采样结果:")
        total_sampled = len(sampled_templates)
        for qtype in sorted(actual_sampled.keys()):
            count = actual_sampled[qtype]
            percentage = count / total_sampled * 100 if total_sampled > 0 else 0
            print(f"  {qtype}: {count} 个 ({percentage:.1f}%)")

        print(f"✅ 分段 {segment_id} 问题模板采样完成: {total_sampled} 个模板")

        return sampled_templates

    def get_templates_by_type(self, question_type: str) -> List[Dict[str, Any]]:
        """获取指定类型的所有模板"""
        return [t for t in self.templates if t.get('question_topic') == question_type]

    def get_random_template(self, question_type: str = None) -> Dict[str, Any]:
        """获取随机模板"""
        if question_type:
            available_templates = self.get_templates_by_type(question_type)
        else:
            available_templates = self.templates

        if not available_templates:
            raise ValueError(f"没有找到类型为 {question_type} 的模板")

        return random.choice(available_templates)
