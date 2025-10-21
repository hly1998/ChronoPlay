"""QA对继承管理器组件"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
from openai import OpenAI


class QAInheritanceManager:
    """QA对继承管理器"""

    def __init__(self, base_path: Path, game_name: str, openai_client: OpenAI = None):
        self.base_path = base_path
        self.game_name = game_name
        self.corpus_path = base_path / "data" / game_name / "corpus"
        self.qa_data_path = base_path / "generation" / "data" / game_name
        self.openai_client = openai_client
        # 缓存片段实体，避免重复抽取
        self._segment_entities_cache = {}

    def load_qa_pairs_from_segment(self, segment_id: int) -> List[Dict[str, Any]]:
        """从指定片段加载QA对"""
        qa_pairs = []
        qa_file = self.qa_data_path / f"segment_{segment_id}" / "generated_qa_pairs.jsonl"

        if not qa_file.exists():
            return qa_pairs

        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        qa_pair = json.loads(line.strip())
                        qa_pairs.append(qa_pair)
                    except json.JSONDecodeError:
                        continue

        return qa_pairs

    def extract_and_save_qa_entities(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理QA对的实体信息，直接使用从corpus文件中获取的实体

        Args:
            qa_pair: 问答对数据

        Returns:
            包含实体信息的QA对
        """
        # 如果QA对已经包含从corpus中获取的实体信息，直接使用
        entities_from_corpus = qa_pair.get('entities', [])
        if entities_from_corpus:
            # 提取实体文本用于继承判断
            entity_texts = [entity.get('text', '') for entity in entities_from_corpus if entity.get('text')]
            qa_pair['extracted_entities'] = entity_texts
        else:
            qa_pair['extracted_entities'] = []

        # 添加时间戳用于追踪
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        qa_text = f"问题: {question}\n答案: {answer}"
        qa_pair['entity_extraction_timestamp'] = json.dumps(
            {"timestamp": hashlib.md5(qa_text.encode()).hexdigest()[:8]}
        )

        return qa_pair

    def _extract_new_segment_entities_from_corpus(self, segment_id: int) -> Set[str]:
        """
        从corpus文件的metadata中直接获取片段的所有游戏实体
        """
        all_entities = set()
        corpus_file = self.corpus_path / f"segment_{segment_id}" / "corpus.jsonl"

        if not corpus_file.exists():
            return all_entities

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        # 从corpus文件中直接获取实体信息（字符串列表）
                        entities_list = data.get('entities', [])

                        # 提取实体文本
                        for entity in entities_list:
                            if isinstance(entity, str):
                                entity_text = entity.strip()
                                if entity_text:
                                    all_entities.add(entity_text)

                    except json.JSONDecodeError:
                        continue

        return all_entities

    def get_segment_entities(self, segment_id: int) -> Set[str]:
        """
        获取片段实体（带缓存）

        Args:
            segment_id: 片段ID

        Returns:
            片段中的游戏实体集合
        """
        # 检查缓存
        if segment_id in self._segment_entities_cache:
            return self._segment_entities_cache[segment_id]

        # 从corpus文件中获取实体并缓存
        entities = self._extract_new_segment_entities_from_corpus(segment_id)

        # 缓存结果
        self._segment_entities_cache[segment_id] = entities
        print(f"片段 {segment_id} 实体已缓存: {len(entities)} 个实体")

        return entities

    def clear_segment_entities_cache(self, segment_id: int = None):
        """
        清理片段实体缓存

        Args:
            segment_id: 要清理的片段ID，如果为None则清理所有缓存
        """
        if segment_id is None:
            self._segment_entities_cache.clear()
            print("已清理所有片段实体缓存")
        else:
            if segment_id in self._segment_entities_cache:
                del self._segment_entities_cache[segment_id]
                print(f"已清理片段 {segment_id} 的实体缓存")

    def _check_qa_entity_overlap(self, qa_pair: Dict[str, Any],
                                 current_segment_entities: Set[str]) -> bool:
        """
        检查QA对与当前段落实体是否存在重叠

        Args:
            qa_pair: 问答对
            current_segment_entities: 当前段落中的游戏实体集合

        Returns:
            True表示存在重叠（需要去除），False表示无重叠（可以保留）
        """
        # 获取QA对的实体信息
        qa_entities = set()
        if 'extracted_entities' in qa_pair:
            # 从实体字典列表中提取实体文本
            entity_list = qa_pair['extracted_entities']
            if isinstance(entity_list, list):
                qa_entities = set(entity.get('text', '') for entity in entity_list if isinstance(entity, dict))

        # 如果QA对没有实体信息或当前段落没有实体，则认为无重叠
        if not qa_entities or not current_segment_entities:
            return False

        # 检查是否存在任何重叠
        intersection = qa_entities.intersection(current_segment_entities)
        return len(intersection) > 0

    def plan_qa_inheritance(self, current_segment_id: int,
                            current_segment_info: Dict[str, Any],
                            target_sample_size: int,
                            previous_segment_id: int = None) -> Dict[str, Any]:
        """
        规划QA继承策略
        优化版本：先抽取当前片段实体，过滤与当前段落实体有重叠的QA对，然后根据当前片段的类型分布重新采样

        Args:
            current_segment_id: 当前片段ID
            current_segment_info: 当前片段信息（包含类型分布）
            target_sample_size: 目标总样本大小
            previous_segment_id: 前一个片段ID

        Returns:
            继承计划的详细信息
        """
        if previous_segment_id is None:
            previous_segment_id = current_segment_id - 1

        if previous_segment_id < 1:
            return {
                'inherit_count': 0,
                'discarded_count': 0,
                'inherited_qas': [],
                'previous_segment_total': 0
            }

        # 先获取当前片段的实体（这里会进行缓存，后续不会重复抽取）
        print(f"开始抽取片段 {current_segment_id} 的实体...")
        current_segment_entities = self.get_segment_entities(current_segment_id)

        # 加载前一个片段的QA对
        previous_qas = self.load_qa_pairs_from_segment(previous_segment_id)

        print(f"准备处理 {len(previous_qas)} 个QA对，片段实体数量: {len(current_segment_entities)}")

        # 第一步：过滤掉与当前段落实体存在重叠的QA对
        candidate_qas = []
        discarded_count = 0

        for i, qa in enumerate(previous_qas):
            # 检查QA对实体与当前段落实体是否存在重叠
            has_overlap = self._check_qa_entity_overlap(qa, current_segment_entities)

            if has_overlap:
                # 存在实体重叠，抛弃该QA对
                discarded_count += 1
            else:
                # 无实体重叠，可以作为候选
                candidate_qas.append(qa)

            # 进度显示
            if (i + 1) % 50 == 0:
                print(f"  已处理 {i + 1}/{len(previous_qas)} 个QA对")

        print(f"实体重叠过滤结果: 候选{len(candidate_qas)}个, 抛弃{discarded_count}个")

        # 第二步：根据用户指定的目标数量和当前片段的类型比例计算继承和生成策略
        inheritance_plan = self._calculate_inheritance_and_generation_plan(
            candidate_qas,
            current_segment_info['type_distribution'],
            target_sample_size
        )

        return {
            'inherit_count': inheritance_plan['total_inherit'],
            'discarded_count': len(previous_qas) - inheritance_plan['total_inherit'],
            'inherited_qas': inheritance_plan['inherited_qas'],
            'previous_segment_total': len(previous_qas),
            'candidate_count': len(candidate_qas),
            'generation_needed': inheritance_plan['generation_needed'],
            'inheritance_by_type': inheritance_plan['inheritance_by_type'],
            'generation_by_type': inheritance_plan['generation_by_type']
        }

    def _calculate_inheritance_and_generation_plan(self, candidate_qas: List[Dict[str, Any]],
                                                   original_type_distribution: Dict[str, int],
                                                   target_sample_size: int) -> Dict[str, Any]:
        """
        计算继承和生成计划

        逻辑：
        1. 基于原始类型分布和用户指定的target_sample_size重新计算目标类型分布
        2. 统计前一片段每种问题类型的可用数量（经过实体重叠过滤后）
        3. 对于每种类型：
           - 如果前一片段该类型数量 >= 当前需求，则采样到当前需求数量，无需生成
           - 如果前一片段该类型数量 < 当前需求，则全部继承，还需生成差额

        Args:
            candidate_qas: 候选QA对列表（已经过实体重叠过滤）
            original_type_distribution: 原始片段的类型分布
            target_sample_size: 用户指定的目标总样本数量

        Returns:
            包含继承和生成计划的详细信息
        """
        # 基于原始类型分布和用户指定的target_sample_size重新计算目标分布
        original_total = sum(original_type_distribution.values())
        if original_total == 0:
            print("⚠️  原始类型分布为空")
            return {
                'inherited_qas': [],
                'total_inherit': 0,
                'generation_needed': target_sample_size,
                'inheritance_by_type': {},
                'generation_by_type': {'UNKNOWN': target_sample_size}
            }

        # 计算各类型的比例
        type_proportions = {qtype: count / original_total
                            for qtype, count in original_type_distribution.items()}

        # 基于用户指定的target_sample_size重新计算各类型的目标数量
        target_distribution = {}
        total_allocated = 0

        # 先按比例分配
        for qtype, proportion in type_proportions.items():
            allocated = int(target_sample_size * proportion)
            target_distribution[qtype] = allocated
            total_allocated += allocated

        # 处理舍入误差，将剩余的数量分配给占比最大的类型
        remaining = target_sample_size - total_allocated
        if remaining > 0:
            # 找到占比最大的类型
            max_type = max(type_proportions.keys(), key=lambda x: type_proportions[x])
            target_distribution[max_type] += remaining

        print(f"\n重新计算的目标分布（基于用户指定的{target_sample_size}个样本）:")
        for qtype, count in sorted(target_distribution.items(), key=lambda x: x[1], reverse=True):
            proportion = type_proportions[qtype]
            print(f"  {qtype}: {count} 个 ({proportion:.1%})")

        if not candidate_qas:
            # 如果没有候选QA对，所有都需要生成
            return {
                'inherited_qas': [],
                'total_inherit': 0,
                'generation_needed': sum(target_distribution.values()),
                'inheritance_by_type': {qtype: 0 for qtype in target_distribution.keys()},
                'generation_by_type': dict(target_distribution)
            }

        # 按类型分组候选QA对
        candidate_qas_by_type = {}
        for qa in candidate_qas:
            qa_type = qa.get('question_topic', 'UNKNOWN')
            if qa_type not in candidate_qas_by_type:
                candidate_qas_by_type[qa_type] = []
            candidate_qas_by_type[qa_type].append(qa)

        print("\n=== 继承和生成计划计算 ===")
        print("候选QA对按类型分布（已过滤实体重叠）:")
        for qa_type, qas in candidate_qas_by_type.items():
            print(f"  {qa_type}: {len(qas)} 个")

        print("\n最终目标分布:")
        for qtype, count in sorted(target_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {qtype}: {count} 个")

        # 计算每种类型的继承和生成计划
        inheritance_by_type = {}
        generation_by_type = {}
        selected_qas = []

        print("\n各类型继承和生成计划:")
        import random

        for qtype, target_count in target_distribution.items():
            available_count = len(candidate_qas_by_type.get(qtype, []))

            if target_count == 0:
                # 目标数量为0，无需继承和生成
                inherit_count = 0
                generate_count = 0
            elif available_count >= target_count:
                # 前一片段该类型数量足够，采样到目标数量，无需生成
                inherit_count = target_count
                generate_count = 0

                # 随机采样
                if qtype in candidate_qas_by_type and target_count > 0:
                    available_qas = candidate_qas_by_type[qtype]
                    sampled_qas = random.sample(available_qas, target_count)
                    selected_qas.extend(sampled_qas)

            else:
                # 前一片段该类型数量不足，全部继承，剩余需要生成
                inherit_count = available_count
                generate_count = target_count - available_count

                # 全部继承
                if qtype in candidate_qas_by_type:
                    selected_qas.extend(candidate_qas_by_type[qtype])

            inheritance_by_type[qtype] = inherit_count
            generation_by_type[qtype] = generate_count

            print(f"  {qtype}: 需要{target_count}个 → 继承{inherit_count}个 + 生成{generate_count}个 (可用{available_count}个)")

        total_inherit = sum(inheritance_by_type.values())
        total_generate = sum(generation_by_type.values())

        print("\n总计划:")
        print(f"  继承总数: {total_inherit} 个")
        print(f"  生成总数: {total_generate} 个")
        print(f"  目标总数: {sum(target_distribution.values())} 个")

        return {
            'inherited_qas': selected_qas,
            'total_inherit': total_inherit,
            'generation_needed': total_generate,
            'inheritance_by_type': inheritance_by_type,
            'generation_by_type': generation_by_type
        }

    def _resample_qas_by_distribution(self, candidate_qas: List[Dict[str, Any]],
                                      target_distribution: Dict[str, int],
                                      target_sample_size: int) -> List[Dict[str, Any]]:
        """
        根据目标分布重新采样候选QA对

        Args:
            candidate_qas: 候选QA对列表
            target_distribution: 目标类型分布
            target_sample_size: 目标总样本大小

        Returns:
            重新采样后的QA对列表
        """
        if not candidate_qas:
            return []

        # 按类型分组候选QA对
        qas_by_type = {}
        for qa in candidate_qas:
            qa_type = qa.get('question_topic', 'UNKNOWN')
            if qa_type not in qas_by_type:
                qas_by_type[qa_type] = []
            qas_by_type[qa_type].append(qa)

        print("\n候选QA对按类型分布:")
        for qa_type, qas in qas_by_type.items():
            print(f"  {qa_type}: {len(qas)} 个")

        # 计算目标分布的比例
        total_target = sum(target_distribution.values())
        target_proportions = {qtype: count / total_target
                              for qtype, count in target_distribution.items()}

        print("\n目标分布比例:")
        for qtype, proportion in sorted(target_proportions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {qtype}: {proportion:.1%} ({target_distribution[qtype]}个)")

        allocated_counts = {}
        total_allocated = 0

        print("\n按目标分布严格限制继承数量:")

        # 为每个目标类型计算继承限制
        for qtype in target_distribution.keys():
            target_count = target_distribution[qtype]

            if qtype in qas_by_type:
                available_count = len(qas_by_type[qtype])
                actual_inherit = min(target_count, available_count)

                if actual_inherit > 0:
                    allocated_counts[qtype] = actual_inherit
                    total_allocated += actual_inherit

                print(f"  {qtype}: 目标{target_count}个 → 实际继承{actual_inherit}个 (可用{available_count}个)")
            else:
                print(f"  {qtype}: 目标{target_count}个 → 无候选QA对，继承0个")

        # 如果总继承数量还没有达到合理水平，可以适当增加一些继承
        target_total_inherit = min(int(target_sample_size * 0.6), total_allocated + 20)  # 最多60%，或者当前数量+20
        remaining_budget = max(0, target_total_inherit - total_allocated)

        if remaining_budget > 0:
            print(f"\n还有{remaining_budget}个继承预算，按需求优先级分配:")
            # 计算每个类型还需要多少个来达到目标
            need_more = []
            for qtype in target_distribution.keys():
                if qtype in qas_by_type:
                    current_inherit = allocated_counts.get(qtype, 0)
                    target_count = target_distribution[qtype]
                    available_count = len(qas_by_type[qtype])
                    can_add_more = min(available_count - current_inherit, target_count - current_inherit)

                    if can_add_more > 0:
                        need_more.append((qtype, can_add_more, target_count))

            # 按目标数量从大到小排序，优先给需求大的类型
            need_more.sort(key=lambda x: x[2], reverse=True)

            for qtype, can_add, target_count in need_more:
                if remaining_budget <= 0:
                    break

                additional = min(can_add, remaining_budget)
                if additional > 0:
                    allocated_counts[qtype] = allocated_counts.get(qtype, 0) + additional
                    remaining_budget -= additional
                    print(f"  {qtype}: 额外分配{additional}个")

        print(f"\n最终继承分配（总计{sum(allocated_counts.values())}个）:")
        for qtype in sorted(target_distribution.keys(), key=lambda x: target_distribution[x], reverse=True):
            inherit_count = allocated_counts.get(qtype, 0)
            target_count = target_distribution[qtype]
            available_count = len(qas_by_type.get(qtype, []))
            inherit_ratio = inherit_count / target_count * 100 if target_count > 0 else 0

            print(f"  {qtype}: 继承{inherit_count}个 / 目标{target_count}个 ({inherit_ratio:.1f}%) / 可用{available_count}个")

        # 按分配方案采样QA对
        import random
        selected_qas = []

        for qtype, count in allocated_counts.items():
            available_qas = qas_by_type[qtype]
            if count >= len(available_qas):
                # 如果需要的数量大于等于可用数量，全部选择
                selected_qas.extend(available_qas)
            else:
                # 随机采样指定数量
                sampled = random.sample(available_qas, count)
                selected_qas.extend(sampled)

        return selected_qas
