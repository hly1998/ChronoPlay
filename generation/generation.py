"""游戏QA生成系统 - 基于检索增强的问答对生成"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from components.index_builder import IndexBuilder
from components.question_sampler import QuestionSampler
from components.qa_generator import QAGenerator
from components.qa_inheritance_manager import QAInheritanceManager
from components.qa_filter import QAFilter
from components.qa_refiner import QARefiner
from components.role_matcher import RoleMatcher


class GameQAGenerationSystem:
    """游戏QA生成系统主类"""

    def __init__(self,
                 game_name: str,
                 segment_id: int,
                 api_key: str = "{your-api-key}",
                 base_url: str = "{your-base-url}",
                 model_name: str = "gpt-4o",
                 batch_size: int = 5,
                 target_sample_size: int = 150,
                 enable_role_playing: bool = True,
                 force_rebuild_index: bool = False,
                 similarity_threshold: float = 0.3,
                 similarity_top_k: int = 3,
                 enable_qa_inheritance: bool = True,
                 enable_qa_filtering: bool = True,
                 enable_qa_refining: bool = True,
                 refine_model_name: str = "gpt-4o"):
        """初始化QA生成系统"""

        # 基本参数
        self.game_name = game_name
        self.segment_id = segment_id
        self.batch_size = batch_size
        self.model_name = model_name
        self.target_sample_size = target_sample_size
        self.similarity_threshold = similarity_threshold
        self.similarity_top_k = similarity_top_k
        self.force_rebuild_index = force_rebuild_index

        # 功能开关
        self.enable_role_playing = enable_role_playing
        self.enable_qa_inheritance = enable_qa_inheritance
        self.enable_qa_filtering = enable_qa_filtering
        self.enable_qa_refining = enable_qa_refining
        self.refine_model_name = refine_model_name

        # 初始化路径和配置
        self._init_paths()
        self._load_configurations()
        self._setup_api(api_key, base_url)

        # 初始化所有组件
        self._init_components(api_key, base_url)

        self._print_init_info()

    def _init_paths(self):
        """初始化路径配置"""
        self.base_path = Path(__file__).parent.parent
        self.corpus_path = self.base_path / "data" / self.game_name / "corpus"
        self.template_file = self.base_path / "data" / "question_templates.jsonl"
        self.role_data_file = self.base_path / "data" / "user_persona.jsonl"

        # QA数据输出路径: data/{游戏名称}/segments (永久数据)
        self.qa_output_dir = self.base_path / "data" / self.game_name / "segments"
        self.qa_output_dir.mkdir(parents=True, exist_ok=True)

        # 临时文件路径: generation/data/{游戏名称} (索引、缓存等)
        self.temp_dir = self.base_path / "generation" / "data" / self.game_name
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 验证必要路径
        assert self.corpus_path.exists(), f"语料库路径不存在: {self.corpus_path}"
        assert self.template_file.exists(), f"问题模板文件不存在: {self.template_file}"

    def _load_configurations(self):
        """加载配置文件"""
        # 加载分段信息
        segments_file = self.base_path / "data" / self.game_name / "question_segments_results.json"
        assert segments_file.exists(), f"分段信息文件不存在: {segments_file}"
        with open(segments_file, 'r', encoding='utf-8') as f:
            self.segment_info = json.load(f)

        # 加载问题映射
        question_topics_file = self.base_path / "global_vars" / "question_topics.json"
        question_types_file = self.base_path / "global_vars" / "question_type.json"
        self.question_topics_map = self._load_question_mapping(question_topics_file, "topics")
        self.question_types_map = self._load_question_mapping(question_types_file, "types")

    def _init_components(self, api_key: str, base_url: str):
        """初始化所有组件"""
        # 基础组件 - 索引等临时文件使用 temp_dir
        self.index_builder = IndexBuilder(self.corpus_path, self.temp_dir, self.similarity_top_k)
        self.question_sampler = QuestionSampler(self.template_file)

        # QA生成器（包含角色匹配器）
        role_matcher = self._init_role_matcher(api_key, base_url) if self.enable_role_playing else None
        self.qa_generator = QAGenerator(
            client=OpenAI(api_key=api_key, base_url=base_url),
            model_name=self.model_name,
            question_topics_map=self.question_topics_map,
            question_types_map=self.question_types_map,
            role_matcher=role_matcher
        )

        # 可选组件
        self.inheritance_manager = self._init_inheritance_manager(api_key, base_url)
        self.qa_filter = self._init_qa_filter(api_key, base_url)
        self.qa_refiner = self._init_qa_refiner(api_key, base_url)

    def _init_role_matcher(self, api_key: str, base_url: str):
        """初始化角色匹配器"""
        if not self.role_data_file.exists():
            return None

        try:
            # 角色索引是临时文件，使用 temp_dir
            role_index_dir = self.temp_dir / "role_index"
            role_matcher = RoleMatcher(
                role_data_file=str(self.role_data_file),
                openai_client=OpenAI(api_key=api_key, base_url=base_url),
                use_semantic_matching=True,
                role_index_dir=str(role_index_dir)
            )
            print("✅ 角色扮演功能已启用")
            return role_matcher
        except Exception:
            print("角色扮演功能不可用")
            self.enable_role_playing = False
            return None

    def _init_inheritance_manager(self, api_key: str, base_url: str):
        """初始化QA继承管理器"""
        if not self.enable_qa_inheritance:
            return None
        return QAInheritanceManager(
            self.base_path,
            self.game_name,
            OpenAI(api_key=api_key, base_url=base_url)
        )

    def _init_qa_filter(self, api_key: str, base_url: str):
        """初始化QA过滤器"""
        if not self.enable_qa_filtering:
            return None
        qa_filter = QAFilter(
            api_key=api_key,
            base_url=base_url,
            model_name=self.model_name,
            question_topics_map=self.question_topics_map,
            question_types_map=self.question_types_map
        )
        print("✅ QA过滤功能已启用")
        return qa_filter

    def _init_qa_refiner(self, api_key: str, base_url: str):
        """初始化QA优化器"""
        if not self.enable_qa_refining:
            return None
        qa_refiner = QARefiner(
            api_key=api_key,
            base_url=base_url,
            model_name=self.refine_model_name,
            base_path=self.base_path
        )
        print("✅ QA优化功能已启用")
        return qa_refiner

    def _filter_documents_by_similarity(
            self, documents: List[Dict[str, Any]], threshold: float = None) -> List[Dict[str, Any]]:
        """根据相似度过滤文档"""
        if threshold is None:
            threshold = self.similarity_threshold

        if not documents:
            return documents

        # 过滤低相似度文档
        filtered_docs = []
        for doc in documents:
            score = doc.get('score', 0.0)
            if score >= threshold:
                filtered_docs.append(doc)

        # 按相似度从高到低排序
        filtered_docs.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        if len(documents) > len(filtered_docs):
            filtered_count = len(documents) - len(filtered_docs)
            print(f"已过滤 {filtered_count} 个低相似度文档")

        return filtered_docs

    def _apply_qa_inheritance(self, segment_id: int, inheritance_plan: Dict[str, Any]) -> int:
        """应用QA继承计划"""
        if not inheritance_plan:
            return 0

        # 设置输出文件 - QA数据输出到 qa_output_dir
        segment_output_dir = self.qa_output_dir / f"segment_{segment_id}"
        segment_output_dir.mkdir(exist_ok=True)

        qa_file = segment_output_dir / "generated_qa_pairs.jsonl"

        # 简化的继承统计
        print(f"📊 QA继承: 保留 {inheritance_plan['inherit_count']}, 抛弃 {inheritance_plan['discarded_count']}")

        # 直接继承的QA对
        inherited_qas = inheritance_plan.get('inherited_qas', [])
        inherited_count = 0

        for qa in inherited_qas:
            qa['segment_id'] = segment_id
            qa['inherited_from'] = segment_id - 1
            qa['inheritance_status'] = 'inherited'
            # 清理继承的QA对
            cleaned_qa = self._clean_qa_pair_for_output(qa)
            self.append_qa_to_segment_file(cleaned_qa, qa_file)
            inherited_count += 1

        if inherited_count > 0:
            print(f"✅ 已继承 {inherited_count} 个QA对")

        return inherited_count

    def _load_question_mapping(self, mapping_file: Path, mapping_type: str) -> Dict[str, Any]:
        """加载问题映射文件（topics或types）"""
        assert mapping_file.exists(), f"问题{mapping_type}映射文件不存在: {mapping_file}"

        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _setup_api(self, api_key: str, base_url: str):
        """设置API配置"""
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_API_BASE'] = base_url

        Settings.llm = LlamaOpenAI(
            api_key=api_key, base_url=base_url, model=self.model_name)
        Settings.embed_model = OpenAIEmbedding(
            api_key=api_key,
            api_base=base_url,
            model="text-embedding-3-small"
        )

    def _print_init_info(self):
        """打印初始化信息"""
        print(f"🎮 游戏: {self.game_name}, 分段: {self.segment_id}")
        print(f"🎯 相似度阈值: {self.similarity_threshold:.2f}")

        if self.enable_qa_inheritance:
            print("🔄 QA继承: 已启用")
        if self.enable_qa_refining:
            print(f"✨ QA优化: 已启用 (模型: {self.refine_model_name})")
        if self.enable_qa_filtering:
            print("🔍 QA过滤: 已启用")

    def get_segments_to_process(self) -> List[Dict[str, Any]]:
        """获取需要处理的分段列表"""
        all_segments = [s for s in self.segment_info['segments'] if s['segment_id'] > 0]
        target_segments = [seg for seg in all_segments if seg['segment_id'] == self.segment_id]

        if not target_segments:
            raise ValueError(f"未找到分段ID: {self.segment_id}")
        return target_segments

    def save_segment_progress(self, processed_templates: List[str],
                              attempt_count: int, max_attempts: int, progress_file: Path,
                              generated_count: int = 0):
        """保存分段进度"""
        progress_data = {
            'processed_template_ids': processed_templates,
            'attempt_count': attempt_count,
            'max_attempts': max_attempts,
            'generated_count': generated_count,
            'last_updated': datetime.now().isoformat()
        }

        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    def load_segment_progress(self, progress_file: Path) -> Dict[str, Any]:
        """加载分段进度"""
        if not progress_file.exists():
            return {'processed_template_ids': [], 'attempt_count': 0, 'generated_count': 0}

        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_qa_pair_for_output(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """清理QA对，移除不需要的字段并简化entities"""
        # 创建新的QA对副本
        cleaned_qa = qa_pair.copy()

        # 简化entities字段：只保留text内容，去除重复
        if 'entities' in cleaned_qa and cleaned_qa['entities']:
            entity_texts = []
            seen_texts = set()

            for entity in cleaned_qa['entities']:
                if isinstance(entity, dict) and 'text' in entity:
                    text = entity['text']
                    if text and text not in seen_texts:
                        entity_texts.append(text)
                        seen_texts.add(text)
                elif isinstance(entity, str):
                    # 处理直接是字符串的情况
                    if entity and entity not in seen_texts:
                        entity_texts.append(entity)
                        seen_texts.add(entity)

            cleaned_qa['entities'] = entity_texts

        # 移除不需要的字段
        unwanted_fields = ['template_id', 'source_template', 'extracted_entities']
        for field in unwanted_fields:
            if field in cleaned_qa:
                del cleaned_qa[field]

        return cleaned_qa

    def append_qa_to_segment_file(self, qa_pair: Dict[str, Any], qa_file: Path):
        """将QA对追加到分段文件"""
        with open(qa_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

    def generate_segment_qa_pairs(self, segment: Dict[str, Any]) -> int:
        """为指定分段生成QA对"""
        segment_id = segment['segment_id']
        print(f"=== 开始为分段 {segment_id} 生成QA对 ===")

        # 1. 处理QA继承
        inherited_count, inherited_qas_by_type, inheritance_plan = self._handle_qa_inheritance(segment_id, segment)

        # 2. 构建索引
        if not self.index_builder.build_segment_index(segment_id, self.force_rebuild_index):
            print(f"❌ 分段 {segment_id} 索引构建失败")
            return 0

        # 3. 计算生成目标
        remaining_target = self._calculate_generation_target(inherited_count, inheritance_plan)
        if remaining_target == 0:
            print("✅ 继承的QA对已满足目标数量，无需额外生成")
            return inherited_count

        # 4. 初始化生成环境
        qa_file, progress_file, generated_count, processed_ids, attempt_count = self._setup_generation_environment(
            segment_id, remaining_target)

        # 5. 执行QA生成
        generated_count = self._execute_qa_generation(
            segment, segment_id, qa_file, progress_file,
            generated_count, processed_ids, attempt_count,
            remaining_target, inherited_qas_by_type
        )

        # 6. 输出结果
        total_qa_count = generated_count + inherited_count
        self._print_generation_results(
            segment_id,
            generated_count,
            inherited_count,
            total_qa_count,
            attempt_count,
            remaining_target)

        return total_qa_count

    def _handle_qa_inheritance(self, segment_id: int, segment: Dict[str, Any]) -> tuple:
        """处理QA继承逻辑"""
        inherited_count = 0
        inherited_qas_by_type = {}
        inheritance_plan = None

        if self.enable_qa_inheritance and self.inheritance_manager and segment_id > 1:
            print("🔍 分析QA继承策略...")
            inheritance_plan = self.inheritance_manager.plan_qa_inheritance(
                segment_id, segment, self.target_sample_size)
            inherited_count = self._apply_qa_inheritance(segment_id, inheritance_plan)

            # 统计继承的QA对按类型分布
            inherited_qas = inheritance_plan.get('inherited_qas', [])
            for qa in inherited_qas:
                qa_type = qa.get('question_topic', 'UNKNOWN')
                inherited_qas_by_type[qa_type] = inherited_qas_by_type.get(qa_type, 0) + 1

            print(f"📊 继承QA分布: {inherited_qas_by_type}")

        return inherited_count, inherited_qas_by_type, inheritance_plan

    def _calculate_generation_target(self, inherited_count: int, inheritance_plan: Dict[str, Any]) -> int:
        """计算还需要生成的数量"""
        if self.enable_qa_inheritance and inheritance_plan:
            remaining_target = inheritance_plan.get('generation_needed', 0)
            generation_by_type = inheritance_plan.get('generation_by_type', {})
            print(f"🎯 目标总数: {self.target_sample_size}, 已继承: {inherited_count}, 还需生成: {remaining_target}")
            print(f"📈 各类型生成需求: {generation_by_type}")
        else:
            remaining_target = max(0, self.target_sample_size - inherited_count)
            print(f"🎯 目标总数: {self.target_sample_size}, 已继承: {inherited_count}, 还需生成: {remaining_target}")

        return remaining_target

    def _setup_generation_environment(self, segment_id: int, remaining_target: int) -> tuple:
        """设置生成环境"""
        # 设置输出文件 - QA数据和进度文件输出到 qa_output_dir
        segment_output_dir = self.qa_output_dir / f"segment_{segment_id}"
        segment_output_dir.mkdir(exist_ok=True)
        qa_file = segment_output_dir / "generated_qa_pairs.jsonl"
        progress_file = segment_output_dir / "qa_generation_progress.json"

        # 加载进度
        progress = self.load_segment_progress(progress_file)
        processed_ids = set(progress.get('processed_template_ids', []))
        start_generated_count = progress.get('generated_count', 0)
        start_attempt_count = progress.get('attempt_count', 0)

        # 初始化输出文件
        if not qa_file.exists():
            qa_file.touch()

        if start_generated_count > 0:
            print(f"📂 从进度恢复: 已生成 {start_generated_count} 个QA对，尝试了 {start_attempt_count} 次")

        print(f"🎯 开始动态采样，目标生成: {remaining_target} 个QA对")

        return qa_file, progress_file, start_generated_count, processed_ids, start_attempt_count

    def _execute_qa_generation(self, segment: Dict[str, Any], segment_id: int,
                               qa_file: Path, progress_file: Path,
                               start_generated_count: int, processed_ids: set,
                               start_attempt_count: int, remaining_target: int,
                               inherited_qas_by_type: Dict[str, int]) -> int:
        """执行QA生成的主循环"""
        generated_count = start_generated_count
        attempt_count = start_attempt_count
        batch_qa_pairs = []
        max_attempts = remaining_target * 6

        while (generated_count - start_generated_count) < remaining_target and attempt_count < max_attempts:
            # 动态采样新的template
            needed = remaining_target - (generated_count - start_generated_count)
            templates = self.question_sampler.sample_templates_by_distribution(
                segment, min(10, needed), False, 0, inherited_qas_by_type)

            if not templates:
                print("没有更多问题模板可用")
                break

            generated_count, batch_qa_pairs = self._process_templates(
                templates, segment_id, remaining_target,
                generated_count, start_generated_count,
                attempt_count, processed_ids, qa_file, batch_qa_pairs
            )

            # 显示阶段性进度
            if attempt_count % 20 == 0:
                print(
                    f"  📊 阶段进度: 尝试 {attempt_count}次，已生成 {generated_count - start_generated_count}/{remaining_target} 个QA对")

        # 保存剩余的QA对
        if batch_qa_pairs:
            for qa in batch_qa_pairs:
                self.append_qa_to_segment_file(qa, qa_file)

        # 最终保存进度
        self.save_segment_progress(list(processed_ids), attempt_count, max_attempts, progress_file, generated_count)

        return generated_count

    def _process_templates(self, templates: List[Dict[str, Any]], segment_id: int,
                           remaining_target: int, generated_count: int, start_generated_count: int,
                           attempt_count: int, processed_ids: set, qa_file: Path,
                           batch_qa_pairs: List[Dict[str, Any]]) -> tuple:
        """处理模板生成QA对"""
        for template_data in templates:
            if (generated_count - start_generated_count) >= remaining_target:
                break

            attempt_count += 1
            template_id = template_data.get('id', f't_{attempt_count}')
            template_data["game_name"] = self.game_name

            # 跳过已处理的模板
            if template_id in processed_ids:
                continue

            template_text = template_data.get('template', '')
            print(
                f"  尝试第 {attempt_count} 次，已生成 {generated_count - start_generated_count}/{remaining_target}: {template_text[:50]}...")

            # 生成和处理QA对
            qa_result = self._generate_single_qa(template_data, segment_id)
            if qa_result:
                # 后处理QA对
                final_qa = self._post_process_qa(qa_result, segment_id)
                if final_qa:
                    cleaned_qa = self._clean_qa_pair_for_output(final_qa)
                    batch_qa_pairs.append(cleaned_qa)
                    generated_count += 1
                    print(f"    ✅ 生成成功 ({generated_count - start_generated_count}/{remaining_target})")

            processed_ids.add(template_id)

            # 批量保存
            if len(batch_qa_pairs) >= self.batch_size:
                for qa in batch_qa_pairs:
                    self.append_qa_to_segment_file(qa, qa_file)
                batch_qa_pairs = []

        return generated_count, batch_qa_pairs

    def _generate_single_qa(self, template_data: Dict[str, Any], segment_id: int) -> Dict[str, Any]:
        """生成单个QA对"""
        # 从模板生成假设的问题和答案
        hypothetical_question, hypothetical_answer = self.qa_generator.generate_hypothetical_qa_from_template(
            template_data)

        if not hypothetical_question or not hypothetical_answer:
            print("假设问题答案生成失败")
            return None

        # 检索相关文档
        retrieved_docs = self.index_builder.retrieve_documents(hypothetical_question + hypothetical_answer)

        # 根据相似度过滤文档
        filtered_docs = self._filter_documents_by_similarity(retrieved_docs)

        if not filtered_docs:
            print("所有文档相似度都低于阈值")
            return None

        # 生成QA对
        qa_pairs = self.qa_generator.generate_qa_pair(template_data, filtered_docs, segment_id=segment_id)

        if qa_pairs and qa_pairs[0] and qa_pairs[0].get('question') and qa_pairs[0].get('answer'):
            qa = qa_pairs[0]
            qa['segment_id'] = segment_id
            qa['game_name'] = self.game_name
            return qa
        else:
            print("QA对生成失败")
            return None

    def _post_process_qa(self, qa: Dict[str, Any], segment_id: int) -> Dict[str, Any]:
        """后处理QA对（优化和过滤）"""
        # 处理实体信息
        entities = qa.get('entities', [])
        if entities and self.inheritance_manager:
            print(f"    🏷️  Found entities from corpus: {len(entities)} items")
            qa['extracted_entities'] = entities

        # QA优化
        final_qa = qa
        if self.enable_qa_refining and self.qa_refiner:
            print("    ✨ 开始QA优化...")
            refined_qa = self.qa_refiner.refine_qa_pair(qa)
            if refined_qa:
                final_qa = refined_qa
                print("    ✅ QA优化完成")
            else:
                print("    ⚠️  QA优化失败，使用原始版本")

        # QA过滤
        if self.enable_qa_filtering and self.qa_filter:
            print("    🔍 开始质量过滤...")
            filtered_qa = self.qa_filter.filter_qa_pair(final_qa)
            if filtered_qa:
                print("    ✅ 通过质量过滤")
                return filtered_qa
            else:
                print("    ❌ 未达到高质量标准，丢弃")
                return None

        return final_qa

    def _print_generation_results(self, segment_id: int, generated_count: int,
                                  inherited_count: int, total_qa_count: int,
                                  attempt_count: int, remaining_target: int):
        """打印生成结果"""
        print(f"✅ 分段 {segment_id} 处理完成")
        print(f"   新生成: {generated_count}, 继承: {inherited_count}, 总计: {total_qa_count}")
        print(f"   尝试次数: {attempt_count}, 成功率: {(generated_count/attempt_count*100):.1f}%" if attempt_count > 0 else "")

        if generated_count < remaining_target:
            print(f"⚠️  警告: 未达到目标数量，缺少 {remaining_target - generated_count} 个QA对")

    def generate_qa_pairs(self) -> int:
        """生成QA对的主流程"""
        print("=== 开始QA对生成 ===")

        # 获取要处理的分段
        segments_to_process = self.get_segments_to_process()
        total_generated = 0

        # 为每个分段生成QA对
        for segment in segments_to_process:
            segment_generated = self.generate_segment_qa_pairs(segment)
            total_generated += segment_generated

        print(f"\n🎉 分段 {self.segment_id} 处理完成！生成了 {total_generated} 个QA对")
        return total_generated

    def process(self) -> str:
        """执行完整的QA生成流程"""
        generated_count = self.generate_qa_pairs()
        print(f"\n✅ QA对生成完成, 共生成 {generated_count} 个QA对")

    def reset_progress(self):
        """重置进度"""
        print("🔄 开始重置分段数据...")
        segments_to_process = self.get_segments_to_process()

        for segment in segments_to_process:
            segment_id = segment['segment_id']
            # QA数据和进度文件在 qa_output_dir
            segment_output_dir = self.qa_output_dir / f"segment_{segment_id}"

            if segment_output_dir.exists():
                for file_pattern in ["*generated_qa_pairs.jsonl", "*qa_generation_progress.json"]:
                    for file_path in segment_output_dir.glob(file_pattern):
                        file_path.unlink()
                # 清理临时索引文件（在 temp_dir）
                self.index_builder.cleanup_index(segment_id)

        print("🔄 分段数据重置完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于检索增强的QA对生成')
    parser.add_argument('--game_name', type=str, default='dyinglight2', help='游戏名称')
    parser.add_argument('--segment_id', type=int, required=True, help='时间分段ID（1-n）')
    parser.add_argument('--batch_size', type=int, default=2, help='批量处理大小')
    parser.add_argument('--reset', action='store_true', help='重置生成进度')
    parser.add_argument(
        '--api_key',
        default='{your-api-key}',
        type=str,
        help='OpenAI API密钥')
    parser.add_argument('--base_url', default='{your-base-url}', type=str, help='API基础URL')
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='使用的模型名称')

    parser.add_argument('--target_sample_size', type=int, default=50, help='每个分段的目标采样数量')
    parser.add_argument('--disable_role_playing', action='store_true', help='禁用角色扮演功能')
    parser.add_argument('--force_rebuild_index', action='store_true', help='强制重建索引')
    parser.add_argument('--similarity_threshold', type=float, default=0.5, help='文档相似度过滤阈值')
    parser.add_argument('--similarity_top_k', type=int, default=1, help='检索时返回的相似文档数量')
    parser.add_argument('--disable_qa_inheritance', action='store_true', help='禁用QA继承功能')
    parser.add_argument('--disable_qa_filtering', action='store_true', help='禁用QA过滤功能')
    parser.add_argument('--disable_qa_refining', action='store_true', help='禁用QA优化功能')

    args = parser.parse_args()

    # 创建生成器
    generator = GameQAGenerationSystem(
        game_name=args.game_name,
        segment_id=args.segment_id,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        batch_size=args.batch_size,
        target_sample_size=args.target_sample_size,
        enable_role_playing=not args.disable_role_playing,
        force_rebuild_index=args.force_rebuild_index,
        similarity_threshold=args.similarity_threshold,
        similarity_top_k=args.similarity_top_k,
        enable_qa_inheritance=not args.disable_qa_inheritance,
        enable_qa_filtering=not args.disable_qa_filtering,
        enable_qa_refining=not args.disable_qa_refining,
        refine_model_name=args.model_name
    )

    # 重置进度（如果指定）
    if args.reset:
        generator.reset_progress()

    # 执行生成
    generator.process()


if __name__ == '__main__':
    main()
