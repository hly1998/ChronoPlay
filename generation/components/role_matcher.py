"""角色匹配器 - 负责为问题模板匹配合适的玩家角色"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class RoleMatcher:
    """角色匹配器类"""

    def __init__(self, role_data_file: str,
                 openai_client: Optional[OpenAI] = None,
                 use_semantic_matching: bool = True,
                 role_index_dir: Optional[str] = None):
        """
        初始化角色匹配器

        Args:
            role_data_file: 角色数据文件路径
            openai_client: OpenAI客户端，用于语义匹配
            use_semantic_matching: 是否使用语义匹配
            role_index_dir: 角色索引存储目录，如果不指定则使用默认位置
        """
        self.role_data_file = Path(role_data_file)
        self.roles_data = []
        self.openai_client = openai_client
        self.use_semantic_matching = use_semantic_matching

        # 设置角色索引存储目录
        if role_index_dir:
            self.role_index_dir = Path(role_index_dir)
        else:
            # 默认存储位置：角色数据文件同目录下的role_index子目录
            self.role_index_dir = self.role_data_file.parent / "role_index"

        # 确保索引目录存在
        self.role_index_dir.mkdir(parents=True, exist_ok=True)

        # 语义匹配相关属性
        self.role_embeddings = []
        self.template_cache = {}  # 缓存模板的嵌入向量
        self.role_embeddings_file = self.role_index_dir / "role_embeddings.npy"

        self._load_roles_data()
        self._build_topic_index()

        # 如果启用语义匹配，预计算角色嵌入
        if self.use_semantic_matching and self.openai_client:
            self._precompute_role_embeddings()

    def _load_roles_data(self):
        """加载角色数据"""
        if not self.role_data_file.exists():
            raise FileNotFoundError(f"角色数据文件不存在: {self.role_data_file}")

        try:
            with open(self.role_data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        role_data = json.loads(line)
                        self.roles_data.append(role_data)

            print(f"✅ 成功加载 {len(self.roles_data)} 个角色数据")
        except Exception as e:
            print(f"❌ 加载角色数据失败: {e}")
            raise

    def _build_topic_index(self):
        """构建话题索引，用于快速检索"""
        self.topic_to_roles = {}

        for role in self.roles_data:
            source_topic = role.get('source_topic', '')
            if source_topic:
                # 处理多个话题（用|分隔）
                topics = [topic.strip() for topic in source_topic.split('|')]
                for topic in topics:
                    if topic not in self.topic_to_roles:
                        self.topic_to_roles[topic] = []
                    self.topic_to_roles[topic].append(role)

    def find_matching_role(self, template_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        为给定的问题模板找到最匹配的角色

        Args:
            template_data: 问题模板数据

        Returns:
            匹配的角色数据，如果没有找到则返回None
        """
        if not self.roles_data:
            return None

        # 只使用语义匹配，失败时返回None
        if self.use_semantic_matching and len(self.role_embeddings) > 0:
            return self._semantic_matching(template_data)
        else:
            # 如果语义匹配不可用，直接返回None
            return None

    def _semantic_matching(self, template_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用语义匹配找到最佳角色

        Args:
            template_data: 问题模板数据

        Returns:
            最匹配的角色数据
        """

        # 构建模板的文本表示
        template_text = self._build_template_text_representation(template_data)
        template_id = template_data.get('id', str(hash(template_text)))

        # 检查缓存
        if template_id in self.template_cache:
            template_embedding = self.template_cache[template_id]
        else:
            # 计算模板的嵌入向量
            template_embedding = self._get_embedding(template_text)
            self.template_cache[template_id] = template_embedding

        # 计算与所有角色的相似度
        similarities = cosine_similarity(
            template_embedding.reshape(1, -1),
            self.role_embeddings
        ).flatten()

        # 找到最相似的角色
        best_role_idx = np.argmax(similarities)
        best_similarity = similarities[best_role_idx]

        # 设置相似度阈值，避免匹配度过低的角色
        if best_similarity < 0.6:  # 可调整的阈值
            print(f"🎮 最佳匹配相似度过低 ({best_similarity:.3f})，不使用角色信息")
            return None

        best_role = self.roles_data[best_role_idx]
        print(f"   🎯 语义匹配成功，相似度: {best_similarity:.3f}")

        # 添加匹配信息到角色数据
        best_role_copy = best_role.copy()
        best_role_copy['semantic_similarity'] = float(best_similarity)
        best_role_copy['matching_method'] = 'semantic'

        return best_role_copy

    def _precompute_role_embeddings(self, batch_size: int = 50):
        """预计算所有角色的语义嵌入向量"""
        print("🔄 预计算角色语义嵌入向量...")

        try:
            # 尝试从持久化存储加载
            if self._load_role_embeddings():
                return

            print("   未找到可用的缓存，开始计算嵌入向量...")

            # 批量处理以提高效率
            role_texts = []
            for role in self.roles_data:
                role_text = self._build_role_text_representation(role)
                role_texts.append(role_text)

            # 分批计算嵌入向量
            embeddings = []
            for i in range(0, len(role_texts), batch_size):
                batch_texts = role_texts[i:i + batch_size]
                batch_embeddings = self._get_embeddings_batch(batch_texts)
                embeddings.extend(batch_embeddings)

                # 显示进度
                processed = min(i + batch_size, len(role_texts))
                print(f"   处理进度: {processed}/{len(role_texts)}")

            # 转换为numpy数组
            self.role_embeddings = np.array(embeddings)

            # 保存到持久化存储
            self._save_role_embeddings()

            print(f"✅ 完成 {len(self.role_embeddings)} 个角色的嵌入向量计算")

        except Exception as e:
            print(f"❌ 预计算角色嵌入失败: {e}")
            self.use_semantic_matching = False
            self.role_embeddings = []

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量获取嵌入向量"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [np.array(item.embedding) for item in response.data]
        except Exception as e:
            print(f"批量获取嵌入向量失败: {e}")
            # 逐个处理作为fallback
            return [self._get_embedding(text) for text in texts]

    def _load_role_embeddings(self) -> bool:
        """
        从持久化存储加载角色嵌入向量

        Returns:
            是否成功加载
        """
        try:
            # 简单检查文件是否存在
            if not self.role_embeddings_file.exists():
                return False

            # 直接加载嵌入向量
            self.role_embeddings = np.load(self.role_embeddings_file)

            print(f"✅ 从缓存加载了 {len(self.role_embeddings)} 个角色的嵌入向量")
            return True

        except Exception as e:
            print(f"   ⚠️  缓存加载失败: {e}")
            return False

    def _save_role_embeddings(self):
        """保存角色嵌入向量到持久化存储"""
        try:
            # 简单保存嵌入向量
            np.save(self.role_embeddings_file, self.role_embeddings)
            print(f"💾 嵌入向量已保存: {self.role_embeddings_file}")

        except Exception as e:
            print(f"⚠️  保存失败: {e}")

    def clear_cache(self):
        """清理角色嵌入缓存"""
        try:
            if self.role_embeddings_file.exists():
                self.role_embeddings_file.unlink()
                print(f"✅ 删除缓存文件: {self.role_embeddings_file}")

            # 清理内存数据
            self.role_embeddings = []
            self.template_cache = {}
            print("🔄 缓存已清理")

        except Exception as e:
            print(f"⚠️  清理失败: {e}")

    def rebuild_cache(self):
        """重新构建角色嵌入缓存"""
        print("🔄 开始重新构建角色嵌入缓存...")

        # 清理现有缓存
        self.clear_cache()

        # 如果启用语义匹配，重新计算嵌入
        if self.use_semantic_matching and self.openai_client:
            self._precompute_role_embeddings()

    def _build_role_text_representation(self, role_data: Dict[str, Any]) -> str:
        """构建角色的文本表示，用于计算嵌入向量"""
        components = []

        # 添加角色描述
        player_desc = role_data.get('player_description', '')
        if player_desc:
            components.append(f"玩家特征: {player_desc}")

        # 添加来源内容
        source_content = role_data.get('source_content', '')
        if source_content:
            components.append(f"原始问题: {source_content}")

        # 添加话题信息
        source_topic = role_data.get('source_topic', '')
        if source_topic:
            components.append(f"相关话题: {source_topic}")

        return " ".join(components)

    def _build_template_text_representation(self, template_data: Dict[str, Any]) -> str:
        """构建问题模板的文本表示"""
        components = []

        # 添加模板内容
        template = template_data.get('template', '')
        if template:
            components.append(f"问题模板: {template}")

        # 添加问题类型
        question_topic = template_data.get('question_topic', '')
        if question_topic:
            components.append(f"问题类型: {question_topic}")

        # 添加描述
        description = template_data.get('description', '')
        if description:
            components.append(f"描述: {description}")

        # 添加话题信息
        topic = template_data.get('topic', '')
        if topic:
            components.append(f"话题: {topic}")

        return " ".join(components)

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"获取嵌入向量失败: {e}")
            # 返回随机向量作为fallback
            return np.random.rand(1536)  # text-embedding-3-small的维度

    def get_role_context(self, role_data: Dict[str, Any]) -> str:
        """
        获取角色上下文信息，用于生成时的角色扮演

        Args:
            role_data: 角色数据

        Returns:
            角色上下文字符串
        """
        if not role_data:
            return ""

        player_description = role_data.get('player_description', '')

        context = f"""
            玩家角色背景：
            {player_description}

            请以这个玩家的身份和语言风格来提问和思考问题。

            **重要要求**：作为这个角色提问时，必须提供具体的细节信息：
            - 如果涉及硬件问题，要明确说明具体的硬件型号
            - 如果涉及游戏内容，要使用准确的游戏术语和名称
            - 如果涉及数值，要给出具体的数值或范围
            - **版本相关注意事项**：不要在问题中直接提及版本号，而应该用自然的表述（如"最近"、"现在"等），版本信息通过提问时间来体现
            - 避免使用模糊的表述，确保问题有足够的信息来给出准确回答
        """
        return context.strip()

    def get_statistics(self) -> Dict[str, Any]:
        """获取角色数据统计信息"""
        return {
            'total_roles': len(self.roles_data),
            'total_topics': len(self.topic_to_roles),
            'semantic_matching_enabled': self.use_semantic_matching,
            'embeddings_computed': len(self.role_embeddings) > 0,
            'cache_exists': self.role_embeddings_file.exists()
        }

    def get_top_similar_roles(self, template_data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        获取与模板最相似的前K个角色（用于调试和分析）

        Args:
            template_data: 问题模板数据
            top_k: 返回的角色数量

        Returns:
            相似度最高的角色列表
        """
        if not self.use_semantic_matching or len(self.role_embeddings) == 0:
            # 如果语义匹配不可用，返回空列表
            return []

        try:
            # 构建模板的文本表示并获取嵌入向量
            template_text = self._build_template_text_representation(template_data)
            template_embedding = self._get_embedding(template_text)

            # 计算相似度
            similarities = cosine_similarity(
                template_embedding.reshape(1, -1),
                self.role_embeddings
            ).flatten()

            # 获取前K个最相似的角色
            top_indices = np.argsort(similarities)[::-1][:top_k]

            result = []
            for idx in top_indices:
                role = self.roles_data[idx].copy()
                role['semantic_similarity'] = float(similarities[idx])
                role['rank'] = len(result) + 1
                result.append(role)

            return result

        except Exception as e:
            print(f"获取相似角色失败: {e}")
            return []
