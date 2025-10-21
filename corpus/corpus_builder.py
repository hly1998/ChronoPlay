# 根据时间片段去划分的语料库

import os
import json
from typing import List, Dict, Tuple, Optional, Any
import argparse
import re
from datetime import datetime
from tqdm import tqdm

# 导入llama_index相关组件
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import (
    TokenTextSplitter,
)

# 导入自定义工具模块
from utils.txt_reader import TxtReader
from utils.ner_extractor import GameNERExtractor


class TimeSegmentCorpusBuilder:
    """基于时间片段的语料库构建器"""

    def __init__(self, segments_file: str, data_directory: str, output_dir: str,
                 chunk_size: int = 1000, chunk_overlap: int = 200, clean_html: bool = True,
                 enable_ner: bool = True, enable_time_extraction: bool = True,
                 openai_api_key: Optional[str] = None, openai_base_url: Optional[str] = None,
                 ner_model: str = "gpt-4o", use_self_icl: bool = True, num_pseudo_examples: int = 1):
        """
        初始化构建器

        Args:
            segments_file: 时间片段配置文件路径
            data_directory: 数据目录路径
            output_dir: 输出目录路径
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            clean_html: 是否清理HTML内容
            enable_ner: 是否启用NER功能
            enable_time_extraction: 是否启用时间提取功能
            openai_api_key: OpenAI API密钥
            openai_base_url: OpenAI API基础URL
            ner_model: NER使用的模型
            use_self_icl: 是否使用SELF-ICL技术
            num_pseudo_examples: SELF-ICL生成的伪示例数量
        """
        self.segments_file = segments_file
        self.data_directory = data_directory
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.clean_html = clean_html
        self.enable_ner = enable_ner
        self.enable_time_extraction = enable_time_extraction
        self.use_self_icl = use_self_icl
        self.num_pseudo_examples = num_pseudo_examples

        # 加载时间片段配置
        self.segments = self._load_segments()

        # 创建文本分割器
        self.splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 创建TXT读取器
        self.txt_reader = TxtReader(clean_html=clean_html)

        # 创建NER提取器
        if self.enable_ner:
            self.ner_extractor = GameNERExtractor(
                api_key=openai_api_key,
                base_url=openai_base_url,
                model=ner_model,
                use_self_icl=use_self_icl,
                num_pseudo_examples=num_pseudo_examples,
                max_workers=3  # 使用3个并发线程
            )
        else:
            self.ner_extractor = None

    def _load_segments(self) -> List[Dict]:
        """加载时间片段配置"""
        with open(self.segments_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('segments', [])

    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        """
        从文件名中提取日期

        Args:
            filename: 文件名

        Returns:
            日期字符串 (YYYY-MM-DD) 或 None
        """
        # 匹配 YYYY-MM-DD 格式的日期
        date_pattern = r'^(\d{4}-\d{2}-\d{2})'
        match = re.match(date_pattern, filename)
        if match:
            return match.group(1)
        return None

    def _extract_time_from_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        从文件名中提取时间信息（用于元数据）

        Args:
            filename: 文件名

        Returns:
            时间信息字典或None
        """
        # 只匹配文件名开头的 YYYY-MM-DD 格式日期
        filename_date_pattern = r'^(\d{4}-\d{2}-\d{2})'
        match = re.search(filename_date_pattern, filename)
        if match:
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return {
                'extracted_date': date_str,
                'extracted_year': date_obj.year,
                'extracted_month': date_obj.month,
                'extracted_day': date_obj.day,
                'extraction_source': 'filename',
                'date_format': 'YYYY-MM-DD'
            }

        return None

    def _add_time_metadata_to_document(self, document: Document) -> Document:
        """
        为文档添加时间元数据

        Args:
            document: 原始文档

        Returns:
            添加了时间元数据的文档
        """
        metadata = document.metadata.copy()

        # 从文件名提取时间信息
        title = metadata.get('title', '')
        file_path = metadata.get('file_path', '')

        time_info = None

        # 优先从title提取
        if title:
            time_info = self._extract_time_from_filename(title)

        # 如果title没有时间信息，从文件路径的文件名提取
        if not time_info and file_path:
            filename = os.path.basename(file_path)
            time_info = self._extract_time_from_filename(filename)

        # 更新元数据
        if time_info:
            metadata.update(time_info)
        else:
            # 如果没有时间信息，添加空的时间字段
            metadata.update({
                'extracted_date': None,
                'extracted_year': None,
                'extracted_month': None,
                'extracted_day': None,
                'extraction_source': None,
                'date_format': None
            })

        # 创建新的文档对象，保持原有内容但更新元数据
        updated_document = Document(
            text=document.get_content(),
            metadata=metadata,
            id_=document.id_
        )

        return updated_document

    def _get_segment_for_date(self, date_str: str) -> Optional[int]:
        """
        根据日期获取对应的时间片段ID

        Args:
            date_str: 日期字符串 (YYYY-MM-DD)

        Returns:
            片段ID 或 None
        """
        if not date_str:
            return None

        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        for segment in self.segments:
            start_date = datetime.strptime(segment['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(segment['end_date'], '%Y-%m-%d')

            # 检查日期是否在片段范围内，且不晚于片段结束日期
            if start_date <= date_obj < end_date:
                return segment['segment_id']

        return None

    def _flatten_folder(self, root: str) -> List[str]:
        """
        递归展平文件夹，获取所有文件路径

        Args:
            root: 根目录路径

        Returns:
            包含所有文件路径的列表
        """
        files = []
        for item in os.listdir(root):
            path = os.path.join(root, item)
            if os.path.isdir(path):
                files.extend(self._flatten_folder(path))
            elif path.endswith(('.txt', '.md')):  # 只处理txt和markdown文件
                files.append(path)
        return files

    def _categorize_files_by_segment(self) -> Dict[int, List[str]]:
        """
        按时间片段对文件进行分类

        Returns:
            字典，键为片段ID，值为文件路径列表。特殊键-1表示无时间的文件
        """
        # 获取所有文件路径
        all_files = self._flatten_folder(self.data_directory)

        # 按时间片段分类
        categorized_files = {-1: []}  # -1 表示无时间的文件

        for segment in self.segments:
            categorized_files[segment['segment_id']] = []

        # 分类文件
        for file_path in all_files:
            filename = os.path.basename(file_path)
            file_date = self._extract_date_from_filename(filename)

            if file_date:
                segment_id = self._get_segment_for_date(file_date)
                if segment_id is not None:
                    categorized_files[segment_id].append(file_path)
                else:
                    categorized_files[-1].append(file_path)  # 放入无时间文件组
            else:
                # 无时间的文件
                categorized_files[-1].append(file_path)

        return categorized_files

    def _process_segment_files(
            self, files: List[str], segment_id: int) -> Tuple[List[Document], List[BaseNode], Optional[Dict]]:
        """
        处理单个时间片段的文件

        Args:
            files: 文件路径列表
            segment_id: 片段ID

        Returns:
            (文档列表, 节点列表, NER统计信息)
        """

        # 加载所有文档
        all_documents = []
        for file_path in tqdm(files, desc="加载文档", leave=False):
            # 为每个文档添加片段信息到元数据
            extra_info = {'segment_id': segment_id}
            documents = self.txt_reader.load_data(file_path, extra_info=extra_info)

            # 为每个文档添加时间元数据（如果启用）
            if self.enable_time_extraction:
                documents_with_time = []
                for doc in documents:
                    doc_with_time = self._add_time_metadata_to_document(doc)
                    documents_with_time.append(doc_with_time)
                all_documents.extend(documents_with_time)
            else:
                all_documents.extend(documents)

        # 过滤空文档
        valid_documents = []
        for doc in all_documents:
            if doc.get_content().strip() != '':
                valid_documents.append(doc)

        # 分割文档为节点
        if valid_documents:
            with tqdm(desc="分割文档", leave=False) as pbar:
                nodes = self.splitter.get_nodes_from_documents(valid_documents, show_progress=False)
                pbar.update(1)
        else:
            nodes = []

        # 执行NER提取
        ner_stats = None
        if self.enable_ner and self.ner_extractor and nodes:
            with tqdm(desc="NER实体提取", leave=False) as pbar:
                # 提取所有节点的文本内容
                node_texts = [node.get_content() for node in nodes]

                # 批量进行NER提取
                all_entities = self.ner_extractor.batch_extract_entities(
                    node_texts, show_progress=True
                )

                # 将实体信息添加到节点元数据中
                for node, entities in zip(nodes, all_entities):
                    node.metadata['entities'] = entities

                # 生成NER统计信息
                ner_stats = self.ner_extractor.get_entity_statistics(all_entities)
                pbar.update(1)

        return valid_documents, nodes, ner_stats

    def _save_segment_corpus(self, documents: List[Document], nodes: List[BaseNode],
                             segment_id: int, segment_info: Optional[Dict] = None, ner_stats: Optional[Dict] = None):
        """
        保存时间片段的语料库数据

        Args:
            documents: 原始文档列表
            nodes: 处理后的节点列表
            segment_id: 片段ID
            segment_info: 片段信息（未使用，保持接口兼容性）
            ner_stats: NER统计信息（未使用，保持接口兼容性）
        """
        # 创建输出目录
        if segment_id == -1:
            output_subdir = os.path.join(self.output_dir, 'segment_timeless')
        else:
            output_subdir = os.path.join(self.output_dir, f'segment_{segment_id}')

        os.makedirs(output_subdir, exist_ok=True)

        # 只保存简化的JSONL格式
        corpus_file = os.path.join(output_subdir, 'corpus.jsonl')
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for node in tqdm(nodes, desc="保存节点", leave=False):
                # 只保留必要的字段
                corpus_data = {
                    'id': node.id_,
                    'title': node.metadata.get('title', ''),
                    'contents': node.get_content(),
                    'segment_id': node.metadata.get('segment_id'),
                    'entities': node.metadata.get('entities', []),
                    'extracted_date': node.metadata.get('extracted_date')
                }
                f.write(json.dumps(corpus_data, ensure_ascii=False) + '\n')

    def build_time_segmented_corpus(self):
        """构建按时间片段划分的语料库"""
        # 按时间片段分类文件
        categorized_files = self._categorize_files_by_segment()

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 计算需要处理的片段数量
        valid_segments = [(segment_id, files) for segment_id, files in categorized_files.items() if files]

        # 创建总体进度条
        with tqdm(total=len(valid_segments), desc="构建语料库", unit="片段") as pbar:
            for segment_id, files in valid_segments:
                # 更新进度条描述
                segment_name = f"片段{segment_id}" if segment_id != -1 else "无时间文件"
                pbar.set_description(f"处理{segment_name}")

                # 处理文件
                documents, nodes, ner_stats = self._process_segment_files(files, segment_id)

                # 保存语料库
                self._save_segment_corpus(documents, nodes, segment_id)

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"文件": len(files), "节点": len(nodes)})


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='根据时间片段划分数据语料库')
    parser.add_argument('--segments_file', type=str,
                        default='data/dyinglight2/question_segments_results.json',
                        help='时间片段配置文件路径')
    parser.add_argument('--data_dir', type=str,
                        default='data/dyinglight2/knowledge',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str,
                        default='data/dyinglight2/corpus',
                        help='输出目录路径')
    parser.add_argument('--chunk_size', type=int, default=2048,
                        help='文本块大小')
    parser.add_argument('--chunk_overlap', type=int, default=256,
                        help='文本块重叠大小')
    parser.add_argument('--no_clean_html', action='store_true',
                        help='禁用HTML清理功能')
    parser.add_argument('--disable_ner', action='store_true',
                        help='禁用NER实体识别功能')
    parser.add_argument('--disable_time_extraction', action='store_true',
                        help='禁用时间提取功能')
    parser.add_argument('--openai_api_key', type=str, default="{your-api-key}",
                        help='OpenAI API密钥')
    parser.add_argument('--openai_base_url', type=str, default="{your-base-url}",
                        help='OpenAI API基础URL')
    parser.add_argument('--ner_model', type=str, default='gpt-4o',
                        help='NER使用的模型')
    parser.add_argument('--disable_self_icl', action='store_true',
                        help='禁用SELF-ICL技术，使用传统NER方法')
    parser.add_argument('--num_pseudo_examples', type=int, default=1,
                        help='SELF-ICL生成的伪示例数量')

    args = parser.parse_args()

    # 检查必要文件是否存在
    if not os.path.exists(args.segments_file):
        return

    if not os.path.exists(args.data_dir):
        return

    # 创建构建器并执行构建
    builder = TimeSegmentCorpusBuilder(
        segments_file=args.segments_file,
        data_directory=args.data_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        clean_html=not args.no_clean_html,
        enable_ner=not args.disable_ner,
        enable_time_extraction=not args.disable_time_extraction,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        ner_model=args.ner_model,
        use_self_icl=not args.disable_self_icl,
        num_pseudo_examples=args.num_pseudo_examples
    )

    builder.build_time_segmented_corpus()


if __name__ == '__main__':
    main()
