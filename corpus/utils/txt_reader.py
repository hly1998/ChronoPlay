"""文本文件读取模块"""

import os
from typing import List, Dict
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from .html_processor import HTMLProcessor


class TxtReader(BaseReader):
    """
    自定义的TXT文件读取器类，继承自BaseReader
    用于读取TXT格式的文件并转换为Document对象
    现在包含HTML处理功能
    """

    def __init__(self, clean_html: bool = True):
        """
        初始化读取器

        Args:
            clean_html: 是否清理HTML内容
        """
        self.clean_html = clean_html
        self.html_processor = HTMLProcessor() if clean_html else None

    def load_data(self, file_path: str, extra_info: Dict = None, encoding='utf-8') -> List[Document]:
        """
        读取TXT文件数据

        Args:
            file_path: 文件路径
            extra_info: 额外的元数据信息
            encoding: 文件编码方式

        Returns:
            Document对象列表
        """
        documents = []

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # 清理HTML内容
            if self.clean_html and self.html_processor:
                original_length = len(content)
                content = self.html_processor.clean_html(content)
                cleaned_length = len(content)

                # 记录清理统计
                # 简化HTML清理日志
                if original_length > 0 and (original_length - cleaned_length) > original_length * 0.1:
                    print(f"清理HTML: {os.path.basename(file_path)}")

            # 从文件名中提取标题
            title = os.path.basename(file_path).replace('.txt', '').replace('.md', '')

            # 设置元数据
            metadata = extra_info.copy() if extra_info else {}
            metadata.update({
                'file_path': file_path,
                'title': title,
                'file_type': 'txt' if file_path.endswith('.txt') else 'md',
                'file_size': len(content),
                'html_cleaned': self.clean_html
            })

            # 创建文档对象
            document = Document(text=content, metadata=metadata)
            document.id_ = f"{title}"
            documents.append(document)

        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

        return documents
