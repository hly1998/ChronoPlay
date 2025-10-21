"""HTML内容处理模块"""

import re
import html
from bs4 import BeautifulSoup


class HTMLProcessor:
    """HTML内容处理器"""

    def __init__(self):
        self.use_bs4 = True

    def clean_html(self, text: str) -> str:
        """
        清理HTML内容

        Args:
            text: 包含HTML的原始文本

        Returns:
            清理后的纯文本
        """
        if not text:
            return ""

        # 处理markdown风格的链接 [text](url)
        text = self._process_markdown_links(text)

        # 处理HTML内容
        text = self._clean_html_with_bs4(text)

        # 后处理：清理多余空白和格式
        text = self._post_process_text(text)

        return text

    def _process_markdown_links(self, text: str) -> str:
        """处理markdown风格的链接 [text](url)"""
        # 匹配 [链接文本](URL) 格式
        def replace_link(match):
            link_text = match.group(1)
            # url = match.group(2)
            # 只保留链接文本，可选择性地保留URL信息
            return f"{link_text}"

        # 处理链接
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, text)
        return text

    def _clean_html_with_bs4(self, text: str) -> str:
        """使用BeautifulSoup清理HTML"""
        try:
            # 解析HTML
            soup = BeautifulSoup(text, 'html.parser')

            # 移除script和style标签
            for script in soup(["script", "style"]):
                script.decompose()

            # 处理表格：将表格内容转换为文本格式
            for table in soup.find_all('table'):
                self._process_table(table)

            # 处理列表
            for ul in soup.find_all(['ul', 'ol']):
                self._process_list(ul)

            # 获取纯文本
            text = soup.get_text()

            # 解码HTML实体
            text = html.unescape(text)

            return text

        except Exception as e:
            print("BeautifulSoup处理失败", e)
            return text

    def _process_table(self, table):
        """处理HTML表格，转换为更易读的文本格式"""
        # 为表格行添加换行符
        for tr in table.find_all('tr'):
            tr.append('\n')

        # 为表格单元格添加分隔符
        for td in table.find_all(['td', 'th']):
            if td.string:
                td.string.replace_with(f" {td.string} ")

    def _process_list(self, list_element):
        """处理HTML列表，添加适当的格式"""
        for li in list_element.find_all('li'):
            if li.string:
                li.string.replace_with(f"• {li.string}\n")

    def _post_process_text(self, text: str) -> str:
        """后处理文本：清理多余空白和格式"""
        # 移除引用标记如 [1], [2], [3] 等
        text = re.sub(r'\[[0-9]+\]', '', text)

        # 清理多余的空行
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

        # 清理行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # 移除多余的空格
        text = re.sub(r' +', ' ', text)

        # 清理首尾空白
        text = text.strip()

        return text
