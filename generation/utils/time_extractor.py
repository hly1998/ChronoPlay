"""时间提取工具函数"""

from typing import List, Dict, Any, Optional
from datetime import datetime


def extract_latest_date_from_docs(retrieved_docs: List[Dict[str, Any]]) -> Optional[str]:
    """
    从检索到的文档中提取最晚的extracted_date

    Args:
        retrieved_docs: 检索到的文档列表

    Returns:
        最晚的日期字符串（YYYY-MM-DD格式）或None
    """
    if not retrieved_docs:
        return None

    valid_dates = []

    for doc in retrieved_docs:
        # 检查文档的metadata中是否有extracted_date
        metadata = doc.get('metadata', {})
        extracted_date = metadata.get('extracted_date')

        if extracted_date and extracted_date != 'null' and extracted_date is not None:
            try:
                # 验证日期格式并转换为datetime对象用于比较
                date_obj = datetime.strptime(extracted_date, "%Y-%m-%d")
                valid_dates.append((extracted_date, date_obj))
            except (ValueError, TypeError):
                # 跳过无效的日期格式
                continue

    if not valid_dates:
        return None

    # 找到最晚的日期
    latest_date = max(valid_dates, key=lambda x: x[1])
    return latest_date[0]


def format_qa_time_field(retrieved_docs: List[Dict[str, Any]], segment_id: int = None) -> Optional[str]:
    """
    格式化QA对的时间字段，支持回退策略

    Args:
        retrieved_docs: 检索到的文档列表
        segment_id: 当前分段ID，用于回退策略

    Returns:
        格式化后的时间字符串或None
    """
    # 首先尝试从检索到的文档中提取时间
    latest_date = extract_latest_date_from_docs(retrieved_docs)

    if latest_date:
        return latest_date

    # 如果仍然没有找到有效时间，返回None
    return None
