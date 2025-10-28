# -*- coding: utf-8 -*-
"""
CSV数据加载器
从data.csv文件加载QA数据
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any


class CSVDataLoader:
    """CSV数据加载器"""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    def load_all_data(self) -> List[Dict[str, Any]]:
        """加载所有数据"""
        data = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 解析JSON字段
                parsed_row = self._parse_row(row)
                data.append(parsed_row)
        return data

    def load_by_game(self, game_name: str) -> List[Dict[str, Any]]:
        """按游戏名称加载数据"""
        all_data = self.load_all_data()
        return [item for item in all_data if item['game_name'] == game_name]

    def get_available_games(self) -> List[str]:
        """获取所有可用的游戏名称"""
        games = set()
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                games.add(row['game_name'])
        return sorted(list(games))

    def _parse_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """解析CSV行，将JSON字符串转换为对象"""
        parsed = {
            'game_name': row['game_name'],
            'question': row['question'],
            'answer': row['answer'],
            'question_topic': row['question_topic'],
            'task_type': row['task_type'],
            'time': row['time'],
        }

        # 解析JSON字段
        json_fields = ['references', 'retrieved_docs', 'entities']
        for field in json_fields:
            try:
                parsed[field] = json.loads(row[field]) if row[field] else []
            except json.JSONDecodeError:
                parsed[field] = []

        return parsed

    def convert_to_qa_pairs(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将数据转换为QA对格式（用于评估）"""
        qa_pairs = []
        for item in data:
            qa_pair = {
                'question': item['question'],
                'ground_truth_answer': item['answer'],
                'ground_truth_docs': item['retrieved_docs'],
                'ground_truth_doc_ids': [
                    doc.get('id', '') for doc in item['retrieved_docs']
                ] if item['retrieved_docs'] else [],
                'original_data': item
            }
            qa_pairs.append(qa_pair)
        return qa_pairs
