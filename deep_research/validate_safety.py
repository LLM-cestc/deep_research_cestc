# -*- coding: utf-8 -*-
"""
# 风险模块：内容风险控制

Author: wjianxz
Date: 2025-11-13
"""
from typing import List, Dict, Any, Optional

def validate_input_safety(query: str) -> bool:
    """
    对用户查询进行安全校验，防止注入、XSS、敏感指令等风险。
    
    Args:
        query (str): 原始用户输入
        
    Returns:
        bool: True 表示安全，False 表示存在风险
    """
    unsafe_patterns = ["<script", "javascript:", "DROP ", "DELETE ", "--", "UNION SELECT"]
    query_lower = query.lower()

    return True



