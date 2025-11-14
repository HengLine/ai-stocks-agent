from datetime import datetime, timedelta
from typing import Any

def format_date(date_obj: Any) -> str:
    """
    统一的日期格式化函数，将各种类型的日期转换为标准格式 YYYY-MM-DD
    
    Args:
        date_obj: 要格式化的日期对象（可以是datetime对象、字符串或其他类型）
        
    Returns:
        格式化后的日期字符串，格式为 YYYY-MM-DD
    """
    # 如果已经是datetime对象
    if hasattr(date_obj, 'strftime'):
        return date_obj.strftime('%Y-%m-%d')
    
    # 如果是字符串
    elif isinstance(date_obj, str):
        # 检查是否已经是标准格式 YYYY-MM-DD
        if len(date_obj) == 10 and date_obj[4] == '-' and date_obj[7] == '-':
            return date_obj
        
        # 尝试多种常见的日期格式
        formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%d-%m-%Y', '%d/%m/%Y']
        for fmt in formats:
            try:
                # 处理可能包含时间部分的字符串
                dt = datetime.strptime(date_obj.split(' ')[0], fmt)
                return dt.strftime('%Y-%m-%d')
            except (ValueError, IndexError):
                continue
    
    # 无法识别的格式，返回字符串表示
    return str(date_obj)

def get_date_range(days: int) -> list:
    """
    获取过去N天的日期列表
    
    Args:
        days: 天数
        
    Returns:
        日期对象列表，从最早到最晚排序
    """
    return [datetime.now() - timedelta(days=i) for i in range(days)][::-1]

def format_date_for_filename(date_obj: Any = None) -> str:
    """
    格式化日期用于文件名，使用YYYY-MM-DD格式
    
    Args:
        date_obj: 日期对象，如果为None则使用当前日期
        
    Returns:
        格式化的日期字符串
    """
    if date_obj is None:
        date_obj = datetime.now()
    return format_date(date_obj)

def format_date_for_chart(date_obj: Any) -> str:
    """
    格式化日期用于图表显示，使用x月x日格式（如：08月12日）
    
    Args:
        date_obj: 要格式化的日期对象
        
    Returns:
        格式化后的日期字符串，格式为MM月DD日
    """
    # 首先使用标准格式化函数获取标准格式
    standard_date = format_date(date_obj)
    try:
        # 解析标准格式的日期
        dt = datetime.strptime(standard_date, '%Y-%m-%d')
        # 返回x月x日格式
        return dt.strftime('%m月%d日')
    except ValueError:
        # 如果解析失败，返回原始字符串
        return standard_date

def get_current_year_range(years: int = 5) -> list:
    """
    获取当前年份往前推N年的年份列表
    
    Args:
        years: 年数
        
    Returns:
        年份字符串列表
    """
    current_year = datetime.now().year
    return [str(current_year - i) for i in range(years)]