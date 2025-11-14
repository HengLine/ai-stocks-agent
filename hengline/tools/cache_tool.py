import time
from typing import Dict, Any, Optional, List
from collections import OrderedDict


class CacheEntry:
    """
    缓存条目类，用于存储缓存数据、创建时间和过期时间
    """
    def __init__(self, data: Any, expiry_time: float):
        self.data = data
        self.created_at = time.time()
        self.expiry_time = expiry_time

    def is_valid(self) -> bool:
        """
        检查缓存是否有效（未过期）
        """
        return time.time() < self.expiry_time


class CacheManager:
    """
    通用缓存管理器，支持设置过期时间和最大缓存项数
    使用OrderedDict实现LRU（最近最少使用）策略
    """
    def __init__(self, max_items: int = 100, default_expiry_seconds: int = 300):
        """
        初始化缓存管理器
        
        Args:
            max_items: 最大缓存项数，超过后会移除最不常用的项
            default_expiry_seconds: 默认缓存过期时间（秒）
        """
        # 确保使用OrderedDict来支持LRU功能
        self._cache = OrderedDict()
        self.max_items = max_items
        self.default_expiry_seconds = default_expiry_seconds

    def _generate_key(self, base_key: str, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            base_key: 基础键名
            *args: 位置参数，会被转换为字符串并添加到键中
            **kwargs: 关键字参数，会被排序并添加到键中
            
        Returns:
            生成的缓存键
        """
        parts = [base_key]
        
        # 添加位置参数
        if args:
            parts.extend(str(arg) for arg in args)
        
        # 添加关键字参数（排序以确保一致性）
        if kwargs:
            sorted_items = sorted(kwargs.items())
            parts.extend(f"{k}={v}" for k, v in sorted_items)
        
        return ":".join(parts)

    def get(self, key: str, *args, **kwargs) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 基础缓存键
            *args: 生成完整键的位置参数
            **kwargs: 生成完整键的关键字参数
            
        Returns:
            缓存的数据，如果缓存不存在或已过期则返回None
        """
        full_key = self._generate_key(key, *args, **kwargs)
        
        if full_key not in self._cache:
            return None
        
        entry = self._cache[full_key]
        
        # 检查缓存是否过期
        if not entry.is_valid():
            del self._cache[full_key]
            return None
        
        # 更新LRU顺序（将访问的项移到最后）
        try:
            # 尝试使用move_to_end方法（Python 3.2+）
            self._cache.move_to_end(full_key)
        except AttributeError:
            # 兼容Python 3.1及以下版本
            value = self._cache.pop(full_key)
            self._cache[full_key] = value
        return entry.data

    def set(self, key: str, data: Any, expiry_seconds: Optional[int] = None, 
            *args, **kwargs) -> None:
        """
        设置缓存数据
        
        Args:
            key: 基础缓存键
            data: 要缓存的数据
            expiry_seconds: 缓存过期时间（秒），如果不指定则使用默认值
            *args: 生成完整键的位置参数
            **kwargs: 生成完整键的关键字参数
        """
        full_key = self._generate_key(key, *args, **kwargs)
        
        # 设置过期时间
        if expiry_seconds is None:
            expiry_seconds = self.default_expiry_seconds
        
        expiry_time = time.time() + expiry_seconds
        
        # 如果缓存已满，移除最不常用的项
        if len(self._cache) >= self.max_items and full_key not in self._cache:
            self._cache.popitem()
        
        # 存储数据
        self._cache[full_key] = CacheEntry(data, expiry_time)

    def clear(self, key_prefix: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            key_prefix: 如果指定，则只清除以该前缀开头的缓存项
                        如果不指定，则清除所有缓存项
        """
        if key_prefix:
            # 清除指定前缀的缓存
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(key_prefix)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            # 清除所有缓存
            self._cache.clear()

    def get_keys(self, key_prefix: Optional[str] = None) -> List[str]:
        """
        获取缓存键列表
        
        Args:
            key_prefix: 如果指定，则只返回以该前缀开头的缓存键
                        如果不指定，则返回所有缓存键
            
        Returns:
            缓存键列表
        """
        if key_prefix:
            return [k for k in self._cache.keys() if k.startswith(key_prefix)]
        return list(self._cache.keys())

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计信息的字典
        """
        valid_count = 0
        invalid_count = 0
        total_size = 0
        
        # 检查所有缓存项
        for key, entry in list(self._cache.items()):
            if entry.is_valid():
                valid_count += 1
                total_size += len(str(entry.data))
            else:
                invalid_count += 1
                # 清理过期项
                del self._cache[key]
        
        return {
            "total_items": valid_count,
            "invalidated_items": invalid_count,
            "max_items": self.max_items,
            "default_expiry_seconds": self.default_expiry_seconds,
            "estimated_size_bytes": total_size
        }


# 创建全局缓存管理器实例
# 用于API响应的缓存管理器（24小时过期）
api_cache_manager = CacheManager(max_items=200, default_expiry_seconds=86400)

# 用于价格数据的缓存管理器（5分钟过期）
price_cache_manager = CacheManager(max_items=100, default_expiry_seconds=300)

# 用于股票信息的缓存管理器（1小时过期）
stock_info_cache_manager = CacheManager(max_items=500, default_expiry_seconds=3600)

# 用于新闻数据的缓存管理器（30分钟过期）
news_cache_manager = CacheManager(max_items=200, default_expiry_seconds=1800)

# 用于财务数据的缓存管理器（24小时过期）
financial_cache_manager = CacheManager(max_items=300, default_expiry_seconds=86400)


# 便捷函数
def get_price_cache(symbol: str, period: str, interval: str) -> Optional[Any]:
    """
    获取价格数据缓存
    """
    return price_cache_manager.get("price", symbol=symbol, period=period, interval=interval)


def set_price_cache(symbol: str, period: str, interval: str, data: Any) -> None:
    """
    设置价格数据缓存
    """
    price_cache_manager.set("price", data, symbol=symbol, period=period, interval=interval)


def get_stock_info_cache(symbol: str) -> Optional[Any]:
    """
    获取股票信息缓存
    """
    return stock_info_cache_manager.get("stock_info", symbol=symbol)


def set_stock_info_cache(symbol: str, data: Any) -> None:
    """
    设置股票信息缓存
    """
    stock_info_cache_manager.set("stock_info", data, symbol=symbol)


def get_news_cache(symbol: str, limit: int = 10) -> Optional[Any]:
    """
    获取新闻数据缓存
    """
    return news_cache_manager.get("news", symbol=symbol, limit=limit)


def set_news_cache(symbol: str, data: Any, limit: int = 10) -> None:
    """
    设置新闻数据缓存
    """
    news_cache_manager.set("news", data, symbol=symbol, limit=limit)


def get_financial_cache(symbol: str, report_type: str, years: int = 3) -> Optional[Any]:
    """
    获取财务数据缓存
    """
    return financial_cache_manager.get("financial", symbol=symbol, report_type=report_type, years=years)


def set_financial_cache(symbol: str, data: Any, report_type: str, years: int = 3) -> None:
    """
    设置财务数据缓存
    """
    financial_cache_manager.set("financial", data, symbol=symbol, report_type=report_type, years=years)


def get_api_cache(cache_key: str) -> Optional[Any]:
    """
    获取API响应缓存
    
    Args:
        cache_key: 缓存键
        
    Returns:
        缓存的数据，如果缓存不存在或已过期则返回None
    """
    return api_cache_manager.get("api", cache_key=cache_key)


def set_api_cache(cache_key: str, data: Any, ttl: Optional[int] = None) -> None:
    """
    设置API响应缓存
    
    Args:
        cache_key: 缓存键
        data: 要缓存的数据
        ttl: 缓存过期时间（秒），如果不指定则使用默认值
    """
    api_cache_manager.set("api", data, expiry_seconds=ttl, cache_key=cache_key)


def clear_all_caches() -> None:
    """
    清除所有缓存
    """
    api_cache_manager.clear()
    price_cache_manager.clear()
    stock_info_cache_manager.clear()
    news_cache_manager.clear()
    financial_cache_manager.clear()