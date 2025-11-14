"""
@FileName: prompt_manage.py
@Description: 提示词统一管理模块
@Author: HengLine
@Time: 2025/11/10 18:52
"""
import os
from typing import Dict, Optional, List

import yaml

from hengline.logger import info, error, warning

# 获取prompts目录的绝对路径
PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))


class PromptManager:
    """
    提示词管理器，负责加载、缓存和获取各个智能体的提示词模板
    """

    def __init__(self):
        self._prompt_cache: Dict[str, Dict[str, str]] = {}
        self._load_all_prompts()

    def _load_all_prompts(self):
        """
        加载所有智能体的提示词模板到缓存中
        """
        try:
            for filename in os.listdir(PROMPTS_DIR):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    agent_name = filename.split('.')[0]
                    file_path = os.path.join(PROMPTS_DIR, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            prompt_data = yaml.safe_load(f)
                            if prompt_data:
                                self._prompt_cache[agent_name] = prompt_data
                                info(f"成功加载{agent_name}的提示词模板")
                    except Exception as e:
                        error(f"加载{filename}失败: {str(e)}")
        except Exception as e:
            error(f"加载提示词目录失败: {str(e)}")

    def get_prompt(self, agent_name: str, prompt_key: str = 'analysis') -> Optional[str]:
        """
        获取指定智能体的指定提示词模板
        
        Args:
            agent_name: 智能体名称
            prompt_key: 提示词键名，默认为'analysis'
            
        Returns:
            Optional[str]: 提示词模板内容，如果不存在返回None
        """
        # 如果缓存中没有，尝试重新加载
        if agent_name not in self._prompt_cache:
            file_extension = '.yaml' if os.path.exists(os.path.join(PROMPTS_DIR, f'{agent_name}.yaml')) else '.yml'
            file_path = os.path.join(PROMPTS_DIR, f'{agent_name}{file_extension}')
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompt_data = yaml.safe_load(f)
                        if prompt_data:
                            self._prompt_cache[agent_name] = prompt_data
                            info(f"重新加载{agent_name}的提示词模板")
                except Exception as e:
                    error(f"重新加载{agent_name}失败: {str(e)}")

        # 从缓存中获取
        agent_prompts = self._prompt_cache.get(agent_name, {})
        prompt_template = agent_prompts.get(prompt_key)

        if prompt_template is None:
            warning(f"未找到{agent_name}的{prompt_key}提示词模板")

        return prompt_template

    def update_prompt_cache(self, agent_name: str, prompt_key: str, prompt_content: str):
        """
        更新内存中的提示词缓存（不写入文件）
        
        Args:
            agent_name: 智能体名称
            prompt_key: 提示词键名
            prompt_content: 提示词内容
        """
        if agent_name not in self._prompt_cache:
            self._prompt_cache[agent_name] = {}
        self._prompt_cache[agent_name][prompt_key] = prompt_content
        info(f"更新{agent_name}的{prompt_key}提示词缓存")

    def save_prompt_to_file(self, agent_name: str, prompt_key: str, prompt_content: str):
        """
        保存提示词到YAML文件
        
        Args:
            agent_name: 智能体名称
            prompt_key: 提示词键名
            prompt_content: 提示词内容
        """
        file_path = os.path.join(PROMPTS_DIR, f'{agent_name}.yaml')

        # 如果文件存在，读取现有内容
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = yaml.safe_load(f) or {}
            except Exception as e:
                error(f"读取{agent_name}提示词文件失败: {str(e)}")
                existing_data = {}
        else:
            existing_data = {}

        # 更新数据
        existing_data[prompt_key] = prompt_content

        # 写入文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            info(f"成功保存{agent_name}的{prompt_key}提示词到文件")
            # 同步更新缓存
            self.update_prompt_cache(agent_name, prompt_key, prompt_content)
        except Exception as e:
            error(f"保存{agent_name}提示词文件失败: {str(e)}")

    def list_available_prompts(self) -> Dict[str, List[str]]:
        """
        列出所有可用的提示词模板
        
        Returns:
            Dict[str, List[str]]: 智能体名称到其可用提示词键名列表的映射
        """
        result = {}
        for agent_name, prompts in self._prompt_cache.items():
            result[agent_name] = list(prompts.keys())
        return result


# 创建全局提示词管理器实例
prompt_manager = PromptManager()


def get_prompt(agent_name: str, prompt_key: str = 'analysis') -> Optional[str]:
    """
    便捷函数：获取指定智能体的指定提示词模板
    
    Args:
        agent_name: 智能体名称
        prompt_key: 提示词键名，默认为'analysis'
        
    Returns:
        Optional[str]: 提示词模板内容
    """
    return prompt_manager.get_prompt(agent_name, prompt_key)


def update_prompt_cache(agent_name: str, prompt_key: str, prompt_content: str):
    """
    便捷函数：更新提示词缓存
    
    Args:
        agent_name: 智能体名称
        prompt_key: 提示词键名
        prompt_content: 提示词内容
    """
    prompt_manager.update_prompt_cache(agent_name, prompt_key, prompt_content)


def save_prompt_to_file(agent_name: str, prompt_key: str, prompt_content: str):
    """
    便捷函数：保存提示词到文件
    
    Args:
        agent_name: 智能体名称
        prompt_key: 提示词键名
        prompt_content: 提示词内容
    """
    prompt_manager.save_prompt_to_file(agent_name, prompt_key, prompt_content)


def list_available_prompts() -> Dict[str, List[str]]:
    """
    便捷函数：列出所有可用的提示词
    
    Returns:
        Dict[str, List[str]]: 智能体名称到提示词键名列表的映射
    """
    return prompt_manager.list_available_prompts()
