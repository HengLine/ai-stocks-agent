#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载Hugging Face嵌入模型的工具脚本

该脚本用于预先下载嵌入模型到本地缓存，解决VectorStoreManager无法加载本地模型的问题。
"""

import os
import sys
import argparse
import traceback
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hengline.logger import logger

def download_model(model_name, cache_dir="./data/cache"):
    """
    下载Hugging Face模型到本地缓存
    
    Args:
        model_name: 模型名称或路径
        cache_dir: 缓存目录路径
    """
    try:
        logger.info(f"开始下载模型: {model_name}")
        logger.info(f"缓存目录: {os.path.abspath(cache_dir)}")
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f"已确保缓存目录存在")
        
        # 下载tokenizer
        logger.debug("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        logger.debug("tokenizer下载完成")
        
        # 下载模型
        logger.debug("下载模型...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        logger.debug("模型下载完成")
        
        # 保存模型以确保完全缓存
        logger.debug("保存模型到缓存...")
        tokenizer.save_pretrained(os.path.join(cache_dir, model_name.split('/')[-1]))
        model.save_pretrained(os.path.join(cache_dir, model_name.split('/')[-1]))
        logger.debug("模型保存完成")
        
        logger.info(f"模型下载完成: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"下载模型时出错: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def test_model(model_name, cache_dir="./data/cache"):
    """
    测试下载的模型是否能被LangChain正确加载
    
    Args:
        model_name: 模型名称或路径
        cache_dir: 缓存目录路径
    """
    try:
        logger.info(f"测试模型加载: {model_name}")
        
        # 使用local_files_only=True测试本地模型加载
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=cache_dir,
            local_files_only=True  # 强制使用本地文件
        )
        logger.debug("模型已加载")
        
        # 测试嵌入功能
        test_text = "这是一个测试句子，用于验证嵌入模型是否正常工作"
        logger.debug(f"测试嵌入功能，输入文本: {test_text}")
        embedding = embeddings.embed_query(test_text)
        
        logger.info(f"模型加载成功!")
        logger.debug(f"嵌入向量维度: {len(embedding)}")
        logger.debug(f"嵌入向量示例: {embedding[:5]}")
        return True
        
    except Exception as e:
        logger.error(f"测试模型加载时出错: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def main():
    """主函数"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="下载Hugging Face嵌入模型工具")
    parser.add_argument(
        "--model", 
        type=str, 
        # default="sentence-transformers/all-MiniLM-L6-v2",
        default="thenlper/gte-large-zh",
        help=f"要下载的模型名称，默认为: 'thenlper/gte-large-zh'")
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="./cache",
        help="模型缓存目录，默认为'./data/cache'")
    parser.add_argument(
        "--test", 
        action="store_true",
        help="下载后测试模型加载")
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    logger.debug("="*80)
    logger.debug("            Hugging Face 嵌入模型下载工具            ")
    logger.debug("="*80)
    logger.debug("这个工具可以帮助您预先下载嵌入模型到本地缓存，")
    logger.debug("解决VectorStoreManager无法加载本地模型的问题。")
    logger.debug(f"模型={args.model}, 缓存目录={args.cache_dir}, 测试模式={args.test}")
    
    # 下载模型
    success = download_model(args.model, args.cache_dir)
    
    # 如果下载成功且需要测试，则测试模型
    if success and args.test:
        logger.debug("开始测试模型加载")
        test_model(args.model, args.cache_dir)
    
    # 提供后续使用指南
    logger.debug("="*80)
    logger.debug("使用指南:")
    logger.debug(f"1. 模型已下载到: {os.path.abspath(args.cache_dir)}")
    logger.debug("2. 您可以在.env文件中设置以下环境变量:")
    logger.debug(f"   EMBEDDING_MODEL={args.model}")
    logger.debug(f"   CACHE_DIR={os.path.abspath(args.cache_dir)}")
    logger.debug("3. 或者直接运行main.py，程序现在应该能够使用本地模型了")
    logger.debug("="*80)

if __name__ == "__main__":
    main()