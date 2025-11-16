#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit问答模块
基于knowledge_base的增强RAG股票问答功能
"""

import time
from typing import Dict, Any

import streamlit as st

from hengline.client.client_factory import get_langchain_llm
from hengline.logger import info, warning, error
from hengline.rag.rag_chain import RAGChain
from hengline.rag.vector_store import VectorStoreManager


class StreamlitQA:
    """Streamlit问答界面类"""

    def __init__(self):
        """初始化问答模块"""
        self.rag_chain = None
        self.vector_store_manager = None
        self.llm_client = None

        # 初始化会话状态
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        if 'qa_initialized' not in st.session_state:
            st.session_state.qa_initialized = False

    def initialize_rag_system(self):
        """初始化RAG系统"""
        try:
            with st.spinner("正在初始化问答系统..."):
                # 创建LangChain兼容的LLM实例
                self.llm_client = get_langchain_llm()
                
                if self.llm_client is None:
                    st.error("无法创建语言模型实例，请检查配置")
                    return

                # 创建向量存储管理器，指向knowledge_base目录
                self.vector_store_manager = VectorStoreManager(
                    vector_store_path='data/knowledge_vector_store'
                )

                # 确保向量存储使用knowledge_base目录
                self._update_vector_store_from_knowledge_base()

                # 创建RAG链
                self.rag_chain = RAGChain(
                    model=self.llm_client,
                    vector_store_manager=self.vector_store_manager
                )

                st.session_state.qa_initialized = True
                st.success("问答系统初始化成功！")
                info("问答系统初始化完成")

        except Exception as e:
            st.error(f"问答系统初始化失败: {str(e)}")
            error(f"问答系统初始化失败: {str(e)}")
            st.session_state.qa_initialized = False

    def _update_vector_store_from_knowledge_base(self):
        """从knowledge_base目录更新向量存储"""
        try:
            # 检查knowledge_base目录是否存在
            import os
            kb_dir = 'knowledge_base'
            if not os.path.exists(kb_dir):
                warning(f"知识库目录不存在: {kb_dir}")
                return

            # 更新向量存储，使用knowledge_base作为数据源
            self.vector_store_manager.update_vector_store(kb_dir)
            info(f"已从 {kb_dir} 目录更新向量存储")

        except Exception as e:
            warning(f"更新向量存储时出错: {str(e)}")

    def render_qa_interface(self):
        """渲染问答界面"""
        st.markdown("## 智能股票问答")
        st.markdown("---")

        # 显示系统状态
        self._show_system_status()

        # 如果未初始化，显示初始化按钮
        if not st.session_state.qa_initialized:
            if st.button("初始化问答系统", type="primary", use_container_width=True):
                self.initialize_rag_system()
            return

        # 问答输入区域
        self._render_qa_input()

        # 问答历史
        self._render_qa_history()

        # 清除历史按钮
        if st.button("清除问答历史", use_container_width=True):
            st.session_state.qa_history = []
            st.rerun()

    def _show_system_status(self):
        """显示系统状态"""
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.qa_initialized:
                st.success("系统已就绪")
            else:
                st.warning("系统未初始化")

        with col2:
            history_count = len(st.session_state.qa_history)
            st.metric("问答历史", history_count)

        with col3:
            # 显示知识库状态
            self._show_knowledge_base_status()

    def _show_knowledge_base_status(self):
        """显示知识库状态"""
        try:
            import os
            kb_dir = 'knowledge_base'
            if os.path.exists(kb_dir):
                # 统计知识库文件数量
                txt_files = []
                for root, dirs, files in os.walk(kb_dir):
                    for file in files:
                        if file.endswith('.txt'):
                            txt_files.append(file)

                st.metric("知识库文件", len(txt_files))
            else:
                st.error("知识库不存在")
        except Exception as e:
            st.warning("知识库状态未知")

    def _render_qa_input(self):
        """渲染问答输入区域"""
        st.markdown("### 提问")

        # 预设问题示例
        with st.expander("点击查看示例问题", expanded=False):
            example_questions = [
                "什么是股票的市盈率？如何计算？",
                "股票技术分析中的MACD指标是什么意思？",
                "投资股票时需要注意哪些风险？",
                "什么是价值投资策略？",
                "如何进行股票的基本面分析？",
                "股票交易中的止损策略有哪些？",
                "什么是K线图？如何解读？",
                "投资心理学在股票交易中的作用是什么？"
            ]

            for i, question in enumerate(example_questions, 1):
                if st.button(f"{i}. {question}", key=f"example_{i}"):
                    st.session_state.current_question = question
                    st.rerun()

        # 问题输入
        question = st.text_input(
            "请输入您的股票相关问题：",
            value=st.session_state.get('current_question', ''),
            placeholder="例如：什么是市盈率？如何计算股票的内在价值？",
            key="qa_question_input"
        )

        # 清除临时问题
        if 'current_question' in st.session_state:
            del st.session_state.current_question

        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            retrieval_count = st.slider("检索文档数量", min_value=1, max_value=10, value=4)
        with col2:
            show_sources = st.checkbox("显示参考来源", value=True)

        # 提问按钮
        if st.button("提问", type="primary", use_container_width=True) and question.strip():
            self._process_question(question, retrieval_count, show_sources)

    def _process_question(self, question: str, retrieval_count: int, show_sources: bool):
        """处理用户问题"""
        if not self.rag_chain:
            st.error("问答系统未初始化")
            return

        start_time = time.time()

        with st.spinner("正在思考中，请稍候..."):
            try:
                # 获取相关文档（用于显示来源）
                relevant_docs = []
                if show_sources:
                    relevant_docs = self.rag_chain.get_relevant_docs(question, k=retrieval_count)

                # 生成回答
                answer = self.rag_chain.invoke(question, k=retrieval_count)

                # 计算耗时
                elapsed_time = time.time() - start_time

                # 保存到历史记录
                qa_record = {
                    'question': question,
                    'answer': answer,
                    'sources': relevant_docs if show_sources else [],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'elapsed_time': f"{elapsed_time:.2f}秒"
                }

                st.session_state.qa_history.append(qa_record)

                # 显示回答
                self._display_answer(qa_record)

                st.rerun()

            except Exception as e:
                st.error(f"回答生成失败: {str(e)}")
                error(f"问答处理失败: {str(e)}")

    def _display_answer(self, qa_record: Dict[str, Any]):
        """显示回答"""
        st.markdown("### 回答")
        st.markdown(f"**问题：** {qa_record['question']}")
        st.markdown(f"**回答：** {qa_record['answer']}")

        # 显示参考来源
        if qa_record['sources']:
            st.markdown("#### 参考来源")
            for i, doc in enumerate(qa_record['sources'], 1):
                with st.expander(f"来源 {i}: {doc.metadata.get('source', '未知来源')}", expanded=False):
                    st.markdown(f"**内容片段：**")
                    st.write(doc.page_content)
                    st.markdown(f"**文件路径：** `{doc.metadata.get('source', '未知')}`")

        # 显示时间信息
        st.caption(f"回答时间: {qa_record['timestamp']} | ⚡ 耗时: {qa_record['elapsed_time']}")

        st.markdown("---")

    def _render_qa_history(self):
        """渲染问答历史"""
        if not st.session_state.qa_history:
            st.info("暂无问答历史，请开始提问！")
            return

        st.markdown("### 问答历史")

        # 倒序显示历史记录
        for i, qa_record in enumerate(reversed(st.session_state.qa_history), 1):
            with st.expander(f" Q{i}: {qa_record['question'][:50]}...", expanded=False):
                st.markdown(f"**问题：** {qa_record['question']}")
                st.markdown(f"**回答：** {qa_record['answer']}")
                st.caption(f" 时间: {qa_record['timestamp']} | ⚡ 耗时: {qa_record['elapsed_time']}")

                # 显示来源（如果有）
                if qa_record['sources']:
                    if st.button(f" 查看来源 {i}", key=f"sources_{i}"):
                        for j, doc in enumerate(qa_record['sources'], 1):
                            st.markdown(f"**来源 {j}:** {doc.metadata.get('source', '未知来源')}")
                            with st.expander(f"内容片段 {j}", expanded=False):
                                st.write(doc.page_content)


def show_qa_view():
    """显示问答视图的主函数"""
    qa_module = StreamlitQA()
    qa_module.render_qa_interface()


if __name__ == "__main__":
    # 测试用
    show_qa_view()
