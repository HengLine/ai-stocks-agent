#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: chief_strategy_agent.py
@Description: 首席策略智能体，负责整合其他专业智能体的分析结果并生成最终的投资建议
@Author: HengLine
@Time: 2025/11/10
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

from hengline.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from hengline.logger import debug, error, warning


class ChiefStrategyAgent(BaseAgent):
    """首席策略官智能体"""
    
    def __init__(self, config: AgentConfig = None):
        """
        初始化首席策略官智能体
        
        Args:
            config: 智能体配置
        """
        super().__init__(config)
        
        # 投资建议类型
        self.investment_recommendations = {
            "强烈买入": {
                "description": "强烈建议买入该股票，预期收益显著高于风险",
                "confidence": "非常高",
                "suitable_for": "激进型、积极型投资者",
                "signal_strength": 5
            },
            "买入": {
                "description": "建议买入该股票，预期收益大于风险",
                "confidence": "高",
                "suitable_for": "积极型、平衡型投资者",
                "signal_strength": 4
            },
            "持有": {
                "description": "建议持有现有仓位，收益与风险相对平衡",
                "confidence": "中等",
                "suitable_for": "平衡型、保守型投资者",
                "signal_strength": 3
            },
            "谨慎观望": {
                "description": "建议暂时观望，等待更明确的买入信号",
                "confidence": "低",
                "suitable_for": "保守型投资者",
                "signal_strength": 2
            },
            "卖出": {
                "description": "建议卖出该股票，风险显著高于预期收益",
                "confidence": "高",
                "suitable_for": "所有类型投资者",
                "signal_strength": 1
            }
        }
        
        # 风险等级定义
        self.risk_levels = {
            "低风险": {"range": (0, 20), "color": "green"},
            "中低风险": {"range": (21, 40), "color": "light-green"},
            "中等风险": {"range": (41, 60), "color": "yellow"},
            "中高风险": {"range": (61, 80), "color": "orange"},
            "高风险": {"range": (81, 100), "color": "red"}
        }
        
        # 各智能体权重配置
        self.agent_weights = {
            "FundamentalAgent": 0.25,      # 基本面分析权重
            "TechnicalAgent": 0.20,        # 技术面分析权重
            "IndustryMacroAgent": 0.15,    # 行业与宏观分析权重
            "SentimentAgent": 0.15,        # 舆情与情绪分析权重
            "FundFlowAgent": 0.15,         # 股东与资金流分析权重
            "ESGRiskAgent": 0.10           # ESG与治理风险分析权重
        }
    
    def analyze(self, stock_code: str, agent_results: Dict[str, AgentResult], **kwargs) -> AgentResult:
        """
        整合各智能体分析结果，生成最终投资建议
        
        Args:
            stock_code: 股票代码
            agent_results: 各智能体的分析结果字典
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 最终分析结果
        """
        try:
            debug(f"首席策略官开始整合股票 {stock_code} 的分析结果")
            
            # 验证并过滤有效结果
            filtered_results = self._filter_valid_results(agent_results)
            
            # 计算综合评分
            composite_score = self._calculate_composite_score(filtered_results)
            
            # 评估整体风险
            overall_risk = self._assess_overall_risk(filtered_results)
            
            # 识别关键优势与风险
            key_strengths, key_risks = self._identify_key_factors(filtered_results)
            
            # 检索相关知识库信息
            knowledge = self._retrieve_knowledge(f"投资决策 组合管理 风险评估 市场时机")
            
            # 生成分析提示词
            prompt = self._generate_analysis_prompt(
                stock_code, filtered_results, composite_score, 
                overall_risk, key_strengths, key_risks
            )
            
            # 使用LLM生成综合分析
            llm_analysis = self._generate_analysis(prompt, knowledge)
            
            # 结构化最终结果
            result = self._structure_result(
                stock_code, llm_analysis, filtered_results, 
                composite_score, overall_risk
            )
            
            debug(f"首席策略官完成股票 {stock_code} 的综合分析")
            
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                result=result,
                confidence_score=result.get("confidence_score", 0.90)
            )
            
        except Exception as e:
            error(f"首席策略官分析失败: {str(e)}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                result=self.get_result_template(),
                error_message=str(e),
                confidence_score=0.0
            )
    
    def _filter_valid_results(self, agent_results: Dict[str, AgentResult]) -> Dict[str, Dict[str, Any]]:
        """
        过滤有效分析结果
        
        Args:
            agent_results: 各智能体的分析结果字典
            
        Returns:
            Dict[str, Dict[str, Any]]: 有效分析结果
        """
        filtered = {}
        
        for agent_name, result in agent_results.items():
            if result.success and result.result:
                filtered[agent_name] = result.result
                debug(f"成功包含 {agent_name} 的分析结果，置信度: {result.confidence_score}")
            else:
                warning(f"跳过 {agent_name} 的分析结果，原因: {result.error_message or '无效结果'}")
        
        return filtered
    
    def _calculate_composite_score(self, filtered_results: Dict[str, Dict[str, Any]]) -> float:
        """
        计算综合评分
        
        Args:
            filtered_results: 有效分析结果
            
        Returns:
            float: 综合评分 (0-100)
        """
        if not filtered_results:
            return 50.0  # 默认中性评分
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent_name, result in filtered_results.items():
            # 获取智能体权重
            weight = self.agent_weights.get(agent_name, 0.1)
            
            # 从结果中提取评分信息（这里需要根据不同智能体的结果结构定制）
            score = self._extract_agent_score(agent_name, result)
            
            # 加权求和
            weighted_sum += score * weight
            total_weight += weight
        
        # 避免除零错误
        if total_weight == 0:
            return 50.0
        
        # 计算加权平均分
        composite_score = weighted_sum / total_weight
        
        # 确保分数在0-100范围内
        return max(0, min(100, composite_score))
    
    def _extract_agent_score(self, agent_name: str, result: Dict[str, Any]) -> float:
        """
        从不同智能体的结果中提取评分
        
        Args:
            agent_name: 智能体名称
            result: 智能体分析结果
            
        Returns:
            float: 评分 (0-100)
        """
        # 根据不同智能体定制评分提取逻辑
        if agent_name == "FundamentalAgent":
            # 基本面评分（越高越好）
            return result.get("fundamental_score", 50.0)
        
        elif agent_name == "TechnicalAgent":
            # 技术面评分（越高越好）
            return result.get("technical_score", 50.0)
        
        elif agent_name == "IndustryMacroAgent":
            # 行业与宏观评分（越高越好）
            industry_score = result.get("industry_analysis", {}).get("industry_score", 50.0)
            macro_score = result.get("macro_analysis", {}).get("economic_score", 50.0)
            return (industry_score + macro_score) / 2
        
        elif agent_name == "SentimentAgent":
            # 情绪分析评分（需要从多个指标综合）
            sentiment_score = 50.0  # 默认值
            
            # 尝试从不同位置获取情绪指标
            sentiment_metrics = result.get("sentiment_metrics", {})
            if sentiment_metrics:
                news_sentiment = sentiment_metrics.get("news_sentiment", {})
                pos = news_sentiment.get("positive", 0)
                neg = news_sentiment.get("negative", 0)
                total = pos + neg
                if total > 0:
                    # 转换为0-100分
                    sentiment_score = (pos / total) * 100
            
            return sentiment_score
        
        elif agent_name == "FundFlowAgent":
            # 资金流评分（越高越好）
            key_metrics = result.get("key_metrics", {})
            flow_classification = key_metrics.get("flow_classification", "neutral")
            
            # 根据资金流分类转换为评分
            flow_scores = {
                "strong_inflow": 90,
                "moderate_inflow": 75,
                "weak_inflow": 60,
                "neutral": 50,
                "weak_outflow": 40,
                "moderate_outflow": 25,
                "strong_outflow": 10
            }
            
            return flow_scores.get(flow_classification, 50.0)
        
        elif agent_name == "ESGRiskAgent":
            # ESG评分（越高越好）
            esg_metrics = result.get("esg_metrics", {})
            return esg_metrics.get("overall_score", 50.0)
        
        # 默认评分
        return 50.0
    
    def _assess_overall_risk(self, filtered_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估整体风险
        
        Args:
            filtered_results: 有效分析结果
            
        Returns:
            Dict[str, Any]: 风险评估结果
        """
        overall_risk = {
            "risk_score": 50.0,  # 默认中等风险
            "risk_level": "中等风险",
            "risk_dimensions": {},
            "risk_factors": [],
            "risk_mitigation": []
        }
        
        risk_scores = []
        risk_factors = []
        
        # 从各智能体结果中提取风险信息
        for agent_name, result in filtered_results.items():
            if agent_name == "FundamentalAgent":
                # 基本面风险
                financial_risk = result.get("financial_risks", [])
                risk_factors.extend(financial_risk)
                risk_scores.append(100 - result.get("fundamental_score", 50.0))
                
            elif agent_name == "TechnicalAgent":
                # 技术面风险
                technical_risks = result.get("risk_signals", [])
                risk_factors.extend(technical_risks)
                risk_scores.append(100 - result.get("technical_score", 50.0))
                
            elif agent_name == "ESGRiskAgent":
                # ESG风险
                esg_metrics = result.get("esg_metrics", {})
                esg_risk_level = esg_metrics.get("risk_level", "")
                if esg_risk_level:
                    risk_factors.append(f"ESG风险等级: {esg_risk_level}")
                    
                # ESG风险转换为风险分数
                esg_risk_scores = {
                    "low": 20,
                    "medium": 50,
                    "high": 70,
                    "severe": 90
                }
                risk_scores.append(esg_risk_scores.get(esg_risk_level, 50.0))
        
        # 计算平均风险分数
        if risk_scores:
            overall_risk["risk_score"] = sum(risk_scores) / len(risk_scores)
        
        # 确定风险等级
        risk_score = overall_risk["risk_score"]
        for level, info in self.risk_levels.items():
            min_risk, max_risk = info["range"]
            if min_risk <= risk_score <= max_risk:
                overall_risk["risk_level"] = level
                break
        
        # 添加风险因素
        overall_risk["risk_factors"] = list(set(risk_factors))[:5]  # 去重并保留前5个
        
        # 生成风险缓解建议
        overall_risk["risk_mitigation"] = self._generate_risk_mitigation(overall_risk["risk_level"])
        
        return overall_risk
    
    def _generate_risk_mitigation(self, risk_level: str) -> List[str]:
        """
        生成风险缓解建议
        
        Args:
            risk_level: 风险等级
            
        Returns:
            List[str]: 风险缓解建议
        """
        mitigation_advice = {
            "低风险": ["可考虑适当增加仓位", "关注公司业绩持续改善", "长期持有策略可能适合"],
            "中低风险": ["适度配置，密切关注关键指标变化", "可考虑分批建仓策略", "设置止损位保护收益"],
            "中等风险": ["建议仓位适中，控制在投资组合20%以内", "设置止损位在8-10%", "定期重新评估基本面变化"],
            "中高风险": ["建议谨慎配置，控制在投资组合10%以内", "设置严格止损位在5-7%", "密切关注风险信号变化"],
            "高风险": ["建议极小仓位试探或暂时观望", "如有持仓考虑逐步减仓", "等待风险因素明朗后再评估"]
        }
        
        return mitigation_advice.get(risk_level, ["建议谨慎对待，根据个人风险承受能力决策"])
    
    def _identify_key_factors(self, filtered_results: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        识别关键优势和风险因素
        
        Args:
            filtered_results: 有效分析结果
            
        Returns:
            Tuple[List[str], List[str]]: (关键优势, 关键风险)
        """
        strengths = []
        risks = []
        
        # 从各智能体结果中提取关键发现
        for agent_name, result in filtered_results.items():
            # 提取优势
            key_findings = result.get("key_findings", [])
            for finding in key_findings:
                # 简单的文本分类，实际项目中应该使用更复杂的NLP
                if any(keyword in finding.lower() for keyword in ["优势", "增长", "改善", "强劲", "正面", "上升", "创新"]):
                    strengths.append(finding)
                elif any(keyword in finding.lower() for keyword in ["风险", "挑战", "下降", "负面", "担忧", "问题", "不确定性"]):
                    risks.append(finding)
            
            # 专门提取风险信息
            if "risk_signals" in result:
                risks.extend(result["risk_signals"])
            
            if "financial_risks" in result:
                risks.extend(result["financial_risks"])
        
        # 去重并限制数量
        strengths = list(set(strengths))[:5]
        risks = list(set(risks))[:5]
        
        return strengths, risks
    
    def _generate_analysis_prompt(self, stock_code: str, filtered_results: Dict[str, Dict[str, Any]],
                                 composite_score: float, overall_risk: Dict[str, Any],
                                 key_strengths: List[str], key_risks: List[str]) -> str:
        """
        生成综合分析提示词
        
        Args:
            stock_code: 股票代码
            filtered_results: 有效分析结果
            composite_score: 综合评分
            overall_risk: 整体风险评估
            key_strengths: 关键优势
            key_risks: 关键风险
            
        Returns:
            str: 分析提示词
        """
        # 导入langchain相关组件和提示词管理器
        from langchain_core.prompts import ChatPromptTemplate
        from hengline.prompts.prompt_manage import get_prompt
        
        # 从提示词管理器获取提示模板
        template = get_prompt('chief_strategy_agent', 'analysis')
        
        # 如果获取失败，使用备用模板
        if not template:
            template = """
作为资深投资策略专家，请对股票代码 {{stock_code}} 进行全面、平衡的综合分析并提供最终投资建议。

综合评分：{{composite_score}}/100
整体风险评估：{{risk_level}} (风险分数: {{risk_score}})

请提供详细分析和明确的投资建议。
"""
        
        # 创建提示模板实例
        prompt_template = ChatPromptTemplate.from_template(template)
        
        # 格式化并返回提示内容
        prompt = prompt_template.format(
            stock_code=stock_code,
            composite_score=f"{composite_score:.2f}",
            risk_level=overall_risk['risk_level'],
            risk_score=f"{overall_risk['risk_score']:.2f}",
            key_strengths=json.dumps(key_strengths, indent=2, ensure_ascii=False),
            key_risks=json.dumps(key_risks, indent=2, ensure_ascii=False),
            filtered_results=json.dumps(filtered_results, indent=2, ensure_ascii=False)
        )
        
        return prompt
    
    def _structure_result(self, stock_code: str, llm_analysis: Dict[str, Any],
                         filtered_results: Dict[str, Dict[str, Any]], composite_score: float,
                         overall_risk: Dict[str, Any]) -> Dict[str, Any]:
        """
        结构化最终结果
        
        Args:
            stock_code: 股票代码
            llm_analysis: LLM生成的分析
            filtered_results: 各智能体分析结果
            composite_score: 综合评分
            overall_risk: 整体风险评估
            
        Returns:
            Dict[str, Any]: 结构化的结果
        """
        result = self.get_result_template()
        
        # 获取建议详情
        recommendation = llm_analysis.get("investment_recommendation", "持有")
        recommendation_details = self.investment_recommendations.get(recommendation, {})
        
        result.update({
            "stock_code": stock_code,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "investment_recommendation": recommendation,
            "recommendation_description": recommendation_details.get("description", ""),
            "recommendation_confidence": recommendation_details.get("confidence", ""),
            "signal_strength": recommendation_details.get("signal_strength", 3),
            "recommendation_details": llm_analysis.get("recommendation_details", ""),
            "suitable_investors": llm_analysis.get("suitable_investors", ""),
            "position_suggestion": llm_analysis.get("position_suggestion", ""),
            "key_monitoring_metrics": llm_analysis.get("key_monitoring_metrics", []),
            "risk_disclosure": llm_analysis.get("risk_disclosure", ""),
            "confidence_score": llm_analysis.get("confidence_score", 0.90),
            "analysis_summary": llm_analysis.get("analysis_summary", ""),
            "comprehensive_metrics": {
                "composite_score": round(composite_score, 2),
                "risk_score": round(overall_risk["risk_score"], 2),
                "risk_level": overall_risk["risk_level"],
                "agents_used": list(filtered_results.keys())
            },
            "agent_contributions": {}
        })
        
        # 添加各智能体的贡献摘要
        for agent_name, agent_result in filtered_results.items():
            result["agent_contributions"][agent_name] = {
                "key_findings": agent_result.get("key_findings", []),
                "confidence": agent_result.get("confidence_score", 0.0)
            }
        
        # 验证结果
        if not self.validate_result(result):
            warning("首席策略官分析结果验证失败，使用默认结构")
            result = self.get_result_template()
        
        return result