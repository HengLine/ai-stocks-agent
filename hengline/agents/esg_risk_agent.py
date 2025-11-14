#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: esg_risk_agent.py
@Description: ESG风险评估智能体，负责分析企业的环境、社会和治理风险
@Author: HengLine
@Time: 2025/11/10
"""

import json
from datetime import datetime
from typing import Dict, Any, List
import yfinance as yf
# 从stock_manage统一获取数据
from hengline.stock.stock_manage import StockDataManager

from hengline.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from hengline.logger import debug, error, warning
from utils.log_utils import print_log_exception


class ESGRiskAgent(BaseAgent):
    """ESG与治理风险分析智能体"""

    def __init__(self, config: AgentConfig = None):
        """
        初始化ESG与治理风险分析智能体
        
        Args:
            config: 智能体配置
        """
        super().__init__(config)

        # 初始化股票数据管理器
        self.stock_manager = StockDataManager()

        # ESG分析关键维度
        self.analysis_dimensions = [
            "environmental",  # 环境维度
            "social",  # 社会维度
            "governance",  # 治理维度
            "controversies",  # 争议事件
            "risk_assessment",  # 风险评估
            "sustainability"  # 可持续发展
        ]

        # ESG评分等级
        self.esg_rating_levels = {
            "AAA": (90, 100),  # 领先
            "AA": (80, 89),  # 很强
            "A": (70, 79),  # 强
            "BBB": (60, 69),  # 可接受
            "BB": (50, 59),  # 需要改进
            "B": (40, 49),  # 弱
            "CCC": (0, 39)  # 差
        }

        # 风险等级定义
        self.risk_levels = {
            "low": {"range": (0, 30), "color": "green", "description": "低风险"},
            "medium": {"range": (31, 60), "color": "yellow", "description": "中等风险"},
            "high": {"range": (61, 80), "color": "orange", "description": "高风险"},
            "severe": {"range": (81, 100), "color": "red", "description": "严重风险"}
        }

    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """
        执行ESG与治理风险分析
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 分析结果
        """
        try:
            debug(f"开始对股票 {stock_code} 进行ESG与治理风险分析")

            # 获取股票基本信息
            stock_info = self._get_stock_info(stock_code)

            # 获取ESG数据
            esg_data = self._get_esg_data(stock_code)

            # 获取公司治理数据
            governance_data = self._get_governance_data(stock_code)

            # 获取争议事件
            controversies = self._get_controversies(stock_code)

            # 评估ESG风险
            risk_assessment = self._assess_esg_risk(esg_data, governance_data, controversies)

            # 检索相关知识库信息
            knowledge = self._retrieve_knowledge(f"ESG分析 公司治理 可持续发展 风险管理")

            # 生成分析提示词
            prompt = self._generate_analysis_prompt(
                stock_code, stock_info, esg_data, governance_data,
                controversies, risk_assessment
            )

            # 使用LLM生成分析
            llm_analysis = self._generate_analysis(prompt, knowledge)

            # 结构化分析结果
            result = self._structure_result(
                stock_code, llm_analysis, stock_info, esg_data,
                governance_data, risk_assessment
            )

            debug(f"股票 {stock_code} ESG与治理风险分析完成")

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                result=result,
                confidence_score=result.get("confidence_score", 0.85)
            )

        except Exception as e:
            error(f"ESG与治理风险分析失败: {str(e)}")
            print_log_exception()
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                result=self.get_result_template(),
                error_message=str(e),
                confidence_score=0.0
            )

    def _get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 股票基本信息
        """
        try:
            # 从stock_manage获取股票信息
            stock_info = self.stock_manager.get_stock_info(stock_code)
            return {
                "name": stock_info.get("name", ""),
                "sector": stock_info.get("sector", ""),
                "industry": stock_info.get("industry", ""),
                "description": stock_info.get("description", ""),
                "employees": stock_info.get("employees", 0),
                "country": stock_info.get("country", ""),
                "website": stock_info.get("website", "")
            }
        except Exception as e:
            error(f"获取股票信息失败: {str(e)}")
            return {}

    def _get_esg_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取ESG数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: ESG数据
        """
        esg_data = {
            "overall_score": 0,
            "environmental_score": 0,
            "social_score": 0,
            "governance_score": 0,
            "rating": "",
            "peer_percentile": 0,
            "trend": "stable"
        }

        try:
            stock = yf.Ticker(stock_code)
            sustainability = stock.sustainability

            if not sustainability.empty:
                # 获取整体ESG评分
                overall_score = sustainability.loc[sustainability['Value'] == 'Overall ESG Score', 'Percentile'].values
                if len(overall_score) > 0:
                    esg_data["peer_percentile"] = float(overall_score[0])
                    # 转换百分位到0-100分
                    esg_data["overall_score"] = round(esg_data["peer_percentile"] * 0.7 + 30, 2)  # 调整分布

                # 获取各维度评分
                env_score = sustainability.loc[sustainability['Value'] == 'Environment Score', 'Percentile'].values
                if len(env_score) > 0:
                    esg_data["environmental_score"] = round(float(env_score[0]) * 0.7 + 30, 2)

                social_score = sustainability.loc[sustainability['Value'] == 'Social Score', 'Percentile'].values
                if len(social_score) > 0:
                    esg_data["social_score"] = round(float(social_score[0]) * 0.7 + 30, 2)

                governance_score = sustainability.loc[sustainability['Value'] == 'Governance Score', 'Percentile'].values
                if len(governance_score) > 0:
                    esg_data["governance_score"] = round(float(governance_score[0]) * 0.7 + 30, 2)

            # 如果没有直接评分，使用模拟数据（实际项目中应接入专业ESG数据提供商）
            if esg_data["overall_score"] == 0:
                # 基于行业和公司特征的模拟评分
                import random
                base_score = random.uniform(40, 80)

                # 假设科技公司通常ESG评分较高
                if "Technology" in stock_code or "Tech" in stock_code:
                    base_score += 10

                esg_data["overall_score"] = round(base_score, 2)
                esg_data["environmental_score"] = round(base_score + random.uniform(-10, 10), 2)
                esg_data["social_score"] = round(base_score + random.uniform(-10, 10), 2)
                esg_data["governance_score"] = round(base_score + random.uniform(-10, 10), 2)

            # 确定ESG评级
            esg_data["rating"] = self._calculate_esg_rating(esg_data["overall_score"])

        except Exception as e:
            error(f"获取ESG数据失败: {str(e)}")

        return esg_data

    def _get_governance_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取公司治理数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 公司治理数据
        """
        governance_data = {
            "board_independence": 0.7,  # 董事会独立性
            "executive_compensation": {},  # 高管薪酬
            "shareholder_rights": {},  # 股东权利
            "audit_quality": 0.8,  # 审计质量
            "anti_corruption": 0.9,  # 反腐败措施
            "transparency": 0.75  # 透明度
        }

        try:
            stock = yf.Ticker(stock_code)
            info = stock.info

            # 从基本信息中提取一些治理相关数据
            governance_data["executive_compensation"] = {
                "ceo_pay": info.get("CEO", "未知"),
                "payout_ratio": info.get("payoutRatio", 0)
            }

            # 模拟一些治理指标
            import random
            governance_data["board_independence"] = round(random.uniform(0.5, 0.95), 2)
            governance_data["audit_quality"] = round(random.uniform(0.6, 0.95), 2)
            governance_data["anti_corruption"] = round(random.uniform(0.7, 1.0), 2)
            governance_data["transparency"] = round(random.uniform(0.6, 0.9), 2)

            governance_data["shareholder_rights"] = {
                "proxy_access": random.choice([True, False]),
                "majority_voting": random.choice([True, False]),
                "poison_pill": random.choice([True, False])
            }

        except Exception as e:
            error(f"获取公司治理数据失败: {str(e)}")

        return governance_data

    def _get_controversies(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        获取ESG相关争议事件
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[Dict[str, Any]]: 争议事件列表
        """
        controversies = []

        try:
            stock = yf.Ticker(stock_code)
            sustainability = stock.sustainability

            if not sustainability.empty:
                # 从sustainability数据中提取争议信息
                controversy_columns = sustainability[sustainability['Value'].str.contains('Controversy', na=False)]

                for idx, row in controversy_columns.iterrows():
                    controversy_type = row['Value']
                    controversy_score = float(row['Percentile'])

                    controversies.append({
                        "type": controversy_type,
                        "score": controversy_score,
                        "severity": self._get_controversy_severity(controversy_score)
                    })

            # 如果没有争议数据，模拟一些可能的争议事件
            if not controversies:
                possible_controversies = [
                    "Business Ethics Controversy",
                    "Environmental Controversy",
                    "Human Rights Controversy",
                    "Product Safety Controversy",
                    "Labor Relations Controversy",
                    "Corporate Governance Controversy"
                ]

                import random
                num_controversies = random.randint(0, 3)  # 0-3个争议事件

                for i in range(num_controversies):
                    controversy_type = random.choice(possible_controversies)
                    possible_controversies.remove(controversy_type)  # 避免重复

                    score = random.uniform(0, 100)
                    controversies.append({
                        "type": controversy_type,
                        "score": score,
                        "severity": self._get_controversy_severity(score)
                    })

        except Exception as e:
            error(f"获取争议事件失败: {str(e)}")

        return controversies

    def _assess_esg_risk(self, esg_data, governance_data, controversies):
        """
        评估ESG风险 - 完全独立实现，包含最严格的类型检查
        
        Args:
            esg_data: ESG数据
            governance_data: 公司治理数据
            controversies: 争议事件列表
            
        Returns:
            Dict[str, Any]: 风险评估结果
        """
        # 创建完全独立的结果对象
        result = {
            "overall_risk_score": 0.0,
            "overall_risk_level": "中等风险",
            "risk_by_dimension": {
                "environmental": {"score": 0.0, "level": "中等风险"},
                "social": {"score": 0.0, "level": "中等风险"},
                "governance": {"score": 0.0, "level": "中等风险"}
            },
            "key_risk_factors": [],
            "opportunities": []
        }

        # 1. 完全独立的类型检查和默认值处理
        if esg_data is None or not isinstance(esg_data, dict):
            esg_data = {}
        if governance_data is None or not isinstance(governance_data, dict):
            governance_data = {}
        if controversies is None or not isinstance(controversies, list):
            controversies = []

        # 2. 独立的环境风险计算 - 完全避免使用外部方法
        env_risk = 50.0  # 默认中等风险
        try:
            # 最严格的字段检查
            env_score_val = esg_data.get("environmental_score", 50)
            # 关键修复：先检查是否为dict类型
            if isinstance(env_score_val, dict):
                env_score = 50.0
            else:
                # 尝试转换为float
                try:
                    env_score = float(env_score_val)
                except (ValueError, TypeError):
                    env_score = 50.0
            # 安全计算风险分数
            env_risk = float(100.0 - env_score)
        except Exception:
            env_risk = 50.0

        # 3. 独立的社会风险计算
        social_risk = 50.0  # 默认中等风险
        try:
            # 最严格的字段检查
            social_score_val = esg_data.get("social_score", 50)
            # 关键修复：先检查是否为dict类型
            if isinstance(social_score_val, dict):
                social_score = 50.0
            else:
                # 尝试转换为float
                try:
                    social_score = float(social_score_val)
                except (ValueError, TypeError):
                    social_score = 50.0
            # 安全计算风险分数
            social_risk = float(100.0 - social_score)
        except Exception:
            social_risk = 50.0

        # 4. 独立的治理风险计算 - 完全内部实现
        governance_risk = 50.0  # 默认中等风险
        try:
            # 内部安全转换函数
            def safe_to_float(val, default_val=0.0):
                if isinstance(val, dict):
                    return default_val
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default_val
            
            # 获取并安全转换治理指标
            board_ind = safe_to_float(governance_data.get("board_independence", 0.7))
            audit = safe_to_float(governance_data.get("audit_quality", 0.8))
            anti_corr = safe_to_float(governance_data.get("anti_corruption", 0.9))
            transp = safe_to_float(governance_data.get("transparency", 0.75))
            
            # 安全计算治理分数
            gov_score = float(
                board_ind * 25.0 +
                audit * 25.0 +
                anti_corr * 25.0 +
                transp * 25.0
            )
            governance_risk = float(100.0 - gov_score)
        except Exception:
            governance_risk = 50.0

        # 5. 独立的争议事件处理 - 关键修复位置
        controversy_impact = 0.0
        key_risks = []
        try:
            # 逐个处理争议事件
            for item in controversies:
                # 确保是字典
                if isinstance(item, dict):
                    # 获取严重性并转换为小写
                    severity = str(item.get("severity", "")).lower()
                    # 只处理高严重性事件
                    if severity in ["high", "severe"]:
                        # 关键修复：处理score字段
                        score_val = item.get("score", 0)
                        # 最关键的修复：检测dict类型score
                        if isinstance(score_val, dict):
                            # 如果是dict，使用默认值0
                            score_num = 0.0
                        else:
                            # 安全转换为float
                            try:
                                score_num = float(score_val)
                            except (ValueError, TypeError):
                                score_num = 0.0
                        
                        # 完全安全的累加
                        controversy_impact = float(controversy_impact + (score_num / 10.0))
                        
                        # 添加类型到关键风险因素
                        if "type" in item and isinstance(item["type"], str):
                            key_risks.append(item["type"])
        except Exception:
            # 任何错误都重置为默认值
            controversy_impact = 0.0
            key_risks = []

        # 6. 独立的整体风险计算 - 最严格的类型检查
        total_risk = 50.0
        try:
            # 计算各部分权重分数 - 每一步都确保是float
            env_part = float(env_risk * 0.25)
            social_part = float(social_risk * 0.25)
            governance_part = float(governance_risk * 0.3)
            controversy_part = float(controversy_impact * 0.2)
            
            # 最安全的累加方式 - 每次累加前都确保是float
            total_risk = float(0.0)
            total_risk = float(total_risk + env_part)
            total_risk = float(total_risk + social_part)
            total_risk = float(total_risk + governance_part)
            total_risk = float(total_risk + controversy_part)
            
            # 限制范围
            total_risk = max(0.0, min(100.0, total_risk))
        except Exception:
            total_risk = 50.0

        # 7. 内部定义的风险等级计算
        def get_risk_level(score_val):
            try:
                score = float(score_val)
                if score <= 30:
                    return "低风险"
                elif score <= 60:
                    return "中等风险"
                elif score <= 80:
                    return "高风险"
                else:
                    return "严重风险"
            except Exception:
                return "中等风险"

        # 8. 设置结果 - 每一步都确保是float
        try:
            # 设置整体风险分数和等级
            result["overall_risk_score"] = float(round(total_risk, 2))
            result["overall_risk_level"] = get_risk_level(total_risk)
            
            # 设置各维度风险
            result["risk_by_dimension"] = {
                "environmental": {
                    "score": float(round(env_risk, 2)),
                    "level": get_risk_level(env_risk)
                },
                "social": {
                    "score": float(round(social_risk, 2)),
                    "level": get_risk_level(social_risk)
                },
                "governance": {
                    "score": float(round(governance_risk, 2)),
                    "level": get_risk_level(governance_risk)
                }
            }
        except Exception:
            pass  # 保持默认值

        # 9. 设置关键风险因素
        result["key_risk_factors"] = key_risks[:5]  # 限制数量

        # 10. 识别ESG机会
        try:
            ops = []
            # 风险高表示改进机会大
            if float(env_risk) > 50:
                ops.append("环境合规改进潜力")
            if float(social_risk) > 50:
                ops.append("社会责任提升机会")
            if float(governance_risk) > 50:
                ops.append("公司治理优化空间")
            result["opportunities"] = ops[:3]  # 最多3个机会
        except Exception:
            result["opportunities"] = []

        # 返回完全安全的结果
        return result

    def _calculate_esg_rating(self, score: float) -> str:
        """
        根据分数计算ESG评级
        
        Args:
            score: ESG分数
            
        Returns:
            str: ESG评级
        """
        for rating, (min_score, max_score) in self.esg_rating_levels.items():
            if min_score <= score <= max_score:
                return rating
        return "CCC"  # 默认最低评级

    def _get_controversy_severity(self, score: float) -> str:
        """
        获取争议事件严重程度
        
        Args:
            score: 争议分数
            
        Returns:
            str: 严重程度
        """
        if score >= 75:
            return "severe"
        elif score >= 50:
            return "high"
        elif score >= 25:
            return "medium"
        else:
            return "low"

    def _get_risk_level(self, risk_score: float) -> str:
        """
        根据风险分数获取风险等级
        
        Args:
            risk_score: 风险分数
            
        Returns:
            str: 风险等级
        """
        for level, info in self.risk_levels.items():
            min_risk, max_risk = info["range"]
            if min_risk <= risk_score <= max_risk:
                return level
        return "medium"  # 默认中等风险

    def _generate_analysis_prompt(self, stock_code: str, stock_info: Dict[str, Any],
                                  esg_data: Dict[str, Any], governance_data: Dict[str, Any],
                                  controversies: List[Dict[str, Any]], risk_assessment: Dict[str, Any]) -> str:
        """
        生成分析提示词
        
        Args:
            stock_code: 股票代码
            stock_info: 股票基本信息
            esg_data: ESG数据
            governance_data: 公司治理数据
            controversies: 争议事件
            risk_assessment: 风险评估结果
            
        Returns:
            str: 分析提示词
        """
        # 导入langchain相关组件和提示词管理器
        from langchain_core.prompts import ChatPromptTemplate
        from hengline.prompts.prompt_manage import get_prompt

        # 从提示词管理器获取提示模板
        template = get_prompt('esg_risk_agent', 'analysis')

        # 如果获取失败，使用备用模板
        if not template:
            template = """
请对股票代码 {{stock_code}} 进行全面的ESG与治理风险分析。

股票基本信息：
{{stock_info}}

ESG数据：
{{esg_data}}

请提供ESG风险分析和投资影响评估。
"""

        # 创建提示模板实例
        prompt_template = ChatPromptTemplate.from_template(template)

        # 格式化并返回提示内容
        prompt = prompt_template.format(
            stock_code=stock_code,
            stock_info=json.dumps(stock_info, indent=2, ensure_ascii=False),
            esg_data=json.dumps(esg_data, indent=2, ensure_ascii=False),
            governance_data=json.dumps(governance_data, indent=2, ensure_ascii=False),
            controversies=json.dumps(controversies, indent=2, ensure_ascii=False),
            risk_assessment=json.dumps(risk_assessment, indent=2, ensure_ascii=False)
        )

        return prompt

    def _calculate_governance_quality(self, governance_data: Dict[str, Any]) -> float:
        """
        计算治理质量分数，只考虑数值类型的值
        
        Args:
            governance_data: 公司治理数据
            
        Returns:
            float: 治理质量分数
        """
        # 只提取数值类型的值
        numeric_values = [v for v in governance_data.values() if isinstance(v, (int, float))]
        
        # 如果有数值类型的值，则计算平均值
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
        else:
            return 0.0
            
    def _structure_result(self, stock_code: str, llm_analysis: Dict[str, Any],
                          stock_info: Dict[str, Any], esg_data: Dict[str, Any],
                          governance_data: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        结构化分析结果
        
        Args:
            stock_code: 股票代码
            llm_analysis: LLM生成的分析
            stock_info: 股票基本信息
            esg_data: ESG数据
            governance_data: 公司治理数据
            risk_assessment: 风险评估结果
            
        Returns:
            Dict[str, Any]: 结构化的结果
        """
        result = self.get_result_template()
        result.update({
            "stock_code": stock_code,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "company_name": stock_info.get("name", ""),
            "key_findings": llm_analysis.get("key_findings", []),
            "detailed_analysis": llm_analysis.get("detailed_analysis", {}),
            "esg_summary": llm_analysis.get("esg_summary", {}),
            "investment_impact": llm_analysis.get("investment_impact", ""),
            "risk_signals": llm_analysis.get("risk_signals", []),
            "improvement_recommendations": llm_analysis.get("improvement_recommendations", []),
            "confidence_score": llm_analysis.get("confidence_score", 0.85),
            "esg_metrics": {
                "overall_score": esg_data.get("overall_score", 0),
                "rating": esg_data.get("rating", ""),
                "risk_level": risk_assessment.get("overall_risk_level", ""),
                "governance_quality": self._calculate_governance_quality(governance_data) if governance_data else 0
            }
        })

        # 验证结果
        if not self.validate_result(result):
            warning("ESG与治理风险分析结果验证失败，使用默认结构")
            result = self.get_result_template()

        return result
