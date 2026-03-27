#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内容生成模块使用示例

展示如何使用ContentService生成文章/视频内容，包括：
1. 生成文章初稿
2. AI审核文章
3. 人工审核文章
4. 润色文章
5. 格式转换
6. 生成视频脚本
"""

from typing import Dict, Any
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.apps.api.services.content_service import ContentService


def generate_and_process_article_example():
    """文章生成和处理完整流程示例"""
    # 初始化内容服务
    content_service = ContentService()
    
    # 1. 准备分析数据（通常来自AnalysisAgent）
    # 这里使用模拟数据，实际应用中应从AnalysisAgent获取
    analysis_data = {
        "time_frame": "3个月",  # 时间范围
        "technical_analysis": {
            "ma_signal": "金叉",
            "rsi_signal": "超买",
            "macd_signal": "看涨"
        },
        "fundamental_analysis": {
            "revenue_growth": "30%",
            "gross_margin": "稳定"
        },
        "risk_assessment": {
            "risk_level": "中等"
        },
        "investment_suggestion": {
            "action": "buy",
            "confidence": "medium",
            "reasoning": "综合技术面和基本面分析，建议买入"
        }
    }
    
    # 2. 生成文章初稿
    print("\n=== 步骤1: 生成文章初稿 ===")
    article_params = {
        "template": "professional",  # 模板类型
        "style": "professional",     # 风格
        "format": "markdown"          # 初始格式
    }
    
    generate_result = content_service.generate_content(
        content_type="article",
        ticker="AAPL",
        analysis_data=analysis_data,
        params=article_params
    )
    
    if generate_result["success"]:
        print(f"文章生成成功！内容ID: {generate_result['content_id']}")
        print(f"标题: {generate_result['content']['title']}")
        print(f"状态: {generate_result['status']}")
        
        # 保存内容ID用于后续操作
        content_id = generate_result["content_id"]
    else:
        print(f"文章生成失败: {generate_result['error']}")
        return
    
    # 3. AI审核文章
    print("\n=== 步骤2: AI审核文章 ===")
    review_focus = ["事实准确性", "逻辑一致性", "用词风险", "投资建议合理性"]
    
    ai_review_result = content_service.review_content_with_ai(
        content_id=content_id,
        focus=review_focus
    )
    
    if ai_review_result["success"]:
        print(f"AI审核完成！状态: {ai_review_result['status']}")
        print("审核问题:")
        for issue in ai_review_result['review'].get('issues', []):
            print(f"- [{issue.get('level', 'unknown')}] {issue.get('type', 'unknown')}: {issue.get('detail', '')}")
        
        print("修改建议:")
        for suggestion in ai_review_result['review'].get('suggestions', []):
            print(f"- {suggestion}")
    else:
        print(f"AI审核失败: {ai_review_result['error']}")
        return
    
    # 4. 人工审核文章（模拟）
    print("\n=== 步骤3: 人工审核文章 ===")
    manual_issues = [
        {"type": "内容完善", "detail": "建议增加最近一个季度的具体财务数据", "level": "medium"},
        {"type": "格式调整", "detail": "章节标题格式需要统一", "level": "low"}
    ]
    
    manual_suggestions = [
        "补充2023年第四季度营收和利润数据",
        "统一使用'## 二级标题'的格式",
        "风险提示部分建议增加具体的市场风险因素"
    ]
    
    manual_review_result = content_service.review_content_manually(
        content_id=content_id,
        reviewer_id=1,  # 审核人ID
        issues=manual_issues,
        suggestions=manual_suggestions,
        approved=True   # 通过审核
    )
    
    if manual_review_result["success"]:
        print(f"人工审核记录已保存！审核ID: {manual_review_result['review_id']}")
        print(f"审核结果: {'通过' if manual_review_result['status'] == 'approved' else '拒绝'}")
    else:
        print(f"人工审核记录保存失败: {manual_review_result['error']}")
        return
    
    # 5. 润色文章
    print("\n=== 步骤4: 润色文章 ===")
    refine_result = content_service.refine_content(
        content_id=content_id,
        tone="professional",     # 专业语气
        language="zh-CN"          # 中文
    )
    
    if refine_result["success"]:
        print(f"文章润色成功！状态: {refine_result['status']}")
        print("润色后前100字符:")
        print(refine_result['refined_text'][:100] + "...")
    else:
        print(f"文章润色失败: {refine_result['error']}")
        return
    
    # 6. 格式转换 - 转换为HTML
    print("\n=== 步骤5: 格式转换 (HTML) ===")
    html_format_result = content_service.format_content(
        content_id=content_id,
        target_format="html"
    )
    
    if html_format_result["success"]:
        print(f"HTML格式转换成功！")
        print("HTML内容前100字符:")
        print(html_format_result['formatted_content'][:100] + "...")
    else:
        print(f"HTML格式转换失败: {html_format_result['error']}")
    
    # 7. 格式转换 - 转换为PDF
    print("\n=== 步骤6: 格式转换 (PDF) ===")
    pdf_format_result = content_service.format_content(
        content_id=content_id,
        target_format="pdf"
    )
    
    if pdf_format_result["success"]:
        print(f"PDF格式转换成功！")
        print("PDF内容已准备就绪")
    else:
        print(f"PDF格式转换失败: {pdf_format_result['error']}")


def generate_video_script_example():
    """视频脚本生成示例"""
    # 初始化内容服务
    content_service = ContentService()
    
    # 1. 准备分析数据
    analysis_data = {
        "time_frame": "1个月",
        "technical_analysis": {
            "ma_signal": "死叉",
            "rsi_signal": "超卖",
            "macd_signal": "看跌"
        },
        "trend_analysis": {
            "short_term": "下跌",
            "long_term": "震荡",
            "strength": "medium",
            "price_change_percent": -5.2
        },
        "risk_level": "high",
        "investment_suggestion": {
            "action": "sell",
            "confidence": "medium"
        },
        "key_points": [
            "MACD形成死叉",
            "成交量萎缩",
            "RSI进入超卖区间"
        ]
    }
    
    # 2. 生成视频脚本
    print("\n=== 视频脚本生成示例 ===")
    video_params = {
        "style": "professional",   # 视频风格
        "duration": "medium",      # 视频时长
        "aspect_ratio": "16:9",    # 宽高比
        "language": "zh-CN"        # 语言
    }
    
    video_result = content_service.generate_content(
        content_type="video",
        ticker="TSLA",
        analysis_data=analysis_data,
        params=video_params
    )
    
    if video_result["success"]:
        print(f"视频脚本生成成功！内容ID: {video_result['content_id']}")
        print(f"视频标题: {video_result['content']['script']['title']}")
        print(f"总时长: {video_result['content']['script']['total_duration']}秒")
        print(f"场景数量: {len(video_result['content']['script']['scenes'])}")
        
        # 打印第一个场景示例
        if video_result['content']['script']['scenes']:
            first_scene = video_result['content']['script']['scenes'][0]
            print(f"\n第一个场景:\n类型: {first_scene['section']}\n内容: {first_scene['content']}\n时长: {first_scene['duration']}秒")
    else:
        print(f"视频脚本生成失败: {video_result['error']}")


def main():
    """主函数，运行所有示例"""
    print("===== AIGC股票分析内容生成模块示例 =====")
    
    # 运行文章生成和处理示例
    print("\n\n[文章生成和处理完整流程]")
    generate_and_process_article_example()
    
    # 运行视频脚本生成示例
    print("\n\n[视频脚本生成流程]")
    generate_video_script_example()
    
    print("\n\n===== 示例运行完成 =====")


if __name__ == "__main__":
    main()