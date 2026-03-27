from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import re
from .orchestrator import run_plan
from .providers.market_provider import fetch_kline
from .indicators import build_indicators
from .services.vector_service import VectorService
from .services.llm_service import LLMService


@api_view(["GET"]) 
def health(request):
    return Response({"status": "ok"})


def _parse_time_window(text: str) -> str:
    if not text:
        return ""
    patterns = [
        (r"(\d+)\s*个月?", "M"),
        (r"(\d+)\s*月", "M"),
        (r"(\d+)\s*周", "W"),
        (r"(\d+)\s*天", "D"),
        (r"(\d+)\s*年", "Y"),
    ]
    for pat, suffix in patterns:
        m = re.search(pat, text)
        if m:
            return f"{m.group(1)}{suffix}"
    return ""


def _parse_dimensions(text: str) -> list:
    dims = []
    mapping = {
        "technical": ["技术", "指标", "K线", "均线", "MACD", "RSI", "布林"],
        "fundamental": ["基本面", "财报", "盈利", "估值", "ROE", "营收"],
        "sentiment": ["情绪", "舆情", "新闻", "热度", "微博", "论坛"],
    }
    for key, kws in mapping.items():
        if any(k in text for k in kws):
            dims.append(key)
    return dims or ["technical"]


def _parse_output(text: str) -> str:
    if any(k in text for k in ["视频", "配音", "字幕"]):
        return "video"
    return "article"


def _parse_ticker(text: str) -> str:
    # 支持形如 600519、000001、SZ:300750、SH600519、宁德时代(300750)
    m = re.search(r"(SH|SZ)[:]?\s?(\d{6})", text, re.I)
    if m:
        return f"{m.group(1).upper()}:{m.group(2)}"
    m = re.search(r"(\d{6})", text)
    if m:
        return m.group(1)
    return ""


from .intents import global_intent_service

@api_view(["POST"]) 
def intent_parse(request):
    payload = request.data or {}
    text = str(payload.get("text", ""))
    ticker = payload.get("ticker", "")
    time_window = payload.get("time_window", "")
    dimensions = payload.get("dimensions", [])
    output = payload.get("output", "")
    input_type = payload.get("input_type", "auto")  # auto, text, form, voice
    recognizer_name = payload.get("recognizer", None)  # rule_based, nlp_based, hybrid

    if not ticker and not text:
        return Response({"detail": "缺少 text 或 ticker"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # 根据输入类型选择不同的处理方式
        if input_type == 'form' or (not text and (ticker or time_window or dimensions or output)):
            # 使用结构化表单数据
            structured_data = {
                'ticker': ticker,
                'time_window': time_window,
                'dimensions': dimensions,
                'output': output
            }
            result = global_intent_service.parse_structured_input(structured_data)
        elif input_type == 'voice':
            # 处理语音输入
            voice_data = {
                'transcribed_text': text,
                'duration': payload.get('duration'),
                'confidence': payload.get('confidence')
            }
            result = global_intent_service.parse_voice_input(voice_data)
        else:
            # 处理文本输入（默认）
            result = global_intent_service.recognize_intent(
                text=text,
                recognizer_name=recognizer_name,
                ticker=ticker,
                time_window=time_window,
                dimensions=dimensions,
                output=output
            )
        result["raw"] = {"text": text}
        return Response(result)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"]) 
def plan_run(request):
    payload = request.data or {}
    plan = payload.get("plan", [])
    context = payload.get("context", {})
    try:
        output = run_plan(plan=plan, context=context)
        return Response(output)
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"]) 
def market_kline(request):
    ticker = request.query_params.get("ticker", "")
    window = request.query_params.get("window", "3M")
    if not ticker:
        return Response({"detail": "缺少 ticker"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        klines = fetch_kline(ticker=ticker, window=window)
        inds = build_indicators(klines)
        return Response({"ticker": ticker, "window": window, "klines": klines, "indicators": inds})
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def chat(request):
    """对话接口"""
    payload = request.data or {}
    message = payload.get("message", "")
    user_id = payload.get("user_id", "default_user")
    
    if not message:
        return Response({"detail": "缺少 message"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        vector_service = VectorService()
        llm_service = LLMService()
        
        # 搜索相似对话作为上下文
        similar_conversations = vector_service.search_similar(message, user_id, limit=3)
        context = [conv["metadata"] for conv in similar_conversations]
        
        # 生成回复
        response = llm_service.chat_response(message, context)
        
        # 保存对话到向量数据库
        vector_service.add_conversation(user_id, message, response)
        
        return Response({
            "message": message,
            "response": response,
            "context_count": len(context)
        })
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def chat_history(request):
    """获取对话历史"""
    user_id = request.query_params.get("user_id", "default_user")
    limit = int(request.query_params.get("limit", 10))
    
    try:
        vector_service = VectorService()
        history = vector_service.get_user_history(user_id, limit)
        return Response({"history": history})
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Create your views here.

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from datetime import datetime

from .models import GeneratedContent, ContentTemplate, VideoProductionJob, ContentReview
from .services.content_service import ContentService
from .services.review_service import ReviewService


@api_view(["POST"])
def generate_content(request):
    """生成内容（文章或视频）"""
    try:
        data = request.data
        ticker = data.get('ticker')
        content_type = data.get('content_type', 'article')  # article or video
        template = data.get('template', 'professional')
        style = data.get('style', 'professional')
        format_type = data.get('format', 'markdown')
        
        if not ticker:
            return Response({'error': '股票代码不能为空'}, status=status.HTTP_400_BAD_REQUEST)
        
        service = ContentService()

        # 创建内容记录
        content = GeneratedContent.objects.create(
            ticker=ticker,
            content_type=content_type,
            format=format_type,
            title=f"{ticker}分析内容",
            status='generating'
        )

        if content_type == 'article':
            result = service.generate_article(ticker, template, style, format_type)
        elif content_type == 'video':
            result = service.enqueue_video_job(content, style=style)
        else:
            return Response({'error': '不支持的内容类型'}, status=status.HTTP_400_BAD_REQUEST)
        
        if result.get('success'):
            # 更新内容记录
            content.title = result.get('title', content.title)
            content.content = result.get('content', '')
            content.file_path = result.get('file_path', '')
            content.thumbnail_path = result.get('thumbnail_path', '')
            content.status = 'completed'
            content.analysis_data = result.get('analysis_data', {})
            content.metadata = result.get('metadata', {})
            content.word_count = result.get('word_count', 0)
            content.duration = result.get('duration', 0)
            content.save()
            
            return Response({
                'success': True,
                'content_id': content.id,
                'title': content.title,
                'content': content.content,
                'file_path': content.file_path,
                'thumbnail_path': content.thumbnail_path,
                'metadata': content.metadata
            })
        else:
            content.status = 'failed'
            content.save()
            return Response({'error': result.get('error', '生成失败')}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _generate_article(*args, **kwargs):
    return {'success': False, 'error': 'deprecated'}


def _generate_video(*args, **kwargs):
    return {'success': False, 'error': 'deprecated'}


def _process_video_async(*args, **kwargs):
    return None


@api_view(["GET"])
def content_list(request):
    """获取内容列表"""
    try:
        content_type = request.query_params.get('type', '')
        ticker = request.query_params.get('ticker', '')
        status_filter = request.query_params.get('status', '')
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))
        
        queryset = GeneratedContent.objects.all()
        
        if content_type:
            queryset = queryset.filter(content_type=content_type)
        if ticker:
            queryset = queryset.filter(ticker__icontains=ticker)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        total = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        
        contents = queryset[start:end]
        
        data = []
        for content in contents:
            data.append({
                'id': content.id,
                'ticker': content.ticker,
                'content_type': content.content_type,
                'format': content.format,
                'title': content.title,
                'status': content.status,
                'word_count': content.word_count,
                'duration': content.duration,
                'created_at': content.created_at.isoformat(),
                'file_path': content.file_path,
                'thumbnail_path': content.thumbnail_path
            })
        
        return Response({
            'success': True,
            'data': data,
            'total': total,
            'page': page,
            'page_size': page_size
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def content_detail(request, content_id):
    """获取内容详情"""
    try:
        content = GeneratedContent.objects.get(id=content_id)
        
        return Response({
            'success': True,
            'data': {
                'id': content.id,
                'ticker': content.ticker,
                'content_type': content.content_type,
                'format': content.format,
                'title': content.title,
                'content': content.content,
                'status': content.status,
                'file_path': content.file_path,
                'thumbnail_path': content.thumbnail_path,
                'analysis_data': content.analysis_data,
                'metadata': content.metadata,
                'word_count': content.word_count,
                'duration': content.duration,
                'created_at': content.created_at.isoformat(),
                'updated_at': content.updated_at.isoformat()
            }
        })
        
    except GeneratedContent.DoesNotExist:
        return Response({'error': '内容不存在'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def video_job_status(request, job_id):
    """获取视频制作任务状态"""
    try:
        job = VideoProductionJob.objects.get(id=job_id)
        
        return Response({
            'success': True,
            'data': {
                'id': job.id,
                'status': job.status,
                'progress': job.progress,
                'error_message': job.error_message,
                'output_files': job.output_files,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'created_at': job.created_at.isoformat()
            }
        })
        
    except VideoProductionJob.DoesNotExist:
        return Response({'error': '任务不存在'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
def content_review(request, content_id):
    """提交内容审核"""
    try:
        data = request.data
        review_type = data.get('review_type', 'manual')
        status_filter = data.get('status', 'reviewing')
        comments = data.get('comments', '')
        suggestions = data.get('suggestions', '')
        
        content = GeneratedContent.objects.get(id=content_id)
        
        # 创建审核记录
        review = ContentReview.objects.create(
            content=content,
            review_type=review_type,
            status=status_filter,
            comments=comments,
            suggestions=suggestions
        )
        
        # 更新内容状态
        content.status = status_filter
        content.save()
        
        return Response({
            'success': True,
            'review_id': review.id,
            'message': '审核提交成功'
        })
        
    except GeneratedContent.DoesNotExist:
        return Response({'error': '内容不存在'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
def content_refine(request):
    """内容润色/改写接口（Grammarly风格）"""
    try:
        data = request.data or {}
        text = data.get('text', '')
        tone = data.get('tone', 'professional')  # professional/humor/concise
        language = data.get('language', 'zh-CN')

        if not text:
            return Response({'error': 'text 不能为空'}, status=status.HTTP_400_BAD_REQUEST)

        service = ReviewService()
        result = service.refine(text, tone=tone, language=language)
        return Response(result)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
def content_ai_review(request):
    """AI审核接口：输出风险点与修改建议"""
    try:
        data = request.data or {}
        text = data.get('text', '')
        focus = data.get('focus', ['事实准确性', '逻辑一致性', '用词风险'])

        if not text:
            return Response({'error': 'text 不能为空'}, status=status.HTTP_400_BAD_REQUEST)

        service = ReviewService()
        result = service.ai_review(text, focus=focus)
        return Response(result)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)