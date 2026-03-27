from django.urls import path
from .views import (
    health, intent_parse, plan_run, market_kline, chat, chat_history,
    generate_content, content_list, content_detail, video_job_status, content_review,
    content_refine, content_ai_review
)


urlpatterns = [
    path('health/', health, name='health'),
    path('intent/parse', intent_parse, name='intent-parse'),
    path('plan/run', plan_run, name='plan-run'),
    path('market/kline', market_kline, name='market-kline'),
    path('chat', chat, name='chat'),
    path('chat/history', chat_history, name='chat-history'),
    
    # 内容生成相关API
    path('content/generate', generate_content, name='generate-content'),
    path('content/list', content_list, name='content-list'),
    path('content/<int:content_id>', content_detail, name='content-detail'),
    path('video/job/<int:job_id>/status', video_job_status, name='video-job-status'),
    path('content/<int:content_id>/review', content_review, name='content-review'),
    path('content/refine', content_refine, name='content-refine'),
    path('content/ai-review', content_ai_review, name='content-ai-review'),
]


