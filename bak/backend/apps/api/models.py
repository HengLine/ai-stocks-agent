from django.db import models
from django.contrib.auth.models import User


class ContentTemplate(models.Model):
    """内容模板模型"""
    TEMPLATE_TYPES = [
        ('article', '文章模板'),
        ('video', '视频模板'),
    ]
    
    STYLES = [
        ('professional', '专业风格'),
        ('humor', '幽默风格'),
        ('brief', '简洁风格'),
        ('detailed', '详细风格'),
    ]
    
    name = models.CharField(max_length=100, verbose_name="模板名称")
    template_type = models.CharField(max_length=20, choices=TEMPLATE_TYPES, verbose_name="模板类型")
    style = models.CharField(max_length=20, choices=STYLES, verbose_name="风格")
    title_template = models.TextField(verbose_name="标题模板")
    content_template = models.JSONField(verbose_name="内容模板")
    is_active = models.BooleanField(default=True, verbose_name="是否启用")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        verbose_name = "内容模板"
        verbose_name_plural = "内容模板"
        unique_together = ['name', 'template_type', 'style']
    
    def __str__(self):
        return f"{self.name} ({self.get_template_type_display()})"


class GeneratedContent(models.Model):
    """生成的内容模型"""
    CONTENT_TYPES = [
        ('article', '文章'),
        ('video', '视频'),
    ]
    
    FORMATS = [
        ('markdown', 'Markdown'),
        ('html', 'HTML'),
        ('plain', '纯文本'),
        ('mp4', 'MP4视频'),
    ]
    
    STATUS_CHOICES = [
        ('generating', '生成中'),
        ('completed', '已完成'),
        ('failed', '失败'),
        ('reviewing', '审核中'),
        ('approved', '已审核'),
        ('rejected', '已拒绝'),
    ]
    
    ticker = models.CharField(max_length=20, verbose_name="股票代码")
    content_type = models.CharField(max_length=20, choices=CONTENT_TYPES, verbose_name="内容类型")
    format = models.CharField(max_length=20, choices=FORMATS, verbose_name="输出格式")
    template = models.ForeignKey(ContentTemplate, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="使用的模板")
    title = models.CharField(max_length=200, verbose_name="标题")
    content = models.TextField(verbose_name="内容")
    file_path = models.CharField(max_length=500, blank=True, null=True, verbose_name="文件路径")
    thumbnail_path = models.CharField(max_length=500, blank=True, null=True, verbose_name="缩略图路径")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='generating', verbose_name="状态")
    analysis_data = models.JSONField(blank=True, null=True, verbose_name="分析数据")
    metadata = models.JSONField(blank=True, null=True, verbose_name="元数据")
    word_count = models.IntegerField(default=0, verbose_name="字数")
    duration = models.FloatField(default=0, verbose_name="时长(秒)")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="创建者")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        verbose_name = "生成内容"
        verbose_name_plural = "生成内容"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.ticker} - {self.get_content_type_display()}"


class ContentReview(models.Model):
    """内容审核模型"""
    REVIEW_TYPES = [
        ('manual', '人工审核'),
        ('ai', 'AI审核'),
    ]
    
    content = models.ForeignKey(GeneratedContent, on_delete=models.CASCADE, verbose_name="内容")
    review_type = models.CharField(max_length=20, choices=REVIEW_TYPES, verbose_name="审核类型")
    reviewer = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="审核者")
    status = models.CharField(max_length=20, choices=GeneratedContent.STATUS_CHOICES, verbose_name="审核状态")
    comments = models.TextField(blank=True, null=True, verbose_name="审核意见")
    suggestions = models.TextField(blank=True, null=True, verbose_name="修改建议")
    reviewed_at = models.DateTimeField(auto_now=True, verbose_name="审核时间")
    
    class Meta:
        verbose_name = "内容审核"
        verbose_name_plural = "内容审核"
    
    def __str__(self):
        return f"{self.content.ticker} - {self.get_status_display()}"


class VideoProductionJob(models.Model):
    """视频制作任务模型"""
    STATUS_CHOICES = [
        ('pending', '等待中'),
        ('processing', '处理中'),
        ('completed', '已完成'),
        ('failed', '失败'),
    ]
    
    content = models.ForeignKey(GeneratedContent, on_delete=models.CASCADE, verbose_name="关联内容")
    script_data = models.JSONField(verbose_name="脚本数据")
    production_plan = models.JSONField(verbose_name="制作计划")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name="状态")
    progress = models.IntegerField(default=0, verbose_name="进度百分比")
    error_message = models.TextField(blank=True, null=True, verbose_name="错误信息")
    output_files = models.JSONField(blank=True, null=True, verbose_name="输出文件")
    started_at = models.DateTimeField(null=True, blank=True, verbose_name="开始时间")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="完成时间")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    
    class Meta:
        verbose_name = "视频制作任务"
        verbose_name_plural = "视频制作任务"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.content.ticker} - 视频制作任务"


class ContentAnalytics(models.Model):
    """内容分析模型"""
    content = models.ForeignKey(GeneratedContent, on_delete=models.CASCADE, verbose_name="内容")
    view_count = models.IntegerField(default=0, verbose_name="浏览次数")
    like_count = models.IntegerField(default=0, verbose_name="点赞次数")
    share_count = models.IntegerField(default=0, verbose_name="分享次数")
    download_count = models.IntegerField(default=0, verbose_name="下载次数")
    engagement_score = models.FloatField(default=0, verbose_name="参与度评分")
    last_updated = models.DateTimeField(auto_now=True, verbose_name="最后更新")
    
    class Meta:
        verbose_name = "内容分析"
        verbose_name_plural = "内容分析"
    
    def __str__(self):
        return f"{self.content.ticker} - 分析数据"
