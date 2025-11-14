# -*- coding: utf-8 -*-
"""
@FileName: index_api.py
@Description: FastAPI应用，提供索引接口
@Author: HengLine
@Time: 2025/10/22 23:40
"""

from fastapi import APIRouter

app = APIRouter()


# API文档路由
@app.get("/api", include_in_schema=False)
async def api_docs():
    """
    API文档路径，重定向到Swagger UI
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
