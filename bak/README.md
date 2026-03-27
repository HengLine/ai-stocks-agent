# 股票智能平台 - AIGC Stocks Agent

<div align="center">
  <img src="https://placehold.co/800x400/e6f7ff/1890ff?text=AIGC+Stocks+Agent&font=roboto" alt="AIGC Stocks Agent" style="border-radius: 8px;" />
  <p style="margin-top: 16px; font-size: 18px; font-weight: bold;">智能驱动的股票分析与内容生成平台</p>
</div>

## 1. 项目概述

股票智能平台是一款结合用户交互、AI智能体协作和多模态内容生成能力的金融科技应用。平台旨在为用户提供智能化的股票分析、内容生成和投资辅助功能，通过先进的AI技术提升用户的投资决策效率。

### 1.1 核心价值
- **智能化股票分析**：基于多维度数据的智能分析和投资建议
- **自动化内容生成**：支持自动生成股票分析文章和视频
- **AI驱动的用户交互**：提供智能对话接口，实时解答用户问题
- **高效的智能体协作**：多个专业智能体协同工作，提供综合服务

## 2. 系统架构

平台采用前后端分离的现代化架构，具有良好的扩展性和可维护性。

### 2.1 整体架构图

```
┌─────────────────────────┐      ┌─────────────────────────┐
│       前端应用         │      │       后端服务         │
│  Vue3 + Vite + TypeScript│      │  Django + DRF + AI Agents│
├─────────────────────────┤      ├─────────────────────────┤
│ - 用户界面层           │      │ - API接口层            │
│ - 状态管理             │◄────►│ - 业务逻辑层            │
│ - 数据可视化           │      │ - 智能体协作层          │
│ - API请求处理         │      │ - 数据服务层            │
└─────────────────────────┘      └───────────┬───────────┘
                                             │
                              ┌─────────────┼─────────────┐
                              │             │             │
                     ┌────────▼─────┐ ┌─────▼────────┐ ┌───▼────────────┐
                     │  外部API集成 │ │  向量数据库  │ │  关系型数据库  │
                     │(OpenAI等)   │ │(ChromaDB)   │ │(SQLite/MySQL) │
                     └─────────────┘ └─────────────┘ └────────────────┘
```

### 2.2 技术栈选型

#### 2.2.1 前端技术
- **框架**：Vue 3 + TypeScript
- **构建工具**：Vite
- **路由**：Vue Router
- **状态管理**：Pinia
- **HTTP客户端**：Axios
- **图表库**：ECharts
- **UI工具**：@vueuse/core

#### 2.2.2 后端技术
- **框架**：Python Django + Django REST Framework
- **数据库**：SQLite（开发环境）/ PostgreSQL（生产环境）
- **向量数据库**：ChromaDB
- **AI服务集成**：OpenAI API
- **嵌入模型**：SentenceTransformer (all-MiniLM-L6-v2)
- **跨域处理**：django-cors-headers

## 3. 快速开始

### 3.1 开发环境要求
- Node.js 18+（含 npm）
- Python 3.10+
- Git

### 3.2 后端启动步骤

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 激活虚拟环境（macOS/Linux）
# source .venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 运行数据库迁移
python manage.py migrate

# 启动开发服务器
python manage.py runserver 0.0.0.0:8000
```

健康检查接口：`http://127.0.0.1:8000/api/health/`

### 3.3 前端启动步骤

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev -- --host --port 5173
```

本地访问：`http://127.0.0.1:5173`

### 3.4 环境变量配置

#### 3.4.1 后端环境变量

在`backend`目录下创建`.env`文件：

```env
# Django配置
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# OpenAI API配置（可选）
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

#### 3.4.2 前端环境变量

在`frontend`目录下创建`.env`文件：

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api
VITE_APP_TITLE=股票智能平台
```

## 4. 项目结构

```
aigc-stocks-agent/
├── backend/               # 后端工程（Django + DRF）
│   ├── apps/              # Django应用
│   │   └── api/           # API应用
│   │       ├── agents/    # AI智能体实现
│   │       ├── providers/ # 数据提供者
│   │       ├── services/  # 核心服务
│   │       ├── models.py  # 数据模型
│   │       ├── views.py   # 视图函数
│   │       └── urls.py    # URL配置
│   ├── core/              # Django项目核心配置
│   │   ├── settings.py    # 项目设置
│   │   ├── urls.py        # 主URL配置
│   │   └── wsgi.py        # WSGI入口
│   ├── db.sqlite3         # SQLite数据库（开发环境）
│   ├── manage.py          # Django管理脚本
│   └── requirements.txt   # Python依赖
├── frontend/              # 前端工程（Vue3 + Vite + TS）
│   ├── src/               # 源代码目录
│   │   ├── assets/        # 静态资源
│   │   ├── components/    # Vue组件
│   │   ├── router/        # 路由配置
│   │   ├── services/      # API服务
│   │   ├── views/         # 页面组件
│   │   ├── App.vue        # 根组件
│   │   ├── main.ts        # 入口文件
│   │   └── style.css      # 全局样式
│   ├── public/            # 静态资源
│   ├── index.html         # HTML入口
│   ├── package.json       # NPM依赖
│   └── vite.config.ts     # Vite配置
├── .gitignore             # Git忽略规则
└── README.md              # 项目说明文档
```

## 5. 核心功能模块

### 5.1 智能体协作系统

平台采用多智能体协作架构，通过不同专业领域的智能体协同工作，提供全面的股票分析和内容生成服务。

#### 5.1.1 主要智能体
- **DataAgent**：负责从市场数据源获取股票数据
- **AnalysisAgent**：提供技术分析和投资建议
- **WritingAgent**：生成股票分析文章
- **VideoAgent**：生成股票分析视频

#### 5.1.2 智能体协作流程
```python
# 智能体协作示例代码
from apps.api.orchestrator import run_plan

# 定义协作计划
plan = [{
    "ticker": "600519",
    "pipeline": ["DataAgent", "AnalysisAgent", "WritingAgent"],
    "params": {
        "DataAgent": {"window": "3M"},
        "WritingAgent": {"style": "professional"}
    }
}]

# 执行计划
result = run_plan(plan=plan)
print(result)
```

### 5.2 内容生成系统

平台支持自动生成股票分析文章和视频，帮助用户快速获取结构化的股票分析内容。

#### 5.2.1 文章生成
- 支持不同风格的文章生成（专业、幽默、简洁等）
- 自动整合技术指标和市场数据
- 提供结构化的分析报告

#### 5.2.2 视频生成
- 自动生成视频脚本
- 支持不同风格的视频制作
- 跟踪视频生成进度

### 5.3 对话系统

平台提供智能对话功能，用户可以直接与AI助手交流，获取股票相关问题的解答。

#### 5.3.1 对话功能
- 基于上下文的智能回复
- 支持股票代码查询和分析
- 对话历史记录和检索

#### 5.3.2 向量数据库支持
使用ChromaDB向量数据库存储和检索对话历史，提供更智能的上下文理解能力。

### 5.4 数据可视化

平台集成了ECharts图表库，提供丰富的数据可视化功能，帮助用户直观地理解股票走势和技术指标。

#### 5.4.1 图表功能
- K线图展示
- 技术指标叠加
- 交互式数据探索

## 6. API接口文档

### 6.1 基础API
- **GET /api/health/**：健康检查接口
- **POST /api/intent/parse**：意图解析接口
- **POST /api/plan/run**：计划执行接口

### 6.2 市场数据API
- **GET /api/market/kline**：获取股票K线数据

### 6.3 对话API
- **POST /api/chat**：对话接口
- **GET /api/chat/history**：获取对话历史

### 6.4 内容管理API
- **POST /api/content/generate**：生成内容
- **GET /api/content/list**：获取内容列表
- **GET /api/content/<id>**：获取内容详情
- **GET /api/video/job/<id>/status**：获取视频任务状态
- **POST /api/content/<id>/review**：内容审核
- **POST /api/content/refine**：内容润色
- **POST /api/content/ai-review**：AI审核

## 7. 部署指南

### 7.1 Docker部署

使用Docker Compose快速部署整个应用：

1. 创建`docker-compose.yml`文件：

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    container_name: aigc_stocks_backend
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - ./backend:/app/backend
      - ./chroma_db:/app/backend/chroma_db
    ports:
      - "8000:8000"
    environment:
      - DEBUG=True
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL}
    depends_on:
      - db
    networks:
      - aigc_stocks_network

  frontend:
    build: ./frontend
    container_name: aigc_stocks_frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    networks:
      - aigc_stocks_network

  db:
    image: postgres:14-alpine
    container_name: aigc_stocks_db
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      - POSTGRES_DB=aigc_stocks
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    networks:
      - aigc_stocks_network

volumes:
  postgres_data:

networks:
  aigc_stocks_network:
    driver: bridge
```

2. 启动服务：

```bash
docker-compose up -d
```

3. 访问应用：

   前端：`http://localhost`
   后端API：`http://localhost:8000/api`

### 7.2 生产环境部署建议

- 使用Gunicorn/uWSGI替代Django开发服务器
- 配置Nginx作为反向代理
- 使用PostgreSQL或MySQL作为生产数据库
- 配置HTTPS安全访问
- 定期备份数据库和重要文件
- 配置监控和日志收集系统

## 8. 后续规划

### 8.1 功能扩展
- [ ] 集成鉴权、用户体系
- [ ] 股票数据抓取与缓存
- [ ] 多数据源集成和数据清洗
- [ ] 高级技术指标和量化分析
- [ ] 个性化推荐系统
- [ ] 社交分享功能

### 8.2 技术优化
- [ ] 模型优化和本地化部署
- [ ] 性能优化和系统扩展
- [ ] 微服务架构重构
- [ ] 实时数据处理能力
- [ ] 多语言支持
- [ ] 移动应用开发

## 9. 文档与资源

- **架构设计文档**：详细描述系统架构和模块设计
- **技术实现细节**：提供具体的代码实现示例
- **API文档**：详细的接口说明和使用示例
- **开发指南**：帮助新开发者快速上手项目

## 10. 贡献指南

欢迎对项目进行贡献！贡献前请先阅读以下指南：

1. Fork项目仓库
2. 创建特性分支
3. 提交代码更改
4. 推送到远程仓库
5. 创建Pull Request

## 11. 免责声明

本平台提供的股票分析和投资建议仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。用户应根据自己的判断做出投资决策，并自行承担投资风险。

