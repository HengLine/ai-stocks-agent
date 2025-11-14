# 股票分析智能体

AI股票分析代理系统是一个基于大语言模型的智能股票分析平台，通过多代理协作机制，从技术面、基本面、情绪面、行业宏观、ESG风险、资金流向等多个维度对股票进行全面分析，为投资者提供专业、客观的投资决策参考。

## 功能特性

### 多维度分析
- **技术分析代理**：基于K线图、技术指标等进行趋势判断和买卖点识别
- **基本面分析代理**：分析公司财务数据、盈利能力、估值水平等
- **情绪分析代理**：监测市场情绪、社交媒体讨论热度等
- **行业宏观代理**：分析行业发展趋势、宏观经济环境影响
- **ESG风险代理**：评估企业环境、社会和治理风险
- **资金流向代理**：追踪机构资金、散户资金动向
- **首席策略代理**：综合各维度分析结果，提供最终投资策略建议

### 可视化展示
- 交互式K线图表，支持技术指标叠加
- 成交量、RSI等技术指标可视化
- 分析报告自动生成和展示

### 灵活配置
- 支持多种大语言模型后端（OpenAI、DeepSeek、Qwen、Ollama等）
- 可定制提示词模板和分析参数
- 支持知识库扩展，增强分析深度

## 系统架构

### 核心模块
- **代理协调器**：管理多代理协作流程
- **基础代理类**：定义代理通用接口和行为
- **专项分析代理**：实现各维度分析功能
- **客户端接口**：与各种LLM服务对接
- **股票数据源**：提供股票历史数据和实时行情
- **知识库**：存储投资知识和分析方法

### 技术栈
- Python
- Streamlit (前端可视化)
- 大语言模型API
- Pandas (数据处理)
- Plotly (图表绘制)

## 安装指南

### 前提条件
- Python 3.8+
- pip
- 相应的API密钥（如OpenAI、DeepSeek等）

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/HengLine/ai-stocks-agent
cd ai-stocks-agent
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
复制`.env.example`文件为`.env`，并填写相应的API密钥：
```bash
cp .env.example .env
# 编辑.env文件，填入你的API密钥
```

## 快速使用

### 1. 启动应用

```bash
# 使用批处理脚本启动
start.bat

# 或者直接运行
python .\main.py --dashboard
```

### 2. 使用API接口

```python
from api.stock_agent_api import StockAgentAPI

api = StockAgentAPI()
result = api.analyze_stock(stock_code="600000", analysis_type="comprehensive")
print(result)
```

## 注意事项

1. **API密钥安全**：请确保API密钥不被泄露，不要提交到版本控制系统
2. **模型选择**：不同的LLM模型在分析质量和成本上有所差异，请根据需求选择合适的模型
3. **数据延迟**：股票数据可能存在延迟，实时交易决策请谨慎参考
4. **免责声明**：系统分析结果仅供参考，不构成投资建议，请投资者自行承担风险
5. **资源消耗**：使用大型语言模型可能产生较高的API调用费用，请合理控制分析频率

## 其他说明

### 自定义配置

编辑`config/config.json`文件可以调整系统配置：
- 代理参数配置
- 模型选择和参数
- 分析阈值设置

### 扩展知识库

在`knowledge_base/`目录下添加或修改知识文档，可以增强系统的分析能力：
- 投资心理学知识
- 股票基础知识
- 风险管理策略
- 技术分析方法

### 开发指南

如果您想参与开发或自定义功能，请参考以下文件：
- `hengline/agents/base_agent.py`: 了解代理基类实现
- `hengline/client/client_factory.py`: 添加新的LLM客户端
- `hengline/prompts/`: 修改或添加提示词模板



## 联系我们

如有问题或建议，请通过以下方式联系我们：
- 邮箱：[haeng2030@gmail.com](mailto:haeng2030@gmail.com)
- GitHub：https://github.com/HengLine/ai-stocks-agent