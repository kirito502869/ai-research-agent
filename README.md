# AI Research Agent

一个**完全本地运行**的科研助理 Agent：能读论文 PDF（RAG 检索），也能直接对 CSV 实验数据做统计分析，由本地大模型自主判断该用哪个工具。

全部推理在本地完成，不依赖任何云端 API。

```
用户提问 → Agent 判断意图 → 需要算数据？→ DataAnalysis（Python/Pandas 沙箱）
                          → 需要查论文？→ PaperSearch（ChromaDB 向量检索）
                          → 两者都要？   → 依次调用，综合作答
```

## 为什么写这个项目

科研场景下最常见的两个动作是「翻论文的某个结论」和「看看这组数据什么趋势」。它们通常需要切换不同工具完成 —— 一个 PDF 阅读器，一个 Jupyter。

这个项目把它们收敛成一个对话入口，并且让本地小模型（Qwen3 8B，消费级 Mac 可跑）来充当路由器，而不是写死 if-else 流程。

## 核心特性

- **工具自主路由**：给模型两个工具 + 一套明确的调用规则，由 `create_agent` 自主决定调用哪个、是否需要连续调用
- **RAG 检索**：PyPDFLoader 解析 → RecursiveCharacterTextSplitter 切分（800/120）→ bge-m3 嵌入 → ChromaDB 持久化
- **MMR 检索**：`search_type="mmr"`，k=4 / fetch_k=12 / λ=0.7，缓解相似片段冗余
- **全文总结路由**：识别「这篇论文讲了什么」类问题，绕过检索直接取首尾页组装摘要上下文（纯检索在这种问题上会漏信息）
- **数据分析沙箱**：PythonAstREPLTool 注入已加载好的 `df`，模型写 Pandas 代码即时执行，**数值必须真实计算，禁止编造**
- **三种交互入口**：命令行 / REST API / Streamlit 聊天界面

## 技术栈

| 层 | 选型 |
|---|---|
| Agent 编排 | LangChain 1.x `create_agent` |
| 本地 LLM | Ollama + Qwen3 8B（`temperature=0` 保证确定性） |
| 向量检索 | ChromaDB 1.5 + bge-m3 嵌入模型 |
| API | FastAPI |
| 前端 Demo | Streamlit |
| 数据分析 | Pandas + PythonAstREPLTool |

## 目录结构

```
AI-Agent-Project/
├── app.py                  # Streamlit 聊天界面
├── src/
│   ├── agent_core.py       # ★ 主入口：DataAnalysis + PaperSearch 双工具 Agent
│   ├── rag.py              # PDF  ingestion + RAG 问答（含总结型问题路由）
│   ├── research_agent.py   # 数据分析 Agent 的另一种实现
│   ├── api.py              # FastAPI 服务（POST /ask）
│   ├── agent.py            # 最小 Ollama 连通性测试
│   └── main.py             # 占位
└── data/
    ├── test_data.csv       # 示例时序数据（day, value）
    ├── documents/          # 放你要检索的 PDF
    └── chroma_db/          # 向量库（自动生成，不入库）
```

## 环境准备

项目用 [Ollama](https://ollama.com) 跑本地模型，先装 Ollama 并拉取所需模型：

```bash
ollama pull qwen3:8b      # Agent 主模型
ollama pull bge-m3        # 中文嵌入模型
```

Python 依赖（3.11 验证通过）：

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

### 1. 构建论文知识库

把自己的 PDF 放到 `data/documents/paper.pdf`，然后：

```bash
python src/rag.py
```

首次运行会打印切分页数与文本块数量并建立向量库，之后重复运行会直接加载已有库。

### 2. 命令行对话（RAG）

```bash
python src/rag.py
```

支持总结型问题：「这篇论文主要讲了什么」会自动走全文摘要模式。

### 3. 启动 API + 聊天界面

```bash
# 终端 1
uvicorn src.api:app --reload

# 终端 2
streamlit run app.py
```

### REST API

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "实验数据的变化趋势是什么？"}'
```

```json
{
  "question": "实验数据的变化趋势是什么？",
  "answer": "value 从第 1 天的 10 上升至第 6 天的 35（由 DataAnalysis 实际计算）"
}
```

交互式文档：<http://127.0.0.1:8000/docs>

## 设计取舍

| 决策 | 理由 |
|---|---|
| 用本地 Qwen3 而非云端 API | 论文数据不出本机；推理成本为零 |
| `temperature=0` | 工具调用需要稳定复现，采样抖动会让 Agent 行为不可预测 |
| 数据分析用 AST REPL 而非 `create_pandas_dataframe_agent` | 用 locals 精确注入 `df`，避免 Agent 读到整个进程命名空间 |
| 给工具写详细 description | 小模型的路由准确率高度依赖工具描述质量，这是本项目调得最多的地方 |
| MMR 而非纯相似度 | 论文章节重复度高，纯相似度 top-k 常常返回同一节的多段文字 |

## 已知限制

- `PythonAstREPLTool` 会执行模型生成的代码，生产环境必须放进沙箱容器
- PDF 解析对双栏排版论文支持有限，表格内容易丢失
- 当前无对话历史记忆，多轮追问依赖单次上下文
- 8B 模型在复杂多步推理上偶有中断，换更大模型可改善

## License

MIT
