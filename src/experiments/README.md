# 演进实验：为什么最后的实现长这样

这三个文件按时间顺序记录了「数据分析 Agent」的三次迭代。**最终采用的版本是 `src/agent_core.py`**，下面的旧版本保留在这里是为了说明演进过程，不是生产代码。

| 版本 | 文件 | 做法 | 为什么被替换 |
|---|---|---|---|
| v1 | `v1_pandas_dataframe_agent.py` | LangChain 现成的 `create_pandas_dataframe_agent`，配 `allow_dangerous_code=True` | 开发的黑盒不可控：不知道它拼接了什么 prompt，出错时无从调试；Agent 会把整个 DataFrame 塞进上下文，长数据直接爆 token |
| v2 | `v2_handwritten_tool.py` | 用 `@tool` 手写工具，自己解析关键词（"平均"/"最大"/"趋势"）匹配到预置的 Pandas 表达式 | 关键词匹配覆盖不了自然语言问法，换个说法就失效；每加一种问法就要改代码，本质是 if-else 伪装成 Agent |
| v3 ✅ | `src/agent_core.py` | `create_agent` + `PythonAstREPLTool`，用 `locals` 精确注入 `df`，让模型自己写 Pandas 代码执行 | 泛化能力来自模型而非规则；`df` 通过 locals 注入，Agent 拿不到进程命名空间，比 `exec` 全局执行安全 |

## 核心改进点

**1. 用 locals 注入代替危险执行**

```python
python_tool = PythonAstREPLTool(locals={"df": df, "pd": pd})
```

对比 v1 的 `allow_dangerous_code=True` 全量注入，这里 Agent 能访问的只有 `df` 和 `pd` 两个名字。

**2. 工具描述决定路由质量**

小模型（8B）判断要不要用工具几乎完全依赖 description 的措辞。message v3 里给 `DataAnalysis` 写了一整段能力清单 + 运算列表，并在 system prompt 里反复强调「不许估算」—— 这是三次迭代里调得最多的地方。

**3. 双工具而不是单工具**

数据分析和论文检索分开注册，让模型在一次对话里可以先后调用两个工具再综合回答，而不是走预设流程。
