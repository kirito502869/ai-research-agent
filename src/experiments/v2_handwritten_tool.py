from pathlib import Path
import pandas as pd

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

from langchain.agents import create_agent


# ==================================================
# 1. 路径
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "data" / "test_data.csv"

DB_PATH = PROJECT_ROOT / "data" / "chroma_db"


# ==================================================
# 2. 大模型
# ==================================================

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)


# ==================================================
# 3. Data Analysis Tool
# ==================================================

df = pd.read_csv(CSV_PATH)


@tool
def data_analysis(question: str) -> str:
    """
    分析CSV实验数据。

    可以回答：
    - 数据趋势
    - 平均值
    - 最大值
    - 最小值
    - 描述统计
    """

    result = []

    q = question.lower()


    # 平均值

    if "平均" in question or "mean" in q:

        result.append(
            f"value平均值为：{df['value'].mean():.2f}"
        )


    # 最大值

    if "最大" in question or "max" in q:

        result.append(
            f"value最大值为：{df['value'].max()}"
        )


    # 最小值

    if "最小" in question or "min" in q:

        result.append(
            f"value最小值为：{df['value'].min()}"
        )


    # 趋势

    if "趋势" in question or "trend" in q:

        first = df["value"].iloc[0]
        last = df["value"].iloc[-1]


        if last > first:

            result.append(
                "value整体呈上升趋势。"
            )

        elif last < first:

            result.append(
                "value整体呈下降趋势。"
            )

        else:

            result.append(
                "value整体保持稳定。"
            )


    # 默认统计

    if not result:

        result.append(
            str(df.describe())
        )


    return "\n".join(result)



# ==================================================
# 4. Paper RAG Tool
# ==================================================


embeddings = OllamaEmbeddings(
    model="bge-m3"
)


vector_store = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":12,
        "lambda_mult":0.7
    }
)



@tool
def paper_search(question: str) -> str:
    """
    查询论文知识库。

    用于回答：
    - 论文研究问题
    - 方法
    - 实验设计
    - 结论
    - 理论解释
    """

    docs = retriever.invoke(question)


    result = []


    for doc in docs:

        page = doc.metadata.get(
            "page",
            0
        ) + 1


        result.append(
            f"""
PDF 第 {page} 页：

{doc.page_content}
"""
        )


    return "\n\n".join(result)



# ==================================================
# 5. 创建 Agent
# ==================================================


tools = [
    data_analysis,
    paper_search
]


agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""

你是一个科研智能分析助手。

你可以使用两个工具：

1. data_analysis:
分析实验CSV数据。

2. paper_search:
查询论文知识库。


规则：

- 如果问题涉及实验数据，调用 data_analysis。
- 如果问题涉及论文内容，调用 paper_search。
- 如果问题同时涉及数据和论文，可以调用多个工具。
- 必须基于工具返回的信息回答。
- 不要编造不存在的信息。

"""
)



# ==================================================
# 6. 对话循环
# ==================================================


print("==============================")
print(" Research Agent Started ")
print("==============================")

print("输入 exit 退出\n")


while True:


    question = input(
        "请输入问题:\n>"
    )


    if question.lower() == "exit":

        print("退出")

        break



    response = agent.invoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content":question
                }
            ]
        }
    )


    print(
        "\n==========回答==========\n"
    )


    print(
        response["messages"][-1].content
    )


    print(
        "\n========================\n"
    )