from langchain_ollama import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent
from pathlib import Path
import pandas as pd


# 加载数据

PROJECT_ROOT = Path(__file__).resolve().parent.parent

csv_path = PROJECT_ROOT / "data" / "test_data.csv"

df = pd.read_csv(csv_path)

# 本地模型

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)


# 创建数据分析Agent

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True
)


question = input(
    "请输入数据分析问题：\n>"
)


response = agent.invoke(question)


print("\n结果：")
print(response["output"])