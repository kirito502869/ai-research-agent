from pathlib import Path
import os

import pandas as pd

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent

from langchain_experimental.tools import PythonAstREPLTool


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


# ==================================================
# 1. Paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "data" / "test_data.csv"
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"


# ==================================================
# 2. Model
# ==================================================

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    base_url=OLLAMA_BASE_URL,
    temperature=0
)


# ==================================================
# 3. Load CSV
# ==================================================

df = pd.read_csv(CSV_PATH)


# ==================================================
# 4. Python Data Analysis Tool
# ==================================================

python_tool = PythonAstREPLTool(
    locals={
        "df": df,
        "pd": pd
    }
)

python_tool.name = "DataAnalysis"

python_tool.description = """
Use this tool whenever the user asks questions about the CSV dataset.

The pandas DataFrame is already loaded as `df`.

You may write and execute Python/Pandas code to:
- inspect the dataset
- calculate statistics
- analyze trends
- calculate correlations
- find maximum/minimum values
- detect unusual values
- compare rows or columns
- perform other reasonable data analysis

Always calculate using the actual dataframe instead of guessing.

When executing code, make sure the final result is printed or returned.
"""


# ==================================================
# 5. Paper RAG Tool
# ==================================================

embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"),
    base_url=OLLAMA_BASE_URL
)


vector_store = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 12,
        "lambda_mult": 0.7
    }
)


def paper_search(question: str) -> str:
    """
    Search the PDF knowledge base for relevant information.
    """

    docs = retriever.invoke(question)

    results = []

    for doc in docs:

        page = doc.metadata.get("page", 0) + 1

        results.append(
            f"""
Source: PDF page {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(results)


# create_agent accepts normal callable tools
paper_search.name = "PaperSearch"
paper_search.description = """
Search the paper knowledge base.

Use this when the user asks about:
- the paper
- research questions
- methodology
- experiment design
- findings
- conclusions
- theoretical explanations
"""


# ==================================================
# 6. Agent
# ==================================================

tools = [
    python_tool,
    paper_search
]


agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
You are a research and data analysis assistant.

You have access to two tools:

1. DataAnalysis
   A Python/Pandas execution environment.
   The CSV dataset is already available as a dataframe named `df`.

2. PaperSearch
   Searches the PDF knowledge base.

Rules:

- For any question involving the dataset, use DataAnalysis.
- Do not estimate or guess numerical values. Calculate them with DataAnalysis.
- Always execute the code and use its return value. Never answer with a code
  snippet, a pandas expression, or a placeholder — report the actual numbers.
- If one question asks for several figures, run one piece of code that computes
  all of them at once, then report every figure.
- If code raises an exception, read the error, fix the expression, and retry.
- You decide which Python/Pandas operations are appropriate.
- For questions involving the paper, use PaperSearch.
- If a question requires both dataset analysis and paper information,
  use both tools before answering.
- Base the final answer on actual tool results.
- Do not invent statistics, references, or findings.
"""
)


# ==================================================
# 7. Function used by CLI / FastAPI
# ==================================================

def ask_agent(question: str) -> str:

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    return response["messages"][-1].content