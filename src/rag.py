from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ==================================================
# 1. 基础配置
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "paper.pdf"
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"


embeddings = OllamaEmbeddings(
    model="bge-m3"
)

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)


# ==================================================
# 2. 读取原始 PDF
# ==================================================

loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()


# ==================================================
# 3. 建立 / 读取向量数据库
# ==================================================

if DB_PATH.exists() and any(DB_PATH.iterdir()):

    print("发现已有向量数据库，直接加载...")

    vector_store = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )

else:

    print("第一次运行，正在建立知识库...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(documents)

    print(f"PDF：{len(documents)} 页")
    print(f"文本块：{len(chunks)} 个")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_PATH),
    )

    print("向量数据库建立完成")


# ==================================================
# 4. MMR Retriever
# ==================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 12,
        "lambda_mult": 0.7,
    },
)


# ==================================================
# 5. 判断是不是“全文总结型问题”
# ==================================================

def is_summary_question(question):

    question_lower = question.lower()

    summary_keywords = [
        "这篇文章讲了什么",
        "这篇论文讲了什么",
        "主要讲什么",
        "主要研究什么",
        "总结这篇",
        "总结论文",
        "总结文章",
        "main idea",
        "main topic",
        "what is this paper about",
        "summarize this paper",
        "summary of this paper",
    ]

    return any(
        keyword in question_lower
        for keyword in summary_keywords
    )


# ==================================================
# 6. 全文总结 Context
# ==================================================

def build_summary_context():

    selected_pages = []

    total_pages = len(documents)

    # 前几页通常包含 Abstract / Introduction
    front_pages = documents[:min(4, total_pages)]

    # 最后正文部分通常包含 Discussion / Conclusion
    back_start = max(0, total_pages - 4)
    back_pages = documents[back_start:]

    selected_pages.extend(front_pages)

    for doc in back_pages:
        if doc not in selected_pages:
            selected_pages.append(doc)

    context_parts = []

    for doc in selected_pages:

        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"[PDF 第 {page} 页]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# ==================================================
# 7. 普通 RAG Context
# ==================================================

def build_rag_context(question):

    retrieved_docs = retriever.invoke(question)

    print("\n========== RAG 检索结果 ==========\n")

    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):

        page = doc.metadata.get("page", 0) + 1

        preview = doc.page_content.replace("\n", " ")[:180]

        print(f"来源 {i} | PDF 第 {page} 页")
        print(preview + "...\n")

        context_parts.append(
            f"[来源 {i}，PDF 第 {page} 页]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# ==================================================
# 8. 连续问答
# ==================================================

print("\n===================================")
print("     Local RAG Assistant")
print("===================================")
print("输入 exit 退出\n")


while True:

    question = input("请输入问题：\n> ").strip()

    if question.lower() in ["exit", "quit"]:
        print("程序结束")
        break

    if not question:
        continue


    # ==============================================
    # Router
    # ==============================================

    if is_summary_question(question):

        print("\n[Router] → 全文总结模式")

        context = build_summary_context()

    else:

        print("\n[Router] → RAG 检索模式")

        context = build_rag_context(question)


    # ==============================================
    # 给 Qwen 的 Prompt
    # ==============================================

    prompt = f"""
你是一名严谨的学术文档分析助手。

请只根据下面提供的 PDF 内容回答问题。

规则：
1. 不要凭空编造信息。
2. 如果资料不足，请明确说明。
3. 如果能够确定来源，请注明 PDF 页码。
4. 回答应当清晰、简洁、有条理。

PDF 内容：

{context}

用户问题：

{question}
"""


    response = llm.invoke(prompt)

    print("\n========== AI 回答 ==========\n")
    print(response.content)

    print("\n" + "=" * 55 + "\n")