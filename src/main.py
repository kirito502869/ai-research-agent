from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:4b",
)

response = llm.invoke(
    "请用简单的话告诉我：什么是RAG？"
)

print(response.content)