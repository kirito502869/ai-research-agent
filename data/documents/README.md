# 把你想让 Agent 检索的 PDF 放这里

文件名保持为 `paper.pdf`，然后运行：

```bash
python src/rag.py
```

首次运行会自动建立向量库到 `data/chroma_db/`，之后直接复用。

> 仓库里没有附带任何 PDF：示例用的是他人论文，出于版权考虑不予公开。
> 换成本人撰写的文档或开放授权的论文即可。
