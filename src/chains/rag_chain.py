from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

rag_template = """你是一名专业的中医助手。请根据提供的【参考资料】回答用户的问题。

【参考资料】：
{context}

【用户问题】：{question}

【回答要求】：
1. 语言要专业、严谨，且条理清晰（可以使用序号）。
2. 必须严格基于【参考资料】的内容。如果资料中没有相关信息，请回答“参考资料中未提及相关内容”。
3. 不要强行套用疾病诊断格式，根据问题类型灵活回答。

专业回答："""


RAG_PROMPT = PromptTemplate(template=rag_template, input_variables=["context", "question"])

def get_rag_chain(llm, retriever):
    # 这里我们直接构建 QA 链
    # 技巧：在大型文档中，我们将 chain_type 设为 "stuff"
    # 但由于我们在 utils 里把 k 设为了 20，我们需要确保 llm 能够处理这么多上下文
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return chain
