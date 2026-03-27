from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

rag_template = """你是一名专业的中医助手。请根据提供的【参考资料】回答用户的问题。

【参考资料】：
{context}

【用户问题】：{question}

【回答要求】：
1. 语言要专业、严谨，且条理清晰（可以使用序号或小标题）。
2. **必须先完整、直接回答用户的问题**：严格对应用户询问的核心内容（例如用户问“用途”，就先详细说明用途/功效/主治；问“制作方法”就先详细说明制法步骤）。
3. **补充信息放在最后**：如果参考资料中还有其他相关内容（如配方、用法、注意事项等），可以在回答**最后**以“补充信息”小标题的形式列出，不要放在主回答里。
4. 必须严格基于【参考资料】的内容。如果资料中没有相关信息，请回答“参考资料中未提及相关内容”。
5. 回答末尾请加上“以上内容严格依据参考资料整理，未作额外补充。”。

专业回答："""

RAG_PROMPT = PromptTemplate(template=rag_template, input_variables=["context", "question"])

def get_rag_chain(llm, retriever):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return chain
