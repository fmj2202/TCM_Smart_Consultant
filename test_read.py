from src.utils import get_retriever

retriever = get_retriever()
if retriever:
    # 强制进行纯关键词检索测试
    results = retriever.invoke("马尾神经瘤")
    print(f"共找到 {len(results)} 个片段")
    for i, res in enumerate(results):
        print(f"--- 片段 {i+1} 内容预览 ---")
        print(res.page_content[:200]) # 打印前200字
else:
    print("向量库不存在，请先上传文档")
