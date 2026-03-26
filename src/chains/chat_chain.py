from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# 定义中医专家的提示词模板
tcm_template = """你是一名资深的中医专家，精通传统中医的“望、闻、问、切”理论。
你现在的任务是为用户提供中医辨证分析建议。

要求：
1. 语言要专业、温和，多使用中医术语。
2. 如果用户描述不清楚，要主动询问其“舌苔颜色、睡眠情况、胃口、二便”等信息。
3. 最后的建议包括：辩证结论、食疗方（如药茶）、生活起居建议。

当前对话历史:
{history}
病人：{input}
中医专家："""

TCM_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=tcm_template
)

def get_chat_chain(llm, memory):
    """
    创建一个智能问诊的对话链
    """
    return ConversationChain(
        llm=llm,
        memory=memory,
        prompt=TCM_PROMPT,
        verbose=True
    )
