import os
import streamlit as st
from dotenv import load_dotenv
from src.utils import get_llm
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from src.chains.chat_chain import get_chat_chain

# 1. 初始化配置
load_dotenv()

st.set_page_config(
    page_title="四诊合参 - 中医智能问诊系统",
    page_icon="🌿",
    layout="wide"
)


# 2. 初始化全局状态（确保对话不丢失）
if "llm" not in st.session_state:
    try:
        st.session_state.llm = get_llm()
    except Exception as e:
        st.error(f"模型初始化失败，请检查 .env 文件中的 API Key。错误信息: {e}")
        st.stop()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "conversation" not in st.session_state:
    st.session_state.conversation = get_chat_chain(
        st.session_state.llm,
        st.session_state.memory
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 侧边栏导航
with st.sidebar:
    st.title("🌿 四诊合参")
    st.markdown("---")
    menu = st.selectbox("功能导航", ["智能问诊", "文献检索", "中医Agent", "系统设置"])
    st.info("💡 提示：本系统结合 LLM 为用户提供中医辅助参考。")

# 4. 页面逻辑切换
if menu == "智能问诊":
    st.header("🩺 中医智能问诊小助手")
    st.write("请描述您的症状，我将为您提供初步的辨证分析。")

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入逻辑
    if prompt := st.chat_input("描述您的症状..."):
        # 存入并显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 AI 获取响应
        with st.chat_message("assistant"):
            with st.spinner("中医专家正在审方辨证..."):
                response = st.session_state.conversation.predict(input=prompt)
                st.markdown(response)

        # 存入助手响应
        st.session_state.messages.append({"role": "assistant", "content": response})

elif menu == "文献检索":
    st.header("📚 中医药文献知识库")

    # 导入刚刚写的工具函数
    from src.utils import process_document, get_retriever
    from src.chains.rag_chain import get_rag_chain

    # 1. 文件上传
    uploaded_file = st.file_uploader("上传中医文献 (PDF/Docx)", type=["pdf", "docx"])

    if uploaded_file:
        # 保存文件到临时目录
        with open(f"data/docs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("正在解析文献并构建向量数据库..."):
            process_document(f"data/docs/{uploaded_file.name}")
            st.success("文献加载完成！AI 现在已经学习了该内容。")

    st.markdown("---")

    # 2. 检索问答
    query = st.text_input("请输入您想基于文献查询的问题：")
    if query:
        retriever = get_retriever()
        if retriever:
            with st.spinner("🔍 正在检索文献..."):
                # 直接使用用户的原始问题 query，不要再加“疾病名称”前缀
                from src.chains.rag_chain import get_rag_chain

                rag_chain = get_rag_chain(st.session_state.llm, retriever)

                # 直接传 query
                response = rag_chain.invoke({"query": query})

                st.write("### 🤖 AI 的分析结果：")
                st.info(response["result"])

                # 保留这个调试功能，让你看清 AI 引用了哪段话
                with st.expander("🛠️ 原始文献参考"):
                    search_results = retriever.invoke(query)
                    for i, doc in enumerate(search_results):
                        st.write(f"**片段 {i + 1}：** {doc.page_content[:300]}...")

elif menu == "中医Agent":
    st.header("🤖 中医智能 Agent — 药材识别")

    # 1. 初始化工具和 Agent (保持逻辑不变)
    from src.tools.herb_tool import tcm_herb_recognition

    tools = [tcm_herb_recognition]
    agent = initialize_agent(
        tools,
        st.session_state.llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # 2. 图片上传组件 (替换掉原来的 text_input)
    uploaded_file = st.file_uploader("请上传中药材图片进行识别", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 在界面上展示上传的图片
        st.image(uploaded_file, caption="已上传的药材图片", width=300)

        # 3. 将上传的文件保存到临时目录，以便工具读取路径
        # 我们创建一个临时文件夹存储上传的内容
        temp_dir = "data/temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 4. 识别按钮
        if st.button("开始识别并分析"):
            with st.spinner("Agent 正在规划任务并调用视觉工具..."):
                try:
                    # 构造任务，告诉 Agent 路径
                    task = f"请帮我识别这张图片里的药材：{temp_file_path}。识别出结果后，请以中医专家的身份，详细解释该药材的药性以及推荐一个对应的食疗方。"

                    # 运行 Agent
                    response = agent.run(task)

                    st.write("### 🧠 Agent 推理分析结果：")
                    st.success(response)
                    st.info("💡 提示：查看 PyCharm 终端可观察 Agent 的 Thought/Action/Observation 思考链。")
                except Exception as e:
                    st.error(f"识别过程中出现错误：{e}")


elif menu == "系统设置":
    st.header("⚙️ 系统配置")
    st.write(f"当前使用的模型: {st.session_state.llm.model_name}")
    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
