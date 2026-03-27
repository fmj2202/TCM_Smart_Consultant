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
    st.header("🩺 四诊合参——规范化问诊")

    # --- 简历亮点：展示问诊进度 ---
    # 我们通过分析历史消息的数量或内容，简单模拟一个进度条
    # 也可以手动在侧边栏显示当前处于哪一环节
    progress_map = {"望": 25, "闻": 50, "问": 75, "切": 100}

    # 默认进度
    current_step_val = 0

    # 简单逻辑：根据对话轮数动态增加进度（实际项目中可以根据AI输出的【当前进度】来匹配）
    chat_len = len(st.session_state.messages)
    if chat_len < 2:
        current_step_val = 10
    elif chat_len < 4:
        current_step_val = 30
    elif chat_len < 6:
        current_step_val = 60
    elif chat_len < 8:
        current_step_val = 90
    else:
        current_step_val = 100

    st.write(f"问诊完成度：{current_step_val}%")
    st.progress(current_step_val)
    st.markdown("---")

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

    # 导入优化后的工具函数
    from src.utils import process_documents, get_retriever, list_knowledge_bases, create_knowledge_base
    from src.chains.rag_chain import get_rag_chain

    # ====================== 知识库管理 ======================
    st.subheader("📁 知识库管理")
    kbs = list_knowledge_bases()
    if not kbs:
        st.warning("暂无知识库，请先创建。")
        kbs = ["默认知识库"]  # 首次使用时显示占位

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_kb = st.selectbox(
            "选择要使用的知识库",
            kbs,
            key="kb_select"
        )
    with col2:
        new_kb_name = st.text_input("新建知识库名称", placeholder="例如：经典古籍")
        if st.button("➕ 创建知识库") and new_kb_name.strip():
            create_knowledge_base(new_kb_name.strip())
            st.success(f"知识库 '{new_kb_name}' 创建成功！")
            st.rerun()

    st.markdown("---")

    # ====================== 文档上传（支持批量） ======================
    st.subheader("📤 上传文档到当前知识库")
    uploaded_files = st.file_uploader(
        "上传中医文献 (PDF / Docx，支持多个文件)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="一次可上传多个文件，全部会存入当前选中的知识库"
    )

    if uploaded_files and st.button("✅ 确认上传并处理"):
        # 保存原始文件到知识库目录
        kb_path = f"data/kbs/{selected_kb}"
        os.makedirs(kb_path, exist_ok=True)

        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(kb_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)

        with st.spinner(f"正在解析 {len(uploaded_files)} 个文档并构建向量索引（知识库：{selected_kb}）..."):
            process_documents(selected_kb, saved_paths)
            st.success(
                f"✅ 已成功将 {len(uploaded_files)} 个文档添加到知识库 '{selected_kb}'！文档已永久保存，可随时使用。")

    st.markdown("---")

    # ====================== 文献提问 ======================
    st.subheader("🔍 基于文献提问")
    query = st.text_input("请输入您想基于文献查询的问题：", key="query_input")

    if query and selected_kb:
        retriever = get_retriever(selected_kb)
        if retriever:
            with st.spinner("🔍 正在检索文献..."):
                rag_chain = get_rag_chain(st.session_state.llm, retriever)
                response = rag_chain.invoke({"query": query})

                st.write("### 🤖 AI 的分析结果：")
                st.info(response["result"])

                # 展示引用片段（便于调试与信任）
                with st.expander("📖 原始文献参考片段"):
                    search_results = retriever.invoke(query)
                    for i, doc in enumerate(search_results):
                        st.write(f"**片段 {i + 1}**（来源：{doc.metadata.get('source', '未知')}）：")
                        st.caption(doc.page_content[:350] + "..." if len(doc.page_content) > 350 else doc.page_content)
        else:
            st.warning("当前知识库还没有任何文档，请先上传文献。")

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

    st.write(f"**当前使用的模型**：{st.session_state.llm.model_name}")

    # ====================== 1. 清空对话历史（原有功能保留） ======================
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.success("对话历史已清空！")
        st.rerun()

    st.markdown("---")

    # ====================== 2. 知识库管理（已包含删除文档 + 新增删除知识库） ======================
    st.subheader("📁 知识库文档管理")
    from src.utils import (
        list_knowledge_bases,
        list_documents_in_kb,
        delete_document,
        delete_knowledge_base,
    )

    kbs = list_knowledge_bases()
    if not kbs:
        st.info("暂无知识库，请先去「文献检索」页面创建并上传文档。")
    else:
        for kb_name in kbs:
            with st.expander(f"📚 知识库：**{kb_name}**", expanded=False):
                # 删除整个知识库按钮（新增功能）
                if st.button("🗑️ 删除整个知识库", key=f"del_kb_{kb_name}", type="secondary"):
                    with st.spinner(f"正在删除知识库 '{kb_name}' ..."):
                        delete_knowledge_base(kb_name)
                        st.success(f"✅ 知识库 '{kb_name}' 已完全删除！")
                        st.rerun()

                st.markdown("---")

                # 文档列表
                docs = list_documents_in_kb(kb_name)
                if not docs:
                    st.caption("该知识库暂无文档")
                else:
                    st.caption(f"共 {len(docs)} 个文档")
                    for doc in docs:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"📄 {doc}")
                        with col2:
                            if st.button("🗑️ 删除文档", key=f"del_doc_{kb_name}_{doc}"):
                                with st.spinner(f"正在删除 {doc} 并更新索引..."):
                                    delete_document(kb_name, doc)
                                    st.success(f"✅ 文档 {doc} 已删除！索引已自动重建。")
                                    st.rerun()