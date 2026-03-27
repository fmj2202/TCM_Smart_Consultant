🌿 **四诊合参** — 基于 LLM 的中医智能辅助系统

**项目简介**  
一款结合大语言模型与传统中医理论的智能辅助平台，支持**规范化四诊问诊**、**多知识库 RAG 文献检索**以及**视觉智能体药材识别**，为用户提供专业、可信的中医辅助参考。

---

### ✨ 核心功能

**🩺 智能问诊**  
- 严格模拟中医“**望闻问切**”四诊流程  
- AI 每次仅引导一个环节，回复开头标注 **【当前进度】**，严禁提前下结论  
- 内置实时问诊进度条 + 多轮对话记忆（ConversationBufferMemory）  
- 四诊完整后自动给出辨证结论与食疗/生活调理建议

**📚 文献检索（RAG）**  
- 支持**多知识库**管理（创建、删除、切换）  
- 批量上传 PDF/Docx 文献，自动切片 + FAISS 向量索引构建  
- 检索时精准引用原始文献片段，有效减少 LLM 幻觉  
- 系统设置页可删除文档/整个知识库，索引自动重建

**🤖 中医 Agent（药材识别）**  
- 上传任意中药材图片（文件名乱码也无影响）  
- 自动调用 **VLM 多模态模型** 进行像素级识别  
- 输出强制格式：**第一句必须是“识别结果为：XXX”**，后续给出性味归经、功效、主治及食疗方  
- 后台可见完整 **ReAct 推理链**（Thought → Action → Observation）

**⚙️ 系统设置**  
- 查看当前模型信息  
- 清空对话历史  
- 知识库文档管理（删除单文档/整个知识库）

---

### 🛠️ 技术栈

- **语言与框架**：Python + Streamlit（前端界面）  
- **核心编排**：LangChain（Chain / Agent / PromptTemplate / Memory / ReAct）  
- **大模型**：DeepSeek-V3（SiliconFlow API）  
- **多模态视觉**：VLM（`internlm/internlm-x2.5-7b-chat`）  
- **RAG 检索**：FAISS 向量库 + HuggingFaceEmbeddings（`BAAI/bge-small-zh-v1.5`）  
- **文档处理**：PyPDFLoader + Docx2txtLoader + RecursiveCharacterTextSplitter  
- **配置管理**：python-dotenv + GitHub 版本控制  

---

### 🚀 快速开始

1. **克隆项目**  
   ```bash
   git clone https://github.com/fmj2202/TCM_Smart_Consultant.git
   cd TCM_Smart_Consultant
   
2.**安装依赖**
   pip install -r requirements.txt
   
3.**配置环境变量**
   API_KEY=sk-你的密钥
   BASE_URL=https://api.siliconflow.cn/v1
   MODEL_NAME=deepseek-ai/DeepSeek-V3
   VISION_MODEL=internlm/internlm-x2.5-7b-chat    # 视觉模型
   
4.**运行项目**
   streamlit run app.py
