# 🌿 四诊合参 — 基于 LLM 的中医智能问诊系统

本项目是一款结合大语言模型（LLM）与传统中医理论的智能辅助系统。

## ✨ 核心功能
- **🩺 智能问诊**：基于 LangChain 实现多轮对话，模拟中医“望闻问切”辨证逻辑。
- **📚 文献检索 (RAG)**：通过 FAISS 向量数据库检索本地中医典籍，解决 LLM 幻觉问题。
- **🤖 中医 Agent**：自主调用识别工具，实现中药材图像识别与药性分析。

## 🛠️ 技术栈
- **LLM**: DeepSeek / OpenAI API
- **框架**: LangChain, Streamlit
- **向量库**: FAISS
- **工具**: Python, Dotenv

## 🚀 快速开始
1. 克隆项目：`git clone [你的仓库地址]`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置 `.env` 文件，填入你的 `API_KEY`
4. 运行：`streamlit run app.py`
