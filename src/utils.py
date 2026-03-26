import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# 设置环境变量，强制使用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载环境变量
load_dotenv()


# --- 1. 获取 LLM 实例 (通过 API) ---
def get_llm():
    """
    初始化并返回大模型实例
    """
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    model_name = os.getenv("MODEL_NAME")

    if not api_key:
        raise ValueError("❌ 错误：未在 .env 中找到 API_KEY，请检查配置。")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.7,
    )


# --- 2. 获取向量模型 (Embedding) ---
def get_embeddings():
    """
    初始化 Embedding 模型。
    注意：第一次运行会从 HuggingFace 下载模型，请保持网络通畅。
    """
    model_name = "shibing624/text2vec-base-chinese"
    # 使用 CPU 运行（对本科生电脑最友好，不需要显卡）
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


# --- 3. 文献处理逻辑 ---
def process_document(file_path):
    """
    核心流程：加载 -> 切分 -> 向量化 -> 存储
    """
    # 3.1 加载文件
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        print("⚠️ 不支持的文件格式")
        return None

    documents = loader.load()

    # 优化切分参数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 增加块大小，从 500 提高到 800
        chunk_overlap=150,  # 增加重叠度，从 50 提高到 150，确保上下文衔接
        length_function=len,
        # 增加中文标点作为切分符，防止在句子中间切断
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    # 3.2 文本切分 (Chunking)
    # 中医文献通常语义紧密，建议 chunk_size 不要太小
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个文本块 500 字
        chunk_overlap=50,  # 块与块之间重叠 50 字，保证语境连续
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # 3.3 创建并保存向量库
    embeddings = get_embeddings()
    # 将切分后的文本转化为向量并存入 FAISS
    vector_db = FAISS.from_documents(texts, embeddings)

    # 持久化存储，防止重启后丢失
    vector_db.save_local("vector_store/db_tcm")
    return vector_db


# --- 4. 获取检索器 ---
def get_retriever():
    if not os.path.exists("vector_store/db_tcm"):
        return None

    embeddings = get_embeddings()
    vector_db = FAISS.load_local(
        "vector_store/db_tcm",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # k=4 是一个黄金参数，既保证了信息量，又不会因为太长让 AI 走神
    return vector_db.as_retriever(search_kwargs={"k": 4})



