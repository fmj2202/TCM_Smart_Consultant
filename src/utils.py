import os
from pathlib import Path
import shutil
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================== 加载环境变量 ====================
load_dotenv()


# ==================== 1. get_llm ====================
def get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        temperature=0.3,
        max_tokens=2048,
        streaming=False
    )


# ==================== 2. 中文 embedding 模型 ====================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)


# ==================== 3. 知识库路径管理 ====================
def get_kb_path(kb_name: str) -> Path:
    return Path("data/kbs").absolute() / kb_name


def list_knowledge_bases() -> list:
    kb_root = Path("data/kbs").absolute()
    kb_root.mkdir(parents=True, exist_ok=True)
    return [d.name for d in kb_root.iterdir() if d.is_dir()]


def create_knowledge_base(kb_name: str):
    path = get_kb_path(kb_name)
    path.mkdir(parents=True, exist_ok=True)


def process_documents(kb_name: str, file_paths: list):
    """文档处理"""
    kb_path = get_kb_path(kb_name)
    kb_path.mkdir(parents=True, exist_ok=True)

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=200)

    for file_path in file_paths:
        if str(file_path).lower().endswith('.pdf'):
            loader = PyPDFLoader(str(file_path))
        elif str(file_path).lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(str(file_path))
        else:
            continue
        docs = loader.load()
        all_docs.extend(docs)

    splits = text_splitter.split_documents(all_docs)

    vectorstore_path = kb_path / "faiss_index"
    vectorstore_path.mkdir(parents=True, exist_ok=True)

    index_faiss = vectorstore_path / "index.faiss"
    index_pkl = vectorstore_path / "index.pkl"

    if index_faiss.exists() and index_pkl.exists():
        vectorstore = FAISS.load_local(
            str(vectorstore_path), embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(splits)
    else:
        vectorstore = FAISS.from_documents(splits, embeddings)

    vectorstore.save_local(str(vectorstore_path))
    print(f"✅ 知识库 '{kb_name}' 索引保存成功")


# ==================== 4. 知识库文档管理（系统设置专用） ====================
def list_documents_in_kb(kb_name: str) -> list:
    """列出知识库中所有上传的原始文档"""
    kb_path = get_kb_path(kb_name)
    if not kb_path.exists():
        return []
    docs = []
    for f in kb_path.iterdir():
        if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".doc"]:
            docs.append(f.name)
    return sorted(docs)


def delete_document(kb_name: str, filename: str):
    """删除单个文档 + 彻底避免维度错误"""
    kb_path = get_kb_path(kb_name)
    file_path = kb_path / filename

    if not file_path.exists():
        print(f"⚠️ 文件 {filename} 不存在")
        return

    file_path.unlink()
    print(f"🗑️ 已删除文档：{filename}")

    # 先删除旧索引，避免维度不匹配
    vectorstore_path = kb_path / "faiss_index"
    if vectorstore_path.exists():
        shutil.rmtree(vectorstore_path)
        print("🗑️ 已清除旧 FAISS 索引")

    # 剩余文档重建索引
    remaining_files = [str(kb_path / f) for f in list_documents_in_kb(kb_name)]
    if remaining_files:
        process_documents(kb_name, remaining_files)
        print(f"✅ 知识库 '{kb_name}' 索引已重建（剩余 {len(remaining_files)} 个文档）")
    else:
        print(f"✅ 知识库 '{kb_name}' 已清空")


def delete_knowledge_base(kb_name: str):
    """删除整个知识库"""
    kb_path = get_kb_path(kb_name)
    if kb_path.exists():
        shutil.rmtree(kb_path)
        print(f"🗑️ 已完全删除知识库 '{kb_name}'")
    else:
        print(f"⚠️ 知识库 '{kb_name}' 不存在")


# ==================== 5. 检索器（文献检索页面必需） ====================
def get_retriever(kb_name: str, k: int = 12):
    """获取指定知识库的检索器"""
    vectorstore_path = get_kb_path(kb_name) / "faiss_index"
    index_faiss = vectorstore_path / "index.faiss"
    index_pkl = vectorstore_path / "index.pkl"

    if not (vectorstore_path.exists() and index_faiss.exists() and index_pkl.exists()):
        return None

    vectorstore = FAISS.load_local(
        str(vectorstore_path), embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
