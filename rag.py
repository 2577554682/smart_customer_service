import logging

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from env_utils import EMBED_MODEL_PATH, RERANKER_MODEL_PATH

logger = logging.getLogger(__name__)

_retriever = None
_reranker = None


def _get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    if not RERANKER_MODEL_PATH:
        logger.warning("未配置 RERANKER_MODEL_PATH，跳过重排序")
        return None
    try:
        _reranker = CrossEncoder(RERANKER_MODEL_PATH, max_length=512, device=_get_device())
        logger.info("Reranker 加载完成")
    except Exception as e:
        logger.error("Reranker 加载失败：%s，降级为无重排序", e)
        _reranker = False
    return _reranker if _reranker is not False else None


def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    logger.info("正在初始化 RAG（加载文档 + 向量化）...")
    loader = UnstructuredLoader(
        file_path="documents/product_guide.txt",
        mode="elements"
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。"],
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(docs)

    embed_device = _get_device()
    logger.info("Embedding 使用设备：%s", embed_device)
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_PATH,
        model_kwargs={'device': embed_device}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory="./chroma_db"
    )
    logger.info("RAG 初始化完成，共 %d 个文档块", vectorstore._collection.count())
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return _retriever
