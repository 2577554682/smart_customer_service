import logging
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from state import State
from rag import get_retriever, get_reranker
from my_llm import gpt

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 500


def _validate_input(query: str) -> str | None:
    if not query or not query.strip():
        return "请输入您的问题。"
    if len(query) > MAX_INPUT_LENGTH:
        return f"您的问题过长（{len(query)}字符），请精简到{MAX_INPUT_LENGTH}字以内。"
    return None


def _format_history(history: List[dict]) -> str:
    if not history:
        return "（无历史对话）"
    lines = []
    for i, turn in enumerate(history[-5:], 1):
        lines.append(f"第{i}轮 - 用户：{turn['user']}")
        lines.append(f"第{i}轮 - 客服：{turn['assistant']}")
    return "\n".join(lines)


# ---- 节点函数 ----

def analyze_intent(state: State) -> State:
    """节点1：分析用户意图"""
    query = state["user_query"]

    error = _validate_input(query)
    if error:
        state["intent"] = "chitchat"
        state["draft_response"] = error
        state["quality_score"] = 100
        logger.warning("输入校验失败：%s", error)
        return state

    prompt = ChatPromptTemplate.from_template("""
        你是一个客服意图分类器。分析用户输入，只输出下面四个词之一：
        - return_order：退货、退款、换货相关
        - consultation：咨询产品信息、政策、规则、使用方法
        - complaint：投诉、吐槽、表达不满或愤怒
        - chitchat：问候、闲聊、感谢、告别、无实际业务内容

        用户说：{query}

        意图：
        """)

    try:
        chain = prompt | gpt | StrOutputParser()
        intent = chain.invoke({"query": query})
        state["intent"] = intent.strip().lower()
    except Exception as e:
        logger.error("意图识别失败：%s，默认按咨询处理", e)
        state["intent"] = "consultation"

    logger.info("意图识别：%s", state["intent"])
    return state


def retrieve_documents(state: State) -> State:
    """节点2：RAG 检索 + 重排序（仅咨询/退货类需要）"""
    if state["intent"] not in ["consultation", "return_order"]:
        state["rag_context"] = ""
        logger.info("跳过检索（意图：%s）", state["intent"])
        return state

    try:
        retriever = get_retriever()
        docs = retriever.invoke(state["user_query"])
    except Exception as e:
        logger.error("检索失败：%s", e)
        state["rag_context"] = ""
        return state

    if not docs:
        state["rag_context"] = ""
        logger.warning("未检索到相关文档")
        return state

    reranker = get_reranker()
    if reranker is not None and len(docs) > 2:
        try:
            pairs = [(state["user_query"], doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            docs = [doc for doc, _ in ranked[:2]]
            logger.info("RAG 检索 %d 篇 → 重排序后取 top-%d", len(scores), len(docs))
        except Exception as e:
            logger.error("重排序失败：%s，使用原始召回结果", e)
            docs = docs[:2]
    else:
        docs = docs[:2]

    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[来源{i}] {doc.page_content}")
    state["rag_context"] = "\n\n".join(context_parts)
    logger.info("RAG 检索到 %d 个相关片段", len(docs))
    return state


def generate_response(state: State) -> State:
    """节点3：生成回复"""
    query = state["user_query"]
    history_text = _format_history(state.get("chat_history", []))

    # 输入校验失败时跳过 LLM 调用
    if state.get("draft_response") and state["intent"] == "chitchat" and state.get("quality_score") == 100:
        return state

    try:
        if state["intent"] == "chitchat":
            prompt = ChatPromptTemplate.from_template("""
                你是友好热情的客服助手。

                【最近对话】
                {history}

                【用户当前消息】
                {query}

                请友好地回复。保持热情、简短。
                回复：
                """)
            chain = prompt | gpt | StrOutputParser()
            response = chain.invoke({"query": query, "history": history_text})

        elif state["intent"] == "complaint":
            prompt = ChatPromptTemplate.from_template("""
                你是专业的客服专员。用户表达了不满。

                【最近对话】
                {history}

                【用户投诉】
                {query}

                请按以下步骤回复：
                1. 先真诚道歉
                2. 表达理解和共情
                3. 说明会立即处理
                4. 告知已转接人工客服

                回复：
                """)
            chain = prompt | gpt | StrOutputParser()
            response = chain.invoke({"query": query, "history": history_text})
            state["need_human"] = True

        else:  # consultation 或 return_order
            if not state.get("rag_context", "").strip():
                state["draft_response"] = "我查一下相关资料，稍后回复您。"
                state["need_human"] = True
                logger.info("无 RAG 上下文，标记转人工")
                return state

            prompt = ChatPromptTemplate.from_template("""
                你是专业的客服助手。根据以下【参考资料】回答用户问题。

                【最近对话】
                {history}

                【参考资料】
                {context}

                【用户问题】
                {query}

                【要求】
                - 只根据参考资料回答，不要编造信息
                - 如果资料里没有明确答案，说"我查一下，稍后回复您"
                - 引用具体来源（如"根据退货政策…"）
                - 回答简洁、专业

                【回复】
                """)
            chain = prompt | gpt | StrOutputParser()
            response = chain.invoke({
                "context": state["rag_context"],
                "query": query,
                "history": history_text
            })

        state["draft_response"] = response
        logger.info("已生成回复草稿")
    except Exception as e:
        logger.error("回复生成失败：%s", e)
        state["draft_response"] = "抱歉，系统暂时出现故障，请稍后再试或联系人工客服。"
        state["need_human"] = True

    return state


def check_quality(state: State) -> State:
    """节点4：检查回复质量"""
    if state["intent"] in ["chitchat", "complaint"]:
        state["quality_score"] = 100
        logger.info("跳过质量检查（%s）", state["intent"])
        return state

    prompt = ChatPromptTemplate.from_template("""
        评估以下客服回复的质量。请分别对三个维度打分，然后输出总分。

        用户问题：{query}
        客服回复：{response}

        评分维度（0-100）：
        - 准确性（40%）：回答是否基于事实、是否正确
        - 有用性（30%）：是否解决了用户的核心问题
        - 友好度（30%）：语气是否礼貌、专业

        请按如下格式输出（只输出数字和格式，不要其他内容）：
        准确性: XX
        有用性: XX
        友好度: XX
        总分: XX
        """)

    try:
        chain = prompt | gpt | StrOutputParser()
        result = chain.invoke({
            "query": state["user_query"],
            "response": state["draft_response"]
        })
        for line in result.strip().split("\n"):
            if "总分" in line:
                digits = ''.join(c for c in line if c.isdigit())
                if digits:
                    state["quality_score"] = min(max(int(digits), 0), 100)
                    break
        else:
            state["quality_score"] = min(max(int(''.join(c for c in result if c.isdigit()) or 70), 0), 100)
    except Exception as e:
        logger.error("质量评估失败：%s，使用默认分", e)
        state["quality_score"] = 70

    logger.info("质量评分：%d", state["quality_score"])
    return state


def improve_response(state: State) -> State:
    """节点5：改进回复"""
    state["previous_score"] = state["quality_score"]
    state["retry_count"] += 1

    prompt = ChatPromptTemplate.from_template("""
        以下客服回复质量评分为{score}分（满分100），请改进。

        用户问题：{query}
        原回复：{response}

        改进方向：
        - 更准确地回答核心问题
        - 信息更完整，但保持简洁
        - 语气更友好、专业

        改进后的回复：
        """)

    try:
        chain = prompt | gpt | StrOutputParser()
        improved = chain.invoke({
            "score": state["quality_score"],
            "query": state["user_query"],
            "response": state["draft_response"]
        })
        state["draft_response"] = improved
        logger.info("第 %d 次改进完成", state["retry_count"])
    except Exception as e:
        logger.error("改进回复失败：%s，保持原回复", e)

    return state


def finalize(state: State) -> State:
    """节点6：最终处理"""
    if state.get("need_human", False):
        state["final_response"] = f"[转人工]\n{state['draft_response']}\n\n请稍等，人工客服即将接入。"
    else:
        state["final_response"] = state["draft_response"]
    logger.info("最终回复已生成")
    return state


def save_history(state: State) -> State:
    """节点7：保存对话历史"""
    state["chat_history"].append({
        "user": state["user_query"],
        "assistant": state["final_response"],
        "intent": state["intent"]
    })
    if len(state["chat_history"]) > 5:
        state["chat_history"] = state["chat_history"][-5:]
    return state
