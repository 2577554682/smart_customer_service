import logging
from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)


def _make_initial_state(query: str) -> dict:
    return {
        "user_query": query,
        "intent": "",
        "need_human": False,
        "rag_context": "",
        "draft_response": "",
        "quality_score": 0,
        "previous_score": 0,
        "retry_count": 0,
        "final_response": "",
        "chat_history": []
    }


def run_test():
    app = build_graph()

    test_queries = [
        "你好，请问你们支持退货吗？",
        "怎么重置密码？",
        "你们的客服太差了，我等了10分钟没人回！",
        "今天天气真好",
        "",
        "A" * 600,
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"用户：{query}")
        try:
            result = app.invoke(_make_initial_state(query))
        except Exception as e:
            logging.getLogger(__name__).error("invoke 失败：%s", e)
            continue
        print(f"客服：{result['final_response']}")
        print(f"意图：{result['intent']} | 转人工：{result['need_human']} | 重试：{result['retry_count']}")


if __name__ == '__main__':
    run_test()
