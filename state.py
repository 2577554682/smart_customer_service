from typing import TypedDict, List

class State(TypedDict):
    user_query: str
    intent: str
    need_human: bool
    rag_context: str
    draft_response: str
    quality_score: int
    previous_score: int
    retry_count: int
    final_response: str
    chat_history: List[dict]
