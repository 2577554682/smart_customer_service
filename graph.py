from typing import Literal

from langgraph.graph import StateGraph, END

from state import State
from nodes import (
    analyze_intent,
    retrieve_documents,
    generate_response,
    check_quality,
    improve_response,
    finalize,
    save_history,
)


def route_by_intent(state: State) -> Literal["retrieve", "skip_retrieve"]:
    if state["intent"] in ["consultation", "return_order"]:
        return "retrieve"
    return "skip_retrieve"


def route_by_quality(state: State) -> Literal["improve", "finalize"]:
    if state["retry_count"] >= 2:
        return "finalize"
    if state["quality_score"] >= 90:
        return "finalize"
    if state["retry_count"] > 0 and state["quality_score"] <= state.get("previous_score", 0):
        return "finalize"
    return "improve"


def build_graph():
    graph = StateGraph(State)

    graph.add_node("analyze_intent", analyze_intent)
    graph.add_node("retrieve_docs", retrieve_documents)
    graph.add_node("generate_response", generate_response)
    graph.add_node("check_quality", check_quality)
    graph.add_node("improve_response", improve_response)
    graph.add_node("finalize", finalize)
    graph.add_node("save_history", save_history)

    graph.set_entry_point("analyze_intent")

    graph.add_conditional_edges(
        "analyze_intent",
        route_by_intent,
        {"retrieve": "retrieve_docs", "skip_retrieve": "generate_response"}
    )

    graph.add_edge("retrieve_docs", "generate_response")
    graph.add_edge("generate_response", "check_quality")

    graph.add_conditional_edges(
        "check_quality",
        route_by_quality,
        {"improve": "improve_response", "finalize": "finalize"}
    )

    graph.add_edge("improve_response", "check_quality")
    graph.add_edge("finalize", "save_history")
    graph.add_edge("save_history", END)

    return graph.compile()
