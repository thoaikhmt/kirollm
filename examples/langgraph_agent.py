"""A minimal LangGraph tool loop powered by ChatKiro."""

from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from kirollm.langchain import ChatKiro


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


base = ChatKiro()
model_id = base.list_models().default_model
model = ChatKiro(model=model_id).bind_tools([multiply])


def call_model(state: AgentState):
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(AgentState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "model")
builder.add_conditional_edges(
    "model", tools_condition, {"tools": "tools", "__end__": END}
)
builder.add_edge("tools", "model")
agent = builder.compile()

result = agent.invoke(
    {"messages": [HumanMessage(content="What is 12 multiplied by 7?")]}
)
print(result["messages"][-1].content)
