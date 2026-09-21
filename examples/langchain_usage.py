"""
Example: use Kiro as a LangChain chat model (works in LangGraph too).

ChatKiro subclasses langchain_core's BaseChatModel -- the same base ChatOpenAI
uses -- so it drops into chains, agents, and LangGraph graphs. It auto-loads the
Kiro CLI's stored login and refreshes the token automatically.

Install:
    pip install kiro-llm[langchain]

Run:
    python examples/langchain_usage.py
"""

from kiro_llm.langchain import ChatKiro


def basic():
    # No access_token needed: loads + refreshes the Kiro CLI token automatically.
    llm = ChatKiro(model="claude-sonnet-4")
    print(llm.invoke("Write a haiku about event streams").content)

    print("--- streaming ---")
    for chunk in llm.stream("Count to five."):
        print(chunk.content, end="", flush=True)
    print()


def with_tools():
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"It's sunny in {city}."

    llm = ChatKiro(model="claude-sonnet-4").bind_tools([get_weather])
    ai = llm.invoke("What's the weather in Paris?")
    for call in ai.tool_calls:
        print("tool call:", call["name"], call["args"])


def langgraph_agent():
    """A minimal LangGraph ReAct agent backed by Kiro."""
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    llm = ChatKiro(model="claude-sonnet-4")
    agent = create_react_agent(llm, [add])
    result = agent.invoke({"messages": [("user", "What is 21 + 21?")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    basic()
    with_tools()
    # langgraph_agent()  # requires `pip install langgraph`
