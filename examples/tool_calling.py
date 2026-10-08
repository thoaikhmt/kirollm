"""ChatOpenAI-style Kiro tool calling.

Run with KIRO_API_KEY or a supported local IdC credential store.
"""

from langchain_core.tools import tool

from kirollm.langchain import ChatKiro


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Clear skies in {city}"


llm = ChatKiro()
model_id = llm.list_models().default_model
model = ChatKiro(model=model_id).bind_tools(
    [get_weather],
    tool_choice="get_weather",
    strict=True,
    parallel_tool_calls=False,
)
response = model.invoke("What is the weather in Paris?")
print(response.tool_calls)
