"""Basic ChatKiro invocation, streaming, and tool binding.

Install with ``pip install -e '.[langchain]'`` and set ``KIRO_API_KEY`` or use a
supported local IdC credential store.
"""

from langchain_core.tools import tool

from kirollm.langchain import ChatKiro


base = ChatKiro()
catalog = base.list_models()
model_id = catalog.default_model or catalog.model_ids[0]
llm = ChatKiro(model=model_id)

print(llm.invoke("Write a haiku about event streams").content)

for chunk in llm.stream("Count to three"):
    print(chunk.content, end="", flush=True)
print()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Clear skies in {city}"


response = llm.bind_tools([get_weather]).invoke("Weather in Paris?")
print(response.tool_calls)
