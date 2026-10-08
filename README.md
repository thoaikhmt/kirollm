# kirollm

Python, LangChain, and LangGraph access to Kiro's streaming model API.

Repository: [github.com/thoaikhmt/kirollm](https://github.com/thoaikhmt/kirollm)

The distribution name is `kirollm`; the Python import package is `kirollm`
because hyphens are not valid Python identifiers.

> This is an unofficial clean-room client reconstructed from Kiro CLI protocol
> behavior. It is not affiliated with or supported by AWS/Kiro, and private
> service contracts can change.

## Features

- Kiro `GenerateAssistantResponse` event-stream chat
- Server-authoritative, paginated model discovery
- API-key and IdC/OAuth bearer authentication
- Text, images, documents, system prompts, context, and tool results
- Sync and async streaming
- LangChain `BaseChatModel` integration
- LangGraph tool loops, including parallel tool-result history
- ChatOpenAI-style `bind_tools()` arguments
- Pydantic, TypedDict, and JSON Schema structured output
- Reasoning, usage, context, citation, and model-availability metadata
- Opt-in real-model GitHub Actions test

## Install

```bash
pip install -e .

# LangChain only
pip install -e ".[langchain]"

# LangChain + tested LangGraph versions
pip install -e ".[langgraph]"
```

## Authentication

`kiro-cli` is not required. `KiroLLMClient()` resolves credentials in this order:

1. `KIRO_API_KEY` (a `ksk_...` headless API key)
2. `KIRO_ACCESS_TOKEN` (an OAuth/IdC bearer token)
3. Direct IdC credentials from `~/.local/share/kiro-cli/data.sqlite3`, including
   SSO-OIDC refresh

```bash
export KIRO_API_KEY="ksk_..."
```

```python
from kirollm import KiroLLMClient

client = KiroLLMClient()

# Explicit alternatives
client = KiroLLMClient(access_token="ksk_...", region="us-east-1")
client = KiroLLMClient.from_kiro_login()
```

`KiroLLMClient.from_kiro_cli()` is optional for provider-specific sessions that
must delegate refresh to an installed CLI.

## Model discovery

Model availability depends on account entitlements and region. Do not hard-code
a catalog:

```python
from kirollm import KiroLLMClient

client = KiroLLMClient()
catalog = client.list_all_models()

for model in catalog.models:
    print(
        model.model_id,
        model.name,
        model.supported_input_types,
        model.max_input_tokens,
    )

model_id = catalog.default_model or catalog.model_ids[0]
```

`list_models()` returns one page; `list_all_models()` follows all `nextToken`
pages. Both preserve model descriptions, input types, rate multipliers, and token
limits.

## Low-level chat

```python
from kirollm import ChatSession, KiroLLMClient

client = KiroLLMClient()
model_id = client.list_all_models().default_model
chat = ChatSession(client, model_id=model_id)

print(chat.send("Explain event streams in one sentence."))

for text in chat.stream("Give one Python example."):
    print(text, end="", flush=True)
```

## LangChain

`ChatKiro` subclasses the same `BaseChatModel` used by `ChatOpenAI`:

```python
from kirollm.langchain import ChatKiro

base = ChatKiro()
model_id = base.list_models().default_model
llm = ChatKiro(model=model_id)

message = llm.invoke("Hello")
print(message.content)
print(message.usage_metadata)
print(message.response_metadata)
```

### Tool calling

```python
from langchain_core.tools import tool
from kirollm.langchain import ChatKiro

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Clear skies in {city}"

llm = ChatKiro().bind_tools(
    [get_weather],
    tool_choice="get_weather",
    strict=True,
    parallel_tool_calls=False,
)
response = llm.invoke("What is the weather in Paris?")
print(response.tool_calls)
```

Accepted `tool_choice` forms match common `ChatOpenAI` calls: `None`, `False`,
`"auto"`, `"any"`, `"required"`, `True`, a tool name, or an OpenAI function
choice dictionary. Kiro has no native tool-choice wire field, so required/named
choices and `parallel_tool_calls=False` are reinforced and validated locally.

### Pydantic structured output

```python
from pydantic import BaseModel, Field
from kirollm.langchain import ChatKiro

class Destination(BaseModel):
    city: str
    country: str
    confidence: float = Field(ge=0, le=1)

structured = ChatKiro().with_structured_output(
    Destination,
    method="function_calling",
    strict=True,
)
result = structured.invoke("Normalize: Paris, France")
print(result.model_dump())
```

`TypedDict` and JSON Schema inputs return dictionaries. `include_raw=True`
returns the same shape commonly used by `ChatOpenAI`:

```python
{
    "raw": AIMessage(...),
    "parsed": Destination(...),
    "parsing_error": None,
}
```

Kiro structured output supports `method="function_calling"`. OpenAI-native
`json_mode`, `json_schema`, and `bind_tools(response_format=...)` are rejected
with clear errors rather than silently ignored.

### Streaming and async

```python
import asyncio
from kirollm.langchain import ChatKiro

llm = ChatKiro()

for chunk in llm.stream("Count to three"):
    print(chunk.content, end="")

async def main() -> None:
    async for chunk in llm.astream("Name three colors"):
        print(chunk.content, end="")

asyncio.run(main())
```

Tool-call chunks, reasoning content, response metadata, and usage metadata are
preserved during streaming.

## LangGraph

```python
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

class State(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatKiro().bind_tools([multiply])

def call_model(state: State):
    return {"messages": [model.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "model")
builder.add_conditional_edges(
    "model", tools_condition, {"tools": "tools", "__end__": END}
)
builder.add_edge("tools", "model")
agent = builder.compile()

result = agent.invoke(
    {"messages": [HumanMessage(content="What is 12 times 7?")]}
)
print(result["messages"][-1].content)
```

The tested dependency pair is pinned because incompatible `langgraph` and
`langgraph-prebuilt` combinations can fail at import time.

## Images and documents

```python
from kirollm import ChatSession, DocumentInput, ImageInput, KiroLLMClient

chat = ChatSession(KiroLLMClient())

chat.send("Describe this", images=["photo.png"])
chat.send("Summarize this", documents=["report.pdf"])

image = ImageInput.from_data_url("data:image/png;base64,...")
document = DocumentInput.from_file("notes.md")
chat.send("Compare them", images=[image], documents=[document])
```

Images support PNG, JPEG, GIF, and WebP. LangChain `image_url` data-URL blocks
and Anthropic-style image blocks are converted. Remote image URLs must be
downloaded by the caller first.

## Voice input

The installed Kiro API does not expose a stable public voice request schema.
Voice is therefore a provider-neutral transcription adapter; only transcript
text is sent to Kiro:

```python
from kirollm import AudioInput, ChatSession, KiroLLMClient

class WhisperAdapter:
    def transcribe(self, audio: AudioInput) -> str:
        return whisper_transcribe(audio.data, audio.mime_type)

chat = ChatSession(KiroLLMClient())
reply = chat.send_voice(
    AudioInput(open("question.wav", "rb").read()),
    WhisperAdapter(),
)
```

## ChatOpenAI compatibility

| Pattern | ChatKiro |
|---|---|
| `invoke`, `stream`, `ainvoke`, `astream`, `batch` | Supported |
| `bind_tools` with Pydantic, TypedDict, callable, `BaseTool`, dict | Supported |
| `tool_choice` common forms | Supported with local enforcement |
| `strict` structured parsing | Supported by LangChain parser validation |
| `parallel_tool_calls=False` | Locally validated |
| `with_structured_output(Pydantic/TypedDict/JSON Schema)` | Supported |
| `include_raw=True` | Supported |
| OpenAI `json_mode` / native `json_schema` | Not supported by Kiro |
| OpenAI `response_format` wire parameter | Not supported by Kiro |
| Stop sequences | Not exposed by Kiro |

## Context and response types

`UserInputMessageContext` supports tools, tool results, editor state, environment
state, Git state, shell state, user settings, cursor state, documents, relevant
text/documents, additional context, and `useRelevantDocuments`.

Responses expose assistant text, reasoning, tool use, code references,
supplementary web links, context usage, token/metering metadata, invalid state,
and model-unavailable events.

## Live model test

`tests/integration/test_live_model.py` discovers the live model catalog, chooses
the advertised default, and performs one real generation. It is opt-in:

```bash
KIRO_API_KEY="ksk_..." python -m pytest -q -m integration
```

GitHub Actions runs it only via **Run workflow** with `live_model=true`, using the
repository secret `KIRO_API_KEY`. Normal push/PR jobs exclude live tests.

## Examples

- `examples/basic_stream.py`
- `examples/tool_calling.py`
- `examples/structured_output.py`
- `examples/streaming.py`
- `examples/langgraph_agent.py`
- `examples/langchain_usage.py`

## Development

```bash
pip install -e ".[test]"
pytest -q -m "not integration"
ruff check src tests examples
pyright src/kirollm
```

## License

MIT
