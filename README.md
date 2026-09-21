# kiro-llm


Repository: [github.com/thoaikhmt/kiro-llm](https://github.com/thoaikhmt/kiro-llm)

The repository and distribution name use `kiro-llm`. The Python import package
intentionally remains `kiro_llm`, because hyphens are not valid in Python import
identifiers.
A Python and LangChain client for the streaming Kiro model API observed in the
installed Kiro CLI.

It implements the request/response mechanics used by the CLI's Rust agent
backend:

- **Bearer-token auth** — `Authorization: Bearer <token>` (Kiro uses an SSO/IdC
  access token; the underlying AWS Smithy client uses `httpBearerAuth`).
- **`GenerateAssistantResponse`** operation — `application/x-amz-json-1.0` POST
  with header `X-Amz-Target: AmazonCodeWhispererStreamingService.GenerateAssistantResponse`.
- **`ConversationState` payload** — `conversationId`, `currentMessage`
  (`userInputMessage` + `userInputMessageContext` with `tools` / `toolResults`),
  `history`, and `chatTriggerType`.
- **`ChatResponseStream` decoding** — the response body is an
  `application/vnd.amazon.eventstream` binary stream; this library decodes it and
  yields typed events (`AssistantResponseEvent`, `ToolUseEvent`, `ReasoningEvent`,
  `CodeReferenceEvent`, `MetadataEvent`, …).
- **Region endpoints** — the `q.<region>.amazonaws.com` and `runtime.<region>.kiro.dev`
  host tables Kiro ships with.
- **Auto-auth** — delegates token acquisition and provider-specific refresh to
  the installed Kiro CLI; explicit tokens and a legacy SQLite IdC fallback are
  also supported.
- **LangChain / LangGraph** — ships `ChatKiro`, a `BaseChatModel` (the same base
  `ChatOpenAI` extends) that plugs straight into chains, agents, and LangGraph.

> This is an unofficial, clean-room implementation reconstructed from the
> installed Kiro CLI binary's symbols and endpoint tables. It is not affiliated
> with or supported by AWS/Kiro, and private service contracts may change.

## Install

```bash
cd kiro-llm
pip install -e .
# LangChain integration:
pip install -e ".[langchain]"
# for tests:
pip install -e ".[test]"
```

## Authentication

`kiro-cli` is **not required**. By default, `KiroLLMClient()` uses the first
available option:

1. `KIRO_ACCESS_TOKEN` or `KIRO_API_KEY` environment variable.
2. The direct Kiro `auth_kv` SQLite store at
   `~/.local/share/kiro-cli/data.sqlite3` for IdC credentials, including direct
   SSO-OIDC refresh.

```python
from kiro_llm import KiroLLMClient

client = KiroLLMClient()                    # env or direct SQLite auth
client = KiroLLMClient.from_kiro_login()    # explicit direct SQLite auth

# Explicit bearer token (no refresh metadata):
client = KiroLLMClient(access_token="<bearer-token>", region="us-east-1")
```

`KiroLLMClient.from_kiro_cli()` remains an optional adapter for installations
that already use Kiro CLI-specific social, external-IdP, or API-key sessions.
The library does not require the executable for normal environment or IdC
SQLite authentication.

## Usage

```python
from kiro_llm import KiroLLMClient, ChatSession
## Live model integration test

The repository includes one real-model test at
`tests/integration/test_live_model.py`. It is opt-in and skips when
`KIRO_API_KEY` is absent. To run it locally:

```bash
KIRO_API_KEY="ksk_..." python -m pytest -q -m integration
```

GitHub Actions exposes the same test only through **Run workflow** with the
`live_model` input enabled. The key is read from the repository Actions secret
named `KIRO_API_KEY`; it is never stored in the repository or printed by the
workflow. Normal pushes and pull requests run unit/static checks with
integration tests excluded.


client = KiroLLMClient()                       # auto-auth
chat = ChatSession(client, model_id="claude-sonnet-4")

# Streaming
for chunk in chat.stream("Write a haiku about event streams"):
    print(chunk, end="", flush=True)

# Or blocking
reply = chat.send("Now explain it in one sentence.")
print(reply)
```

## LangChain / LangGraph

`ChatKiro` is a `langchain_core` chat model (subclass of `BaseChatModel`, the same
base `ChatOpenAI` extends), so it works anywhere a LangChain chat model does.

```python
from kiro_llm.langchain import ChatKiro

llm = ChatKiro(model="claude-sonnet-4")        # auto-auth
## Model discovery and selection

Model IDs are entitlement- and region-dependent. Discover the catalog instead of
hard-coding model names:

```python
from kiro_llm import KiroLLMClient

client = KiroLLMClient()
catalog = client.list_models()
print(catalog.model_ids)
model_id = catalog.default_model or catalog.model_ids[0]
```

The same discovery method is available on `ChatKiro`:

```python
from kiro_llm.langchain import ChatKiro

llm = ChatKiro()
for model in llm.list_models().models:
    print(model.model_id, model.name, model.supported_input_types)
```

Discovery calls Kiro's `AmazonCodeWhispererService.ListAvailableModels` operation
and preserves descriptions, supported input types, rate multipliers, and token
limits. Pass the selected ID as `ChatKiro(model=model_id)`.

print(llm.invoke("Hello!").content)

for chunk in llm.stream("Count to five"):
    print(chunk.content, end="")

# Tool calling
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    "Get the weather for a city."
    return "sunny"

llm_with_tools = llm.bind_tools([get_weather])
ai = llm_with_tools.invoke("Weather in Paris?")
print(ai.tool_calls)

# LangGraph ReAct agent
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(ChatKiro(model="claude-sonnet-4"), [get_weather])
agent.invoke({"messages": [("user", "What's the weather in Paris?")]})
```

### Tool use

```python
from kiro_llm import ChatSession, KiroLLMClient, ToolSpecification, ToolResult

weather = ToolSpecification(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

client = KiroLLMClient(access_token="...", region="us-east-1")
chat = ChatSession(client, model_id="claude-sonnet-4", tools=[weather])

chat.send("What's the weather in Paris?")
for call in chat.pending_tool_uses:
    # run the tool yourself, then reply with the result:
    result = ToolResult.text(call.tool_use_id, '{"tempC": 18, "sky": "clear"}')
    final = chat.send_tool_results([result])
    print(final)
```

### Low-level client

```python
from kiro_llm import KiroLLMClient, ConversationState, UserInputMessage

client = KiroLLMClient(access_token="...", region="us-east-1")
state = ConversationState(current_message=UserInputMessage(content="Hello!"))
for event in client.generate_assistant_response(state):
    print(event)
```

## Images and voice input

Images are supported through Kiro's observed `ImageBlock` wire shape. Use the
built-in helpers instead of hand-building base64 payloads:

```python
from kiro_llm import ChatSession, KiroLLMClient, image_from_file

chat = ChatSession(KiroLLMClient(), model_id="claude-sonnet-4")
print(chat.send("What is in this image?", images=[image_from_file("photo.png")]))
# A file path is also accepted directly by ChatSession.
print(chat.send("Read this diagram", images=["diagram.jpg"]))
```

LangChain image blocks using base64 data URLs are supported:

```python
from kiro_llm.langchain import ChatKiro

llm = ChatKiro(model="claude-sonnet-4")
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ],
}
print(llm.invoke([message]).content)
```

Voice input is exposed as a safe adapter rather than an unverified private Kiro
voice endpoint. The installed CLI contains voice command strings, but the
currently installed executable does not expose a stable callable voice protocol.
Transcribe audio with Whisper or another ASR provider and send the transcript
through Kiro:

```python
from kiro_llm import AudioInput, ChatSession, KiroLLMClient

class MyTranscriber:
    def transcribe(self, audio: AudioInput) -> str:
        return whisper_transcribe(audio.data, audio.mime_type)

chat = ChatSession(KiroLLMClient())
reply = chat.send_voice(AudioInput(open("question.wav", "rb").read()), MyTranscriber())
```

No audio bytes are sent to the Kiro text endpoint; the adapter owns audio
transcription and only the resulting text enters the verified conversation API.


The Kiro CLI is a split process: a TUI frontend talks to a Rust agent backend
over the Agent Client Protocol (ACP, JSON-RPC). The backend uses an AWS
Smithy-generated `amzn-codewhisperer-streaming-client` to call
`GenerateAssistantResponse`. This library collapses that into a single direct
HTTP client that speaks the same operation, payload, headers, and event-stream
response format.

| Kiro CLI                                                   | kiro-llm                          |
| ---------------------------------------------------------- | --------------------------------- |
| `httpBearerAuth` (SSO/IdC token)                           | `Authorization: Bearer` header    |
| `AmazonCodeWhispererStreamingService.GenerateAssistantResponse` | `X-Amz-Target` header        |
| `ConversationState` / `UserInputMessage` / `ToolSpecification` | `kiro_llm.models`             |
| `ChatResponseStream` (event stream)                        | `kiro_llm.eventstream` + events   |
| `q.<region>.amazonaws.com` / `runtime.<region>.kiro.dev`   | endpoint tables in `client.py`    |
| `auth_kv` secret store + SSO-OIDC refresh                  | `kiro_llm.auth`                   |

## Layout

```
src/kiro_llm/
  __init__.py      public API
  eventstream.py   vnd.amazon.eventstream decoder/encoder
  models.py        ConversationState & ChatResponseStream event types
  client.py        KiroLLMClient: endpoints, bearer auth, streaming call
  auth.py          load Kiro's stored token + auto-refresh (SSO-OIDC)
  media.py         image file/bytes/data-URL and audio input helpers
  voice.py         provider-neutral audio transcription adapter
  chat.py          ChatSession: history + multimodal helpers
  langchain.py     ChatKiro: LangChain BaseChatModel (LangGraph-ready)
examples/
  basic_stream.py
  langchain_usage.py
tests/
```

## License

MIT
