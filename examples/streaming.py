"""Synchronous and asynchronous ChatKiro streaming."""

import asyncio

from kiro_llm.langchain import ChatKiro


llm = ChatKiro()
model_id = llm.list_models().default_model
model = ChatKiro(model=model_id)

for chunk in model.stream("Count from one to three."):
    print(chunk.content, end="", flush=True)
print()


async def main() -> None:
    async for chunk in model.astream("Name three primary colors."):
        print(chunk.content, end="", flush=True)
    print()


asyncio.run(main())
