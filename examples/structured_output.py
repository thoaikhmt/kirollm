"""Pydantic structured output via Kiro function calling."""

from pydantic import BaseModel, Field

from kiro_llm.langchain import ChatKiro


class Destination(BaseModel):
    """A normalized travel destination."""

    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    confidence: float = Field(ge=0, le=1)


llm = ChatKiro()
model_id = llm.list_models().default_model
structured = ChatKiro(model=model_id).with_structured_output(
    Destination,
    method="function_calling",
    strict=True,
)
result = structured.invoke("Normalize: Paris, France")
print(result.model_dump())

raw_result = ChatKiro(model=model_id).with_structured_output(
    Destination,
    include_raw=True,
).invoke("Normalize: Kyoto, Japan")
print(raw_result["raw"])
print(raw_result["parsed"])
print(raw_result["parsing_error"])
