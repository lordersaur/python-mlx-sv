from typing import Any, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    tool_calls: Optional[list[Any]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 32768
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    presence_penalty: Optional[float] = None
    stream: Optional[bool] = False
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    extra_body: Optional[dict[str, Any]] = None

