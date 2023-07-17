from abc import abstractmethod
from datetime import datetime, timezone
from uuid import uuid4, UUID
from httpx import Client, AsyncClient

from pydantic import BaseModel, SecretStr, HttpUrl, Field
from typing import List, Dict, Union, Optional, Set, Any
import orjson


def orjson_dumps(v, *, default, **kwargs):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(v, default=default, **kwargs).decode()


def now_tz():
    # Need datetime w/ timezone for cleanliness
    # https://stackoverflow.com/a/24666683
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    role: str
    content: str
    received_at: datetime = Field(default_factory=now_tz)
    prompt_length: Optional[int] = None
    completion_length: Optional[int] = None
    total_length: Optional[int] = None

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return self.content


class ChatSession(BaseModel):
    id: Union[str, UUID] = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=now_tz)
    auth: Dict[str, SecretStr]
    api_url: HttpUrl
    model: str
    system: str
    params: Dict[str, Any] = {}
    messages: List[ChatMessage] = []
    input_fields: Set[str] = Field(default_factory=Set)
    recent_messages: Optional[int] = None
    save_messages: Optional[bool] = True
    total_prompt_length: int = 0
    total_completion_length: int = 0
    total_length: int = 0
    title: Optional[str] = None

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def __str__(self) -> str:
        sess_start_str = self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        last_message_str = self.messages[-1].received_at.strftime("%Y-%m-%d %H:%M:%S")
        return f"""Chat session started at {sess_start_str}:
        - {len(self.messages):,} Messages
        - Last message sent at {last_message_str}"""

    def format_input_messages(
        self, system_message: ChatMessage, user_message: ChatMessage
    ) -> list:
        recent_messages = (
            self.messages[-self.recent_messages :]
            if self.recent_messages
            else self.messages
        )
        return (
            [system_message.dict(include=self.input_fields)]
            + [m.dict(include=self.input_fields) for m in recent_messages]
            + [user_message.dict(include=self.input_fields)]
        )

    def add_messages(
        self,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        save_messages: bool | None = None,
    ) -> None:
        # if save_messages is explicitly defined, always use that choice
        # instead of the default
        to_save = isinstance(save_messages, bool)

        if to_save:
            if save_messages:
                self.messages.append(user_message)
                self.messages.append(assistant_message)
        elif self.save_messages:
            self.messages.append(user_message)
            self.messages.append(assistant_message)

    @abstractmethod
    def gen(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ) -> str:
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ) -> str:
        pass

    @abstractmethod
    def gen_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        client: Union[Client, AsyncClient],
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def gen_async(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def stream_async(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def gen_with_tools_async(
        self,
        prompt: str,
        tools: List[Any],
        client: Union[Client, AsyncClient],
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        pass
