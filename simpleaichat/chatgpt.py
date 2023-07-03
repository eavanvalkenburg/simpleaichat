from pydantic import HttpUrl, ConfigDict
from httpx import Client, AsyncClient
from typing import List, Dict, Set, Any
import orjson

from .models import ChatMessage, ChatSession
from .prompts import TOOL_PROMPT, TOOL_INPUT_PROMPT
from .utils import remove_a_key


def response_to_chat_message(response: dict[str, Any]) -> ChatMessage | None:
    """Converts a response from the API to a ChatMessage object."""
    if "choices" in response and len(response["choices"]) > 0:
        choices = response["choices"][0]["message"]
        usage = response["usage"]
        return ChatMessage(
            role=choices["role"],
            content=choices["content"],
            prompt_length=usage["prompt_tokens"],
            completion_length=usage["completion_tokens"],
            total_length=usage["total_tokens"],
        )
    return None


class ChatGPTSession(ChatSession):
    api_url: HttpUrl = "https://api.openai.com"
    api_type: str = "openai"
    api_version: str = "2023-05-15-preview"
    input_fields: Set[str] = {"role", "content"}
    system: str = "You are a helpful assistant."
    params: Dict[str, Any] = {"temperature": 0.7}
    model_config: ConfigDict(arbitrary_types_allowed=True)

    def prepare_request(
        self,
        prompt: str,
        system: str | None = None,
        params: Dict[str, Any] | None = None,
        stream: bool = False,
    ):
        if self.api_type == "azure":
            headers = {
                "Content-Type": "application/json",
                "api-key": f"{self.auth['api_key'].get_secret_value()}",
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.auth['api_key'].get_secret_value()}",
            }

        system_message = ChatMessage(role="system", content=system or self.system)
        user_message = ChatMessage(role="user", content=prompt)

        gen_params = params or self.params
        data = {
            "model": self.model,
            "messages": self.format_input_messages(system_message, user_message),
            "stream": stream,
            **gen_params,
        }

        # Add function calling parameters if a schema is provided
        if input_schema or output_schema:
            functions = []
            if input_schema:
                input_function = self.schema_to_function(input_schema)
                functions.append(input_function)
            if output_schema:
                output_function = self.schema_to_function(output_schema)
                functions.append(
                    output_function
                ) if output_function not in functions else None
                if is_function_calling_required:
                    data["function_call"] = {"name": output_schema.__name__}
            data["functions"] = functions

        return headers, data, user_message

    def schema_to_function(self, schema: Any):
        assert schema.__doc__, f"{schema.__name__} is missing a docstring."
        schema_dict = schema.model_json_schema()
        remove_a_key(schema_dict, "title")

        return {
            "name": schema.__name__,
            "description": schema.__doc__,
            "parameters": schema_dict,
        }

    def gen(
        self,
        prompt: str,
        client: Client,
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ):
        endpoint, headers, data, user_message = self.prepare_request(
            prompt, system, params
        )

        r = client.post(
            endpoint,
            json=data,
            headers=headers,
            timeout=None,
        )
        if assistant_message := response_to_chat_message(r.json()):
            self.add_messages(user_message, assistant_message, save_messages)
            self.total_prompt_length += (
                assistant_message.prompt_length
                if assistant_message.prompt_length
                else 0
            )
            self.total_completion_length += (
                assistant_message.completion_length
                if assistant_message.completion_length
                else 0
            )
            self.total_length += (
                assistant_message.total_length if assistant_message.total_length else 0
            )
            return assistant_message.content
        return ""

    def stream(
        self,
        prompt: str,
        client: Client,
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ):
        endpoint, headers, data, user_message = self.prepare_request(
            prompt, system, params, stream=True
        )
        with client.stream(
            "POST",
            endpoint,
            json=data,
            headers=headers,
            timeout=None,
        ) as r:
            content = []
            for chunk in r.iter_lines():
                if len(chunk) > 0:
                    chunk = chunk[6:]  # SSE JSON chunks are prepended with "data: "
                    if chunk != "[DONE]":
                        chunk_dict = orjson.loads(chunk)
                        if "choices" in chunk_dict and len(chunk_dict["choices"]) > 0:
                            delta = chunk_dict["choices"][0]["delta"].get("content")
                            if delta:
                                content.append(delta)
                                yield {"delta": delta, "response": "".join(content)}

        assistant_message = ChatMessage(
            role="assistant",
            content="".join(content),
        )

        self.add_messages(user_message, assistant_message, save_messages)
        return assistant_message

    def gen_with_tools(
        self,
        prompt: str,
        tools: list[Any],
        client: Client,
        system: str | None = None,
        save_messages: bool | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # call 1: select tool and populate context
        tools_list = "\n".join(f"{i+1}: {f.__doc__}" for i, f in enumerate(tools))
        tool_prompt_format = TOOL_PROMPT.format(tools=tools_list)

        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 15 + len(tools) + 1)}

        tool_idx = int(
            self.gen(
                prompt,
                client=client,
                system=tool_prompt_format,
                save_messages=False,
                params={
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logit_bias": logit_bias,
                },
            )
        )

        # if no tool is selected, do a standard generation instead.
        if tool_idx == 0:
            return {
                "response": self.gen(
                    prompt,
                    client=client,
                    system=system,
                    save_messages=save_messages,
                    params=params,
                ),
                "tool": None,
            }
        selected_tool = tools[tool_idx - 1]
        tool_input = self.gen(
            prompt,
            client=client,
            system=TOOL_INPUT_PROMPT.format(tool=selected_tool.__doc__),
            save_messages=False,
            params={"temperature": 0.0, "max_tokens": 200},
        )
        if tool_input == "None":
            tool_input = ""
        context_dict = selected_tool(tool_input)
        if isinstance(context_dict, str):
            context_dict = {"context": context_dict}

        context_dict["tool"] = selected_tool.__name__

        # call 2: generate from the context
        new_system = f"{system or self.system}\n\nYou MUST use information from the context in your response."
        new_prompt = f"Context: {context_dict['context']}\n\nUser: {prompt}"

        context_dict["response"] = self.gen(
            new_prompt,
            client=client,
            system=new_system,
            save_messages=False,
            params=params,
        )

        # manually append the nonmodified user message + normal AI response
        user_message = ChatMessage(role="user", content=prompt)
        assistant_message = ChatMessage(
            role="assistant", content=context_dict["response"]
        )
        self.add_messages(user_message, assistant_message, save_messages)

        return context_dict

    async def gen_async(
        self,
        prompt: str,
        client: AsyncClient,
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ):
        endpoint, headers, data, user_message = self.prepare_request(
            prompt, system, params
        )

        r = await client.post(
            endpoint,
            json=data,
            headers=headers,
            timeout=None,
        )
        if assistant_message := response_to_chat_message(r.json()):
            self.add_messages(user_message, assistant_message, save_messages)
            self.total_prompt_length += (
                assistant_message.prompt_length
                if assistant_message.prompt_length
                else 0
            )
            self.total_completion_length += (
                assistant_message.completion_length
                if assistant_message.completion_length
                else 0
            )
            self.total_length += (
                assistant_message.total_length if assistant_message.total_length else 0
            )
            return assistant_message.content
        return ""

    async def stream_async(
        self,
        prompt: str,
        client: AsyncClient,
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ):
        endpoint, headers, data, user_message = self.prepare_request(
            prompt, system, params, stream=True
        )

        async with client.stream(
            "POST",
            endpoint,
            json=data,
            headers=headers,
            timeout=None,
        ) as r:
            content = []
            async for chunk in r.aiter_lines():
                if len(chunk) > 0:
                    chunk = chunk[6:]  # SSE JSON chunks are prepended with "data: "
                    if chunk != "[DONE]":
                        chunk_dict = orjson.loads(chunk)
                        delta = chunk_dict["choices"][0]["delta"].get("content")
                        if delta:
                            content.append(delta)
                            yield {"delta": delta, "response": "".join(content)}

        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content="".join(content),
        )

        self.add_messages(user_message, assistant_message, save_messages)

    async def gen_with_tools_async(
        self,
        prompt: str,
        tools: List[Any],
        client: AsyncClient,
        system: str | None = None,
        save_messages: bool | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        # call 1: select tool and populate context
        tools_list = "\n".join(f"{i+1}: {f.__doc__}" for i, f in enumerate(tools))
        tool_prompt_format = TOOL_PROMPT.format(tools=tools_list)

        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 15 + len(tools) + 1)}

        tool_idx = int(
            await self.gen_async(
                prompt,
                client=client,
                system=tool_prompt_format,
                save_messages=False,
                params={
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logit_bias": logit_bias,
                },
            )
        )

        # if no tool is selected, do a standard generation instead.
        if tool_idx == 0:
            return {
                "response": await self.gen_async(
                    prompt,
                    client=client,
                    system=system,
                    save_messages=save_messages,
                    params=params,
                ),
                "tool": None,
            }
        selected_tool = tools[tool_idx - 1]
        context_dict = await selected_tool(prompt)
        if isinstance(context_dict, str):
            context_dict = {"context": context_dict}

        context_dict["tool"] = selected_tool.__name__

        # call 2: generate from the context
        new_system = f"{system or self.system}\n\nYou MUST use information from the context in your response."
        new_prompt = f"Context: {context_dict['context']}\n\nUser: {prompt}"

        context_dict["response"] = await self.gen_async(
            new_prompt,
            client=client,
            system=new_system,
            save_messages=False,
            params=params,
        )

        # manually append the nonmodified user message + normal AI response
        user_message = ChatMessage(role="user", content=prompt)
        assistant_message = ChatMessage(
            role="assistant", content=context_dict["response"]
        )
        self.add_messages(user_message, assistant_message, save_messages)

        return context_dict
