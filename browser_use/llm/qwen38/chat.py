"""ChatQwen38 — OpenAI-compatible client for a local Qwen3.8-27B llama.cpp/Ollama endpoint.

Qwen3.8-27B is a 27B dense VLM (qwen35 / Gated DeltaNet hybrid). Official sampling:
thinking mode temperature=1.0 top_p=0.95; instruct mode temperature=0.7 top_p=0.80.
Point this at a llama-server started by Qwen38LocalService, or at Ollama qwen3.8:27b.
"""

# @file purpose: Chat model wrapper for local Qwen3.8-27B quantized llama.cpp/Ollama endpoints

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, overload

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.qwen38.views import DEFAULT_ALIAS_PREFIX, DEFAULT_QUANT, OLLAMA_MODEL, ReasoningEffort
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)

_THINKING_TEMPERATURE = 1.0
_THINKING_TOP_P = 0.95
_INSTRUCT_TEMPERATURE = 0.7
_INSTRUCT_TOP_P = 0.80


def _log_endpoint(base_url: str, model: str) -> str:
	return f'{model} @ {base_url}'


@dataclass
class ChatQwen38(BaseChatModel):
	"""Local Qwen3.8-27B client. `quant` selects the GGUF alias served by llama-server."""

	quant: str = DEFAULT_QUANT
	model: str = ''
	enable_thinking: bool = True
	preserve_thinking: bool = True
	reasoning_effort: ReasoningEffort = 'xhigh'
	temperature: float | None = None
	top_p: float | None = None
	top_k: int = 20
	max_completion_tokens: int | None = 16384
	base_url: str | httpx.URL | None = None
	api_key: str | None = 'local'
	timeout: float | httpx.Timeout | None = 600.0
	max_retries: int = 3
	default_headers: Mapping[str, str] | None = None
	extra_body: dict[str, Any] = field(default_factory=dict)

	@property
	def provider(self) -> str:
		return 'qwen38-local'

	@property
	def name(self) -> str:
		return self.resolved_model

	def __post_init__(self) -> None:
		ollama = self.quant.lower() in {OLLAMA_MODEL, 'qwen3.8', 'qwen3.8:latest'}
		if not self.model:
			self.model = OLLAMA_MODEL if ollama else f'{DEFAULT_ALIAS_PREFIX}-{self.quant}'
		if self.base_url is None:
			if ollama:
				self.base_url = 'http://127.0.0.1:11434/v1'
			else:
				from browser_use.llm.qwen38.service import fallback_catalog

				try:
					self.base_url = f'http://127.0.0.1:{fallback_catalog().get(self.quant).port}/v1'
				except KeyError:
					self.base_url = 'http://127.0.0.1:8080/v1'

	@property
	def resolved_model(self) -> str:
		return self.model

	def _sampling(self) -> dict[str, Any]:
		thinking = self.enable_thinking
		temperature = (
			self.temperature if self.temperature is not None else (_THINKING_TEMPERATURE if thinking else _INSTRUCT_TEMPERATURE)
		)
		top_p = self.top_p if self.top_p is not None else (_THINKING_TOP_P if thinking else _INSTRUCT_TOP_P)
		params: dict[str, Any] = {
			'temperature': temperature,
			'top_p': top_p,
		}
		if self.max_completion_tokens is not None:
			params['max_completion_tokens'] = self.max_completion_tokens
		return params

	def _template_kwargs(self) -> dict[str, Any]:
		body: dict[str, Any] = {
			'chat_template_kwargs': {
				'enable_thinking': self.enable_thinking,
				'preserve_thinking': self.preserve_thinking,
			},
			'reasoning_effort': self.reasoning_effort,
			'top_k': self.top_k,
		}
		body.update(self.extra_body)
		return body

	def get_client(self) -> AsyncOpenAI:
		assert self.base_url is not None
		return AsyncOpenAI(
			api_key=self.api_key or 'local',
			base_url=str(self.base_url),
			timeout=self.timeout,
			max_retries=self.max_retries,
			default_headers=dict(self.default_headers) if self.default_headers else None,
		)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		if response.usage is None:
			return None
		completion_tokens = response.usage.completion_tokens
		details = response.usage.completion_tokens_details
		if details is not None and details.reasoning_tokens is not None:
			completion_tokens += details.reasoning_tokens
		return ChatInvokeUsage(
			prompt_tokens=response.usage.prompt_tokens,
			prompt_cached_tokens=response.usage.prompt_tokens_details.cached_tokens
			if response.usage.prompt_tokens_details is not None
			else None,
			prompt_cache_creation_tokens=None,
			prompt_image_tokens=None,
			completion_tokens=completion_tokens,
			total_tokens=response.usage.total_tokens,
		)

	def _thinking_text(self, response: ChatCompletion) -> str | None:
		message = response.choices[0].message
		for attr in ('reasoning_content', 'reasoning'):
			value = getattr(message, attr, None)
			if isinstance(value, str) and value:
				return value
		dumped = message.model_dump() if hasattr(message, 'model_dump') else {}
		for key in ('reasoning_content', 'reasoning'):
			value = dumped.get(key)
			if isinstance(value, str) and value:
				return value
		return None

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		openai_messages = OpenAIMessageSerializer.serialize_messages(messages)
		model_params = self._sampling()
		try:
			if output_format is None:
				response = await self.get_client().chat.completions.create(
					model=self.resolved_model,
					messages=openai_messages,
					extra_body=self._template_kwargs(),
					**model_params,
				)
				return ChatInvokeCompletion(
					completion=response.choices[0].message.content or '',
					thinking=self._thinking_text(response),
					usage=self._get_usage(response),
				)

			response_format: JSONSchema = {
				'name': 'agent_output',
				'strict': True,
				'schema': SchemaOptimizer.create_optimized_json_schema(output_format),
			}
			response = await self.get_client().chat.completions.create(
				model=self.resolved_model,
				messages=openai_messages,
				response_format=ResponseFormatJSONSchema(json_schema=response_format, type='json_schema'),
				extra_body=self._template_kwargs(),
				**model_params,
			)
			content = response.choices[0].message.content
			if content is None:
				raise ModelProviderError(
					message='Failed to parse structured output from Qwen3.8-27B response',
					status_code=500,
					model=self.name,
				)
			return ChatInvokeCompletion(
				completion=output_format.model_validate_json(content),
				thinking=self._thinking_text(response),
				usage=self._get_usage(response),
			)
		except RateLimitError as e:
			raise ModelProviderError(
				message=str(e),
				status_code=e.response.status_code if e.response is not None else 429,
				model=self.name,
			) from e
		except APIConnectionError as e:
			raise ModelProviderError(
				message=f'Cannot reach local Qwen3.8-27B at {_log_endpoint(str(self.base_url), self.name)}: {e}',
				model=self.name,
			) from e
		except APIStatusError as e:
			try:
				payload = e.response.json().get('error', {})
				error_message = payload.get('message', e.response.text) if isinstance(payload, dict) else str(payload)
			except Exception:
				error_message = e.response.text
			raise ModelProviderError(
				message=str(error_message),
				status_code=e.response.status_code,
				model=self.name,
			) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e


# Keep a literal around for callers that want to type-narrow sampling mode.
Qwen38Mode = Literal['thinking', 'instruct']
