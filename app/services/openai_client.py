"""The only module that knows about the OpenAI SDK.

Everything the rest of the app needs from a reply — text and token counts — is
returned as plain values, so no SDK object leaks into the services or routes.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI, OpenAIError

from app import config
from app.errors import UpstreamError

logger = logging.getLogger(__name__)

# An explicit timeout: without one a hung call would hold the request open.
client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=config.OPENAI_TIMEOUT_SECONDS)


@dataclass
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    # What OpenAI actually served. Asking for "gpt-4o-mini" answers as
    # "gpt-4o-mini-2024-07-18"; kept for the record, not used for pricing.
    resolved_model: str


def complete(model: str, messages: list[dict]) -> Completion:
    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except OpenAIError as exc:
        # Timeouts, rate limits and 5xx all arrive as OpenAIError. The detail
        # goes to the log only: the SDK message can carry the request payload
        # and a masked key, and neither belongs in an HTTP response.
        logger.error("OpenAI request failed: %s", exc)
        raise UpstreamError("Upstream model request failed.") from exc

    return Completion(
        content=response.choices[0].message.content or "",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        resolved_model=response.model,
    )
