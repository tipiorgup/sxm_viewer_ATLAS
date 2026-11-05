"""Per model request settings for the hosted MISO assistant.

Profiles are keyed by the API model name (not the display name shown on a
provider web page). Unknown models fall back to DEFAULT_PROFILE, which sends
no optional sampling parameters, so a model that rejects them is never hurt.

Sampling values for GWDG models follow the recommendations on
https://docs.hpc.gwdg.de/services/ai-services/chat-ai/models/index.html
(checked November 2026). The list of models changes over time; refresh the
table when the service list changes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: int = 45
    # Ask the service for a JSON only reply. Off everywhere until the
    # service is confirmed to accept the option for the model in question.
    json_mode: bool = False

    def sampling_options(self) -> dict:
        """Optional request fields for OpenAI compatible chat completions."""
        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        if self.json_mode:
            options["response_format"] = {"type": "json_object"}
        return options


DEFAULT_PROFILE = ModelProfile()

PROFILES: dict[str, ModelProfile] = {
    # Fast, non thinking model. First choice for this assistant.
    "qwen3-30b-a3b-instruct-2507": ModelProfile(temperature=0.6, top_p=0.95, timeout=60),
    "gemma-4-31b-it": ModelProfile(timeout=60),
    "qwen3.6-35b-a3b": ModelProfile(temperature=1.0, top_p=0.95, timeout=90),
    # Small and fast, but weak at strict formats; a low temperature helps.
    "meta-llama-3.1-8b-instruct": ModelProfile(temperature=0.2, timeout=45),
    "glm-5.3-flash": ModelProfile(temperature=1.0, top_p=0.95, timeout=90),
    "deepseek-v4-flash-0731": ModelProfile(temperature=1.0, top_p=1.0, timeout=120),
    # Large or reasoning models can need a long time before the first token.
    "qwen3.5-397b-a17b": ModelProfile(temperature=0.6, top_p=0.95, timeout=120),
    "qwen3.8-27b": ModelProfile(temperature=1.0, top_p=0.95, timeout=120),
}


def profile_for(model: str) -> ModelProfile:
    """Profile for an API model name; case and surrounding spaces are ignored."""
    return PROFILES.get((model or "").strip().lower(), DEFAULT_PROFILE)
