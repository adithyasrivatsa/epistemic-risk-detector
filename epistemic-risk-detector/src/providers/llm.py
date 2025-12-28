"""LLM provider implementations for OpenAI, Anthropic, and local models."""

import json
import os
from typing import Any

from src.core.config import LLMConfig


class OpenAIProvider:
    """OpenAI API provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
            )
        return self._client

    def complete(self, prompt: str, temperature: float | None = None) -> str:
        """Generate a completion."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Generate a JSON completion conforming to schema."""
        json_prompt = f"""{prompt}

Respond with valid JSON conforming to this schema:
{json.dumps(schema, indent=2)}

Output only the JSON, no other text."""

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": json_prompt}],
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)


class AnthropicProvider:
    """Anthropic API provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            self._client = Anthropic(api_key=api_key)
        return self._client

    def complete(self, prompt: str, temperature: float | None = None) -> str:
        """Generate a completion."""
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Generate a JSON completion conforming to schema."""
        json_prompt = f"""{prompt}

Respond with valid JSON conforming to this schema:
{json.dumps(schema, indent=2)}

Output only the JSON, no other text."""

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": json_prompt}],
        )

        content = response.content[0].text
        # Extract JSON from response (handle potential markdown wrapping)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())


class OllamaProvider:
    """Ollama local model provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"

    def complete(self, prompt: str, temperature: float | None = None) -> str:
        """Generate a completion using Ollama."""
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["response"]

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Generate a JSON completion."""
        json_prompt = f"""{prompt}

Respond with valid JSON conforming to this schema:
{json.dumps(schema, indent=2)}

Output only the JSON, no other text."""

        content = self.complete(json_prompt, temperature=0.0)

        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(config: LLMConfig) -> OpenAIProvider | AnthropicProvider | OllamaProvider:
        """Create an LLM provider based on configuration."""
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "ollama": OllamaProvider,
            "local": OllamaProvider,
        }

        provider_class = providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}")

        return provider_class(config)
