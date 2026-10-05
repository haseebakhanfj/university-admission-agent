from typing import Any, Optional

from crewai.llm import BaseLLM
from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)


class GroqLLM(BaseLLM):
    """
    CrewAI-compatible LLM wrapper for the Groq Python SDK.

    This keeps the application dependent on:
    - CrewAI for agent orchestration
    - Groq for model inference
    """

    def __init__(
        self,
        model: str = GROQ_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            **kwargs,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = Groq(
            api_key=GROQ_API_KEY,
        )

    def call(
        self,
        messages: Any,
        tools: Optional[list] = None,
        callbacks: Optional[list] = None,
        available_functions: Optional[dict] = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        """
        Send messages to Groq and return the generated text.
        """

        if not isinstance(messages, list):
            messages = [
                {
                    "role": "user",
                    "content": str(messages),
                }
            ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return completion.choices[0].message.content
