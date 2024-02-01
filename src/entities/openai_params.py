from dataclasses import dataclass


@dataclass()
class OpenAIParams:
    model: str
    max_tokens: int
    temperature: float
    top_p: float
