"""Remote LLaMA 3 client via HuggingFace Inference API.

Requires `huggingface_hub` package.
Env var:
  - HUGGINGFACEHUB_API_TOKEN: your HF access token
"""

from typing import Optional
import os

try:
    from huggingface_hub import InferenceClient

    class LLMAPIClient:
        def __init__(
            self,
            model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
            token: Optional[str] = None,
            timeout: int = 120,
        ) -> None:
            self.model_id = model_id
            self.token = token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not self.token:
                raise RuntimeError(
                    "HUGGINGFACEHUB_API_TOKEN is not set. Set the env var or pass token=..."
                )
            self.client = InferenceClient(model=self.model_id, token=self.token, timeout=timeout)

        def chat(self, user_msg: str, system_msg: str = "", max_new_tokens: int = 128) -> str:
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": user_msg})

            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )

            choice = resp.choices[0]
            content = getattr(choice.message, "content", None) or choice.message.get("content")
            return (content or "").strip()

except Exception as e:  
    class LLMAPIClient:  
        def __init__(self, *_, **__):
            raise ImportError(
                "huggingface_hub is required for LLMAPIClient. Install with `pip install huggingface_hub`."
            )

        def chat(self, *_, **__):
            raise RuntimeError("LLMAPIClient not available because required dependencies are missing.")


__all__ = ["LLMAPIClient"]
