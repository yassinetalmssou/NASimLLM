"""Local Phi-3 client wrapper."""

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    class LocalPhi3MiniOptimized:
        def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
            self.has_cuda = torch.cuda.is_available()
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 if self.has_cuda else torch.float32,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto" if self.has_cuda else None,
            )
            if not self.has_cuda:
                self.model.to("cpu")

        def chat(self, user_msg: str, system_msg: str = "", max_new_tokens: int = 128) -> str:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

except Exception as e:  # pragma: no cover - missing deps
    class LocalPhi3MiniOptimized:
        def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct"):
            raise ImportError("Install torch, transformers (and optionally bitsandbytes/accelerate).")

        def chat(self, *args, **kwargs):
            raise RuntimeError("LocalPhi3MiniOptimized not available because required dependencies are missing.")


__all__ = ["LocalPhi3MiniOptimized"]
