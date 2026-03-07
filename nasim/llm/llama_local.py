
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    class LocalLlama31:
        def __init__(
            self,
            model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
            use_4bit: bool = True,
            device_map: str = None,
            torch_dtype=None,
        ):
            self.has_cuda = torch.cuda.is_available()
            if torch_dtype is None:
                torch_dtype = torch.bfloat16 if self.has_cuda else torch.float32

            quant_config = None
            if use_4bit and self.has_cuda:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            if device_map is None:
                device_map = "cuda:0" if self.has_cuda else "cpu"

            model_kwargs = {"device_map": device_map}
            if quant_config is not None:
                model_kwargs["quantization_config"] = quant_config
            else:
                model_kwargs["dtype"] = torch_dtype

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            if not self.has_cuda and quant_config is None:
                self.model.to("cpu")

        def chat(
            self,
            user_msg: str,
            system_msg: str = "",
            max_new_tokens: int = 128,
            temperature: float = 0.7,
            top_p: float = 0.9,
        ) -> str:
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": user_msg})

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

except Exception as e:  # pragma: no cover - missing deps
    class LocalLlama31:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LocalLlama31 requires torch, transformers (and optionally bitsandbytes)."
            )

        def chat(self, *args, **kwargs):
            raise RuntimeError("LocalLlama31 not available because required dependencies are missing.")


__all__ = ["LocalLlama31"]
