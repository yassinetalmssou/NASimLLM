"""LLMAdvisor: score actions and cache results."""
from typing import List, Tuple, Dict
import json
import hashlib

from .LLM_API import LLMAPIClient


class LLMAdvisor:
    def __init__(self, client: LLMAPIClient = None):
        self.client = client
        self._cache: Dict[str, Dict[str, float]] = {}
        # logging flags
        self.log_calls = False
        self.force_call = False
        self.log_cache = False

    def _ensure_client(self):
        if self.client is None:
            self.client = LLMAPIClient()


    def _cache_key(self, summary: str, candidates: List[Tuple[str, str]]) -> str:
        key_src = summary + "|" + ";".join(f"{cid}:{desc}" for cid, desc in candidates)
        return hashlib.sha256(key_src.encode("utf-8")).hexdigest()

    def score_actions(self, summary: str, candidates: List[Tuple[str, str]]) -> Dict[str, float]:
        """Ask the LLM to score the provided candidates.

        `candidates` is a list of (id, short_description) tuples. The returned
        mapping maps id -> float score in [0,1]. On any parsing error the
        method returns uniform scores.
        """
        key = self._cache_key(summary, candidates)
        if not self.force_call and key in self._cache:
            if self.log_cache or self.log_calls:
                print(f"LLMAdvisor: cache hit {key[:8]}")
            return self._cache[key]

        try:
            self._ensure_client()
            candidate_lines = "\n".join(f"{cid}: {desc}" for cid, desc in candidates)
            prompt = (
                "You are an advisor that scores candidate actions for a cybersecurity agent. "
                "Return strict JSON with a top-level `scores` mapping from id to a numeric score between 0 and 1, "
                "and an optional `rationale` string. Use only numeric scores in the mapping.\n\n"
                "Context summary:\n" + summary + "\n\n"
                "Candidates:\n" + candidate_lines + "\n\n"
                "Return JSON only. Example: {\"scores\": {\"17\": 0.62, \"42\": 0.21}}"
            )

            if self.log_calls:
                print(f"LLMAdvisor: calling LLM for {len(candidates)} candidates")
            raw = self.client.chat(user_msg=prompt, system_msg="You are a strict JSON-only responder.", max_new_tokens=256)

            if self.log_calls:
                try:
                    print("LLMAdvisor: raw:", raw)
                except Exception:
                    print("LLMAdvisor: raw response (unprintable)")

            start = raw.find('{')
            end = raw.rfind('}')
            if start >= 0 and end > start:
                raw_json = raw[start:end+1]
            else:
                raw_json = raw

            try:
                parsed = json.loads(raw_json)
            except Exception:
                try:
                    alt = raw_json.replace("'", '"')
                    parsed = json.loads(alt)
                except Exception:
                    parsed = {}

            scores = parsed.get("scores", {}) if isinstance(parsed, dict) else {}

            out = {}
            for cid, _ in candidates:
                val = None
                if isinstance(scores, dict):
                    for key_try in (str(cid), cid, int(cid) if cid.isdigit() else None):
                        try:
                            if key_try is None:
                                continue
                            if key_try in scores:
                                val = scores[key_try]
                                break
                        except Exception:
                            continue
                try:
                    f = float(val)
                    f = max(0.0, min(1.0, f))
                except Exception:
                    f = 0.0
                out[str(cid)] = f

            if all(v == 0.0 for v in out.values()):
                n = len(candidates) or 1
                heuristic = {str(candidates[i][0]): float(n - i) / float(n) for i in range(n)}
                out = heuristic

            if self.log_calls:
                print(f"LLMAdvisor: scores: {out}")
            self._cache[key] = out
            return out
        except Exception:
            n = len(candidates) or 1
            uniform = {str(cid): 1.0 / n for cid, _ in candidates}
            if self.log_calls:
                print(f"LLMAdvisor: fallback uniform: {uniform}")
            self._cache[key] = uniform
            return uniform

    def summarize_state_nl(self, env, state, max_new_tokens: int = 150) -> str:
        """Short natural-language description of the state."""
        try:
            stext = self._summarize_for_cache(env, state)
            cache_key = "nl:" + hashlib.sha256(stext.encode("utf-8")).hexdigest()
            if cache_key in self._cache and not self.force_call:
                if self.log_cache:
                    print(f"LLMAdvisor: NL summary cache hit {cache_key[:8]}")
                return self._cache[cache_key]
        except Exception:
            stext = None

        try:
            desc = env.get_state_description() if hasattr(env, "get_state_description") else None
        except Exception:
            desc = None

        prompt_ctx = desc if desc else (stext if stext else f"obs={str(state)[:500]}")

        if self.client is None:
            nl = prompt_ctx if isinstance(prompt_ctx, str) else str(prompt_ctx)
            try:
                self._cache[cache_key] = nl
            except Exception:
                pass
            return nl

        try:
            prompt = (
                "You are an assistant that summarizes a network environment's current state in one concise paragraph. "
                "Given the context, produce a short updated status in natural language (2-4 sentences). "
                "Do not include suggestions or actions — only describe the current observed state.\n\n"
                "Context:\n" + prompt_ctx + "\n\n"
                "Return ONLY the natural-language summary (no JSON)."
            )

            if self.log_calls:
                print("LLMAdvisor: NL summary from LLM")

            raw = self.client.chat(user_msg=prompt, system_msg="", max_new_tokens=max_new_tokens)
            if self.log_calls:
                print("LLMAdvisor: raw NL response:", raw)

            nl = " ".join(raw.strip().split())
            try:
                self._cache[cache_key] = nl
            except Exception:
                pass
            return nl
        except Exception as e:
            if self.log_calls:
                print("LLMAdvisor: NL summarization failed:", e)
            nl = prompt_ctx if isinstance(prompt_ctx, str) else str(prompt_ctx)
            try:
                self._cache[cache_key] = nl
            except Exception:
                pass
            return nl

    def _summarize_for_cache(self, env, state) -> str:
        try:
            desc = env.get_state_description() if hasattr(env, "get_state_description") else None
        except Exception:
            desc = None
        if desc:
            return str(desc)[:1000]
        return str(state)[:1000]


__all__ = ["LLMAdvisor"]
