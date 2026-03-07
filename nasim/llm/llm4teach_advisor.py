import json
import logging
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

from nasim.llm.prompts import SCORE_OPTIONS_STRUCTURED

logger = logging.getLogger(__name__)


OPTION_NAMES = [
    "SCAN",          
    "EXPLOIT",       
    "PRIV_ESC",      
    "PIVOT",         
    "MOVE",          
]


class LLM4TeachAdvisor:
    def __init__(self,
                 llm_client=None,
                 use_cache: bool = True,
                 force_call: bool = False,
                 fallback_strategy: str = "uniform",
                 prompt_variant: str = "structured"):
        self.llm_client = llm_client
        self.use_cache = use_cache
        self.force_call = force_call
        self.fallback_strategy = fallback_strategy
        self.prompt_variant = prompt_variant
        
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.elapsed_llm_s = 0.0  # cumulative wall time spent on actual LLM inference
    
    def score_options(self, state_summary: str) -> np.ndarray:
        if self.use_cache and state_summary in self.cache and not self.force_call:
            logger.debug(f"[LLM4Teach Advisor] Cache hit for summary: {state_summary[:50]}...")
            self.cache_hits += 1
            return self.cache[state_summary].copy()
        
        self.cache_misses += 1
        
        if self.llm_client is not None:
            try:
                import time as _time
                _t0 = _time.perf_counter()
                scores = self._score_via_llm(state_summary)
                self.elapsed_llm_s += _time.perf_counter() - _t0
                logger.debug(f"[LLM4Teach Advisor] LLM scored options: {scores}")
            except Exception as e:
                logger.warning(f"[LLM4Teach Advisor] LLM scoring failed: {e}. Using fallback.")
                scores = self._score_fallback()
        else:
            logger.debug("[LLM4Teach Advisor] No LLM client available. Using fallback.")
            scores = self._score_fallback()
        
        probs = scores / (scores.sum() + 1e-8)
        
        if self.use_cache:
            self.cache[state_summary] = probs.copy()
        
        return probs
    
    def _score_via_llm(self, state_summary: str) -> np.ndarray:
        prompt = SCORE_OPTIONS_STRUCTURED.format(state_summary=state_summary)
        
        try:
            response = self.llm_client.chat(prompt, max_new_tokens=80)
        except (AttributeError, TypeError):
            try:
                response = self.llm_client.generate(prompt, max_tokens=80)
            except (AttributeError, TypeError):
                raise ValueError("LLM client has neither chat() nor generate() method")
        
        scores = self._parse_option_scores(response)
        
        if scores is None:
            raise ValueError(f"Failed to parse LLM response: {response}")
        
        return scores
    
    def _parse_option_scores(self, response: str) -> Optional[np.ndarray]:
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
            logger.warning("[LLM4Teach Advisor] No JSON found in response. Using heuristic.")
            return self._score_fallback()
        
        json_str = response[first_brace:last_brace + 1]
        
        # Parse JSON
        data = None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_str_fixed = json_str.replace("'", '"')
                data = json.loads(json_str_fixed)
            except json.JSONDecodeError:
                logger.warning("[LLM4Teach Advisor] JSON parse failed. Using heuristic.")
                return self._score_fallback()
        
        assert data is not None, "Failed to parse JSON data"
        
        scores = []
        for option_name in OPTION_NAMES:
            score = data.get(option_name, None)
            if score is None:
                score = data.get(option_name.lower(), None)
            if score is None:
                score = data.get(option_name.upper(), None)
            if score is None:
                score = 50.0
            
            scores.append(float(score))
        
        scores = np.array(scores, dtype=np.float32)
        scores = np.clip(scores, 0, 100)
        return scores
    
    def _score_fallback(self) -> np.ndarray:

        if self.fallback_strategy == "uniform":
            scores = np.ones(5, dtype=np.float32) * 50.0
        elif self.fallback_strategy == "action_frequency":
            scores = np.array([80, 60, 40, 30, 20], dtype=np.float32)
        else:
            scores = np.ones(5, dtype=np.float32) * 50.0
        
        return scores
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.cache),
            "elapsed_llm_s": round(self.elapsed_llm_s, 3),
        }

    def reset_timing(self):
        """Reset per-episode timing counter."""
        self.elapsed_llm_s = 0.0
    
    def clear_cache(self):
        """Clear the LLM response cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("[LLM4Teach Advisor] Cache cleared.")


def score_options_from_advisor(advisor: LLM4TeachAdvisor,
                                state_summary: str) -> np.ndarray:

    return advisor.score_options(state_summary)
