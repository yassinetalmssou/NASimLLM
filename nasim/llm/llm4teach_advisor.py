"""LLM4Teach advisor: scores high-level options from a network-state summary via an LLM, with caching and history."""

import json
import logging
import re
from collections import deque
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

from nasim.llm.prompts import SCORE_OPTIONS_SYSTEM, SCORE_OPTIONS_USER

logger = logging.getLogger(__name__)


OPTION_NAMES = [
    "SCAN",
    "EXPLOIT",
    "PRIV_ESC",
    "PIVOT",
    "MOVE",
]

_TARGET_RE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")


class LLM4TeachAdvisor:
    def __init__(self,
                 llm_client=None,
                 use_cache: bool = True,
                 force_call: bool = False,
                 fallback_strategy: str = "uniform",
                 prompt_variant: str = "structured",
                 use_history: bool = True,
                 use_avoidlist: bool = True,
                 use_compact_prompt: bool = True):
        self.llm_client = llm_client
        self.use_cache = use_cache
        self.force_call = force_call
        self.fallback_strategy = fallback_strategy
        self.prompt_variant = prompt_variant
        
        self.use_history = use_history
        self.use_avoidlist = use_avoidlist
        self.use_compact_prompt = use_compact_prompt
        
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.elapsed_llm_s = 0.0
        self._ep_hits_start = 0
        self._ep_misses_start = 0
        
        self.guidance_metrics = {
            'valid_suggestions': 0,
            'invalid_suggestions': 0,
            'total_suggestions': 0,
            'error_types': {
                'invalid_action': 0,
                'unreachable_host': 0,
                'wrong_phase': 0,
                'hallucination': 0,
            },
            'avoidlist_used': 0,
        }
        
        self._action_log: deque = deque(maxlen=8)
        self._failure_counts: dict = {}
        
        self._cache_key_prefix = self._generate_cache_key_prefix()

    def _generate_cache_key_prefix(self) -> str:
        """Generate unique cache key prefix based on ablation configuration."""
        config_str = f"h{int(self.use_history)}_a{int(self.use_avoidlist)}_c{int(self.use_compact_prompt)}"
        return config_str
    
    def _get_cache_key(self, state_summary: str) -> str:
        """Generate cache key that includes ablation configuration."""
        return f"{self._cache_key_prefix}::{state_summary}"

    @staticmethod
    def _extract_target(action_desc: str) -> str:
        """Extract host address from action description, e.g. 'e_ssh(1, 0)' -> '(1,0)'."""
        m = _TARGET_RE.search(action_desc)
        if m:
            return m.group(0).replace(" ", "")
        return "?"

    @staticmethod
    def _parse_success(state_summary: str) -> bool:
        """Read LAST ACTION line from the *next* state to determine if prev action succeeded."""
        for line in state_summary.splitlines():
            if line.startswith("LAST ACTION:"):
                return "Success" in line
        return True

    def _build_history(self) -> str:
        """Build compact action history: AVOID (repeated failures) + RECENT action log."""
        if not self.use_history:
            return ""
        
        parts = []

        if self.use_avoidlist:
            blocked = sorted(
                [(k, v) for k, v in self._failure_counts.items() if v >= 2],
                key=lambda x: -x[1]
            )[:6]
            if blocked:
                avoid_str = " ".join(f"{opt}@{tgt}\u00d7{cnt}" for (opt, tgt), cnt in blocked)
                parts.append(f"AVOID: {avoid_str}")
                self.guidance_metrics['avoidlist_used'] += 1

        log = list(self._action_log)
        if log:
            log_str = " | ".join(
                f"{opt}({tgt})\u2192" + ("\u2713" if ok else "\u2717") for opt, tgt, ok in log
            )
            parts.append(f"RECENT: {log_str}")

        return "\n".join(parts) + "\n\n" if parts else ""

    def reset_episode(self) -> None:
        """Clear per-episode action memory and snapshot cache counters for delta logging."""
        self._action_log.clear()
        self._failure_counts.clear()
        self._ep_hits_start = self.cache_hits
        self._ep_misses_start = self.cache_misses
        self.guidance_metrics['avoidlist_used'] = 0
    
    def get_guidance_metrics(self) -> Dict[str, Any]:
        """Get RQ4 guidance quality metrics."""
        total = self.guidance_metrics['total_suggestions']
        if total == 0:
            return {
                'valid_suggestion_rate': 0.0,
                'invalid_suggestion_rate': 0.0,
                'error_distribution': {},
                'total_suggestions': 0,
                'avoidlist_used': self.guidance_metrics['avoidlist_used']
            }
        
        return {
            'valid_suggestion_rate': self.guidance_metrics['valid_suggestions'] / total,
            'invalid_suggestion_rate': self.guidance_metrics['invalid_suggestions'] / total,
            'error_distribution': self.guidance_metrics['error_types'].copy(),
            'total_suggestions': total,
            'avoidlist_used': self.guidance_metrics['avoidlist_used']
        }

    def score_options(
        self,
        state_summary: str,
        last_option_id: int = None,
        last_action_desc: str = None,
    ) -> np.ndarray:
        if last_option_id is not None and last_action_desc is not None:
            success = self._parse_success(state_summary)
            option_name = OPTION_NAMES[last_option_id] if 0 <= last_option_id < len(OPTION_NAMES) else "?"
            target = self._extract_target(last_action_desc)
            self._action_log.append((option_name, target, success))
            if not success:
                key = (option_name, target)
                self._failure_counts[key] = self._failure_counts.get(key, 0) + 1

        cache_key = self._get_cache_key(state_summary)
        if self.use_cache and cache_key in self.cache and not self.force_call:
            self.cache_hits += 1
            return self.cache[cache_key].copy()

        self.cache_misses += 1

        if self.llm_client is not None:
            try:
                import time as _time
                _t0 = _time.perf_counter()
                scores = self._score_via_llm(state_summary)
                self.elapsed_llm_s += _time.perf_counter() - _t0
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}. Using fallback.")
                scores = self._score_fallback()
        else:
            scores = self._score_fallback()

        probs = scores / (scores.sum() + 1e-8)

        if self.use_cache:
            self.cache[cache_key] = probs.copy()

        return probs

    def _score_via_llm(self, state_summary: str) -> np.ndarray:
        history = self._build_history()
        user_msg = SCORE_OPTIONS_USER.format(history=history, state_summary=state_summary)

        try:
            response = self.llm_client.chat(user_msg, system_msg=SCORE_OPTIONS_SYSTEM, max_new_tokens=60)
        except (AttributeError, TypeError):
            try:
                response = self.llm_client.generate(user_msg, max_tokens=60)
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
            logger.debug("LLM response contained no JSON. Using fallback.")
            return self._score_fallback()
        
        json_str = response[first_brace:last_brace + 1]
        
        data = None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_str_fixed = json_str.replace("'", '"')
                data = json.loads(json_str_fixed)
            except json.JSONDecodeError:
                logger.debug("LLM response JSON parse failed. Using fallback.")
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
            "cache_hits": self.cache_hits - self._ep_hits_start,
            "cache_misses": self.cache_misses - self._ep_misses_start,
            "cache_hits_total": self.cache_hits,
            "cache_misses_total": self.cache_misses,
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


def score_options_from_advisor(advisor: LLM4TeachAdvisor,
                                state_summary: str) -> np.ndarray:

    return advisor.score_options(state_summary)
