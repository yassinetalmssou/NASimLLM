"""Approximate belief state for partially observable NASim.

NASim partial observations are non-cumulative: each step reports only the outcome
of the last action. Overlaying successive non-zero observation entries onto a
running vector reconstructs the agent's accumulated knowledge, which grows
monotonically as it scans and exploits (verified on the small scenario never to
overwrite a known value with a conflicting one). This gives a memoryless
feedforward policy an approximate belief without recurrent state.
"""
import numpy as np


class BeliefTracker:
    def __init__(self):
        self._belief = None

    def reset(self, obs):
        self._belief = np.array(obs, dtype=np.float32, copy=True)
        return self._belief.copy()

    def update(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        mask = obs != 0
        self._belief[mask] = obs[mask]
        return self._belief.copy()
