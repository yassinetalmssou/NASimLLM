import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional


OPTION_ID_MAP = {
    0: "SCAN",
    1: "EXPLOIT",
    2: "PRIV_ESC",
    3: "PIVOT",
    4: "MOVE",
}

OPTION_TO_ID = {v: k for k, v in OPTION_ID_MAP.items()}


def action_to_option_id(action_obj, action_list: List) -> int:
    if action_obj.is_scan():
        return OPTION_TO_ID["SCAN"]
    elif action_obj.is_exploit():
        return OPTION_TO_ID["EXPLOIT"]
    elif action_obj.is_privilege_escalation():
        return OPTION_TO_ID["PRIV_ESC"]
    elif action_obj.is_noop():
        return OPTION_TO_ID["PIVOT"]
    else:
        return OPTION_TO_ID["MOVE"]


class SharedPPONetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 128,
                 num_options: int = 5,
                 num_hidden_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_options = num_options
        self.num_hidden_layers = num_hidden_layers
        
        input_dim = state_dim
        layers = []
        prev_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, num_options)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        hidden = self.feature_extractor(state)
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        
        if squeeze_output:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
        
        return policy_logits, value


class LLM4TeachAgent:
    def __init__(self,
                 state_dim: int,
                 action_list: List,
                 hidden_dim: int = 128,
                 num_options: int = 5,
                 device: str = "cpu",
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        self.state_dim = state_dim
        self.action_list = action_list
        self.num_options = num_options
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.student = SharedPPONetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_options=num_options
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        self.global_step = 0
    
    def choose_option(self, state: np.ndarray, training: bool = False,
                      teacher_probs: Optional[np.ndarray] = None,
                      mix_weight: float = 0.0) -> int:
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            logits, _ = self.student.forward(state_t)
            student_probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze()

        if teacher_probs is not None and mix_weight > 0.0:
            mixed = (1.0 - mix_weight) * student_probs + mix_weight * teacher_probs
            mixed = np.clip(mixed, 1e-8, 1.0)
            mixed /= mixed.sum()
            sampling_probs = mixed
        else:
            sampling_probs = student_probs

        if training:
            option = int(np.random.choice(len(sampling_probs), p=sampling_probs))
        else:
            option = int(np.argmax(sampling_probs))
        return option
    
    def _parse_state_for_filtering(self, state: np.ndarray) -> Dict:
        try:
            num_hosts = len(self.action_list[0].target) if hasattr(self.action_list[0].target, '__len__') else 3
            
            host_info = {}
            features_per_host = len(state) // (num_hosts + 1)
            
            for h_idx in range(num_hosts):
                start = h_idx * features_per_host
                end = start + features_per_host
                host_vec = state[start:end]
                
                host_info[h_idx] = {
                    'reachable': host_vec.sum() > 0.1,
                    'compromised': host_vec.sum() > 2.0,
                    'access': int(host_vec[6]) if len(host_vec) > 6 else 0
                }
            
            return host_info
        except Exception:

            return {i: {'reachable': True, 'compromised': False, 'access': 0} for i in range(3)}
    
    def _filter_valid_actions(self, candidates: List[int], state: np.ndarray) -> List[int]:
        host_info = self._parse_state_for_filtering(state)
        valid = []
        
        for a_idx in candidates:
            if a_idx >= len(self.action_list):
                continue
            
            action = self.action_list[a_idx]
            
            if hasattr(action, 'target') and action.target is not None:
                if isinstance(action.target, tuple) and len(action.target) >= 2:
                    target_host = action.target[1]
                else:
                    target_host = 0
            else:
                target_host = 0
            
            if target_host not in host_info:
                continue
            
            host = host_info[target_host]
            
            if action.is_scan() or action.is_exploit():
                if not host['reachable']:
                    continue
            
            if action.is_privilege_escalation():
                if not host['compromised']:
                    continue
            
            valid.append(a_idx)
        
        return valid if valid else candidates
    
    def select_action_from_option(self, state: np.ndarray,
                                  option_id: int,
                                  candidates: Optional[List[int]] = None
                                  ) -> int:
        if candidates is None:
            candidates = list(range(len(self.action_list)))
        
        valid_candidates = self._filter_valid_actions(candidates, state)

        if option_id in (OPTION_TO_ID["PIVOT"], OPTION_TO_ID["MOVE"]):
            has_exploit = any(
                (idx < len(self.action_list) and self.action_list[idx].is_exploit())
                for idx in valid_candidates
            )
            option_id = OPTION_TO_ID["EXPLOIT"] if has_exploit else OPTION_TO_ID["SCAN"]
        
        matching_actions = []
        for idx in valid_candidates:
            if idx < len(self.action_list):
                action = self.action_list[idx]
                if action_to_option_id(action, self.action_list) == option_id:
                    matching_actions.append(idx)
        
        if not matching_actions:
            return valid_candidates[0] if valid_candidates else (candidates[0] if candidates else 0)
        
        if len(matching_actions) == 1:
            return matching_actions[0]
        
        state_t = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            logits, _ = self.student.forward(state_t)
            probs = F.softmax(logits, dim=-1)
            
            option_conf = probs[option_id].item()
            if option_conf > 0.5:
                return matching_actions[0]
            else:
                return np.random.choice(matching_actions)
    
    def compute_gae(self, rewards: np.ndarray,
                    values: np.ndarray,
                    dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:len(rewards)]
        return advantages, returns
    
    def update_from_rollout(self, states: List[np.ndarray],
                            option_ids: List[int],
                            actions: List[int],
                            rewards: np.ndarray,
                            dones: np.ndarray,
                            teacher_option_probs: Optional[np.ndarray] = None,
                            lambda_kl: float = 0.1,
                            num_epochs: int = 3,
                            batch_size: int = 32,
                            teacher_temp: float = 2.0) -> Dict[str, float]:
        with torch.no_grad():
            values_list = []
            for state in states:
                state_t = torch.from_numpy(state).float().to(self.device)
                _, value = self.student.forward(state_t)
                values_list.append(value.squeeze().cpu().numpy())
            values = np.array(values_list, dtype=np.float32)

            if len(dones) > 0 and float(dones[-1]) < 0.5:
                last_state_t = torch.from_numpy(states[-1]).float().to(self.device)
                _, last_v = self.student.forward(last_state_t)
                final_value = np.array([float(last_v.squeeze().cpu().numpy())], dtype=np.float32)
            else:
                final_value = np.array([0.0], dtype=np.float32)
            values_full = np.concatenate([values, final_value])
        
        advantages, returns = self.compute_gae(rewards, values_full, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states_t = torch.from_numpy(np.array(states)).float().to(self.device)
        option_ids_t = torch.tensor(option_ids, dtype=torch.long, device=self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)
        advantages_t = torch.from_numpy(advantages).float().to(self.device)
        
        if teacher_option_probs is not None:
            teacher_probs_t = torch.from_numpy(teacher_option_probs).float().to(self.device)
        
        with torch.no_grad():
            old_logits, _ = self.student.forward(states_t)
            old_probs = F.softmax(old_logits, dim=-1)
            old_dist = torch.distributions.Categorical(old_probs)
            old_logprobs = old_dist.log_prob(option_ids_t)
        
        num_samples = len(states)
        metrics = {"loss_actor": [], "loss_critic": [], "loss_kl": [], "loss_kl_norm": [], "loss_total": [], "clip_fraction": []}
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                
                batch_states = states_t[batch_indices]
                batch_option_ids = option_ids_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                
                logits_batch, values_batch = self.student.forward(batch_states)
                values_batch = values_batch.squeeze(1)
                
                probs = F.softmax(logits_batch, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logprobs = dist.log_prob(batch_option_ids)
                entropy = dist.entropy()
                
                ratio = torch.exp(logprobs - batch_old_logprobs)
                
                clip_eps = 0.2
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                with torch.no_grad():
                    clip_fraction = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean().item()
                
                critic_loss = F.mse_loss(values_batch, batch_returns)
                
                loss_kl = torch.tensor(0.0, device=self.device)
                loss_kl_norm = torch.tensor(0.0, device=self.device)
                if teacher_option_probs is not None:
                    batch_teacher_probs = teacher_probs_t[batch_indices]
                    T = max(teacher_temp, 1e-3)
                    soft_teacher = batch_teacher_probs.pow(1.0 / T)
                    soft_teacher = soft_teacher / (soft_teacher.sum(dim=-1, keepdim=True) + 1e-8)
                    loss_kl = (soft_teacher * (
                        torch.log(soft_teacher + 1e-8) -
                        torch.log(probs + 1e-8)
                    )).sum(dim=-1).mean()
                    loss_kl_norm = loss_kl / np.log(self.student.num_options)

                loss_total = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean() + lambda_kl * loss_kl_norm
                
                self.optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                metrics["loss_actor"].append(actor_loss.item())
                metrics["loss_critic"].append(critic_loss.item())
                metrics["loss_kl"].append(loss_kl.item())
                metrics["loss_kl_norm"].append(loss_kl_norm.item())
                metrics["loss_total"].append(loss_total.item())
                metrics["clip_fraction"].append(clip_fraction)
        
        return {
            "loss_actor":    np.mean(metrics["loss_actor"])    if metrics["loss_actor"]    else 0.0,
            "loss_critic":   np.mean(metrics["loss_critic"])   if metrics["loss_critic"]   else 0.0,
            "loss_kl":       np.mean(metrics["loss_kl"])       if metrics["loss_kl"]       else 0.0,
            "loss_kl_norm":  np.mean(metrics["loss_kl_norm"])  if metrics["loss_kl_norm"]  else 0.0,
            "loss_total":    np.mean(metrics["loss_total"])    if metrics["loss_total"]    else 0.0,
            "clip_fraction": np.mean(metrics["clip_fraction"]) if metrics["clip_fraction"] else 0.0,
        }
    
    def save_checkpoint(self, filepath: str):
        torch.save({
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
