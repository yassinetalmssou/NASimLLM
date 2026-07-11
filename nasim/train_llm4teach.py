import argparse
import csv
import json
import logging
import time
import numpy as np
import random
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple, TextIO

import nasim
from nasim.agents.llm4teach_agent import LLM4TeachAgent, action_to_option_id
from nasim.llm.llm4teach_advisor import LLM4TeachAdvisor
from nasim.envs.observation import Observation
from prettytable import PrettyTable

try:
    from nasim.llm.llama_local import LocalLlama31
    LLAMA_LOCAL_AVAILABLE = True
except ImportError:
    LLAMA_LOCAL_AVAILABLE = False


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s"
)


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Best-effort reproducibility across Python/NumPy/PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LLM4TeachTrainer:    
    def __init__(self,
                 scenario: str = "tiny",
                 seed: int = 0,
                 num_episodes: int = 100,
                 episode_length: int = 1000,
                 lambda_start: float = 1.0,
                 lambda_decay: float = 0.99,
                 hidden_dim: int = 128,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 batch_size: int = 32,
                 num_epochs: int = 3,
                 device: str = "cpu",
                 save_dir: Optional[str] = None,
                 step_log_dir: Optional[str] = "./llm4teach_step_logs",
                 ckpt_every: int = 0,
                 use_llm: bool = True,
                 force_llm: bool = False,
                 prompt_variant: str = "structured",
                 verbose: bool = True,
                 llama_model: str = "llm_weights/Qwen3-4B",
                 use_local_llama: bool = True,
                 llama_use_4bit: bool = True,
                 fully_obs: bool = True,
                 deterministic_torch: bool = False,
                 lambda_mode: str = "fixed",
                 kl_target: float = 0.01,
                 lambda_min: float = 0.005,
                 competence_window: int = 10,
                 llm_mix_weight: float = 0.3,
                 teacher_temp: float = 2.0,
                 csv_log: Optional[str] = None,
                 use_history_mechanism: bool = True,
                 use_avoidlist: bool = True,
                 use_compact_prompt: bool = True,
                 llm_call_frequency: str = "every_step",
                 condition: str = ""):
        
        self.scenario_name = scenario
        self.seed = int(seed)
        self.fully_obs = fully_obs
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.lambda_start = lambda_start
        self.lambda_decay = lambda_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None
        self.step_log_dir = Path(step_log_dir) if step_log_dir else None
        # 0 = only the final checkpoint (disk-frugal); >0 = also every N episodes.
        self.ckpt_every = int(ckpt_every)
        self.use_llm = use_llm
        self.prompt_variant = prompt_variant
        self.verbose = verbose
        self.llama_model = llama_model
        self.use_local_llama = use_local_llama
        self.llama_use_4bit = llama_use_4bit

        self.lambda_mode = lambda_mode
        self.kl_target = kl_target
        self.lambda_min = lambda_min
        self.competence_window = competence_window
        self.current_lambda = lambda_start
        self.llm_mix_weight = llm_mix_weight
        self.teacher_temp = teacher_temp
        
        self.use_history_mechanism = use_history_mechanism
        self.use_avoidlist = use_avoidlist
        self.use_compact_prompt = use_compact_prompt
        self.llm_call_frequency = llm_call_frequency
        # Resolve condition label for config.json (auto-detect if blank)
        if condition:
            self.condition = condition
        else:
            self.condition = "llm_full" if use_llm else "ppo_options"

        if csv_log:
            self.csv_log_path = Path(csv_log)
        elif self.save_dir:
            self.csv_log_path = Path(save_dir).parent / "train.csv"
        else:
            self.csv_log_path = None
        self._csv_file = None
        self._csv_writer = None

        set_global_seed(self.seed, deterministic_torch=deterministic_torch)
        
        logger.info(f"Loading scenario: {scenario} (fully_obs={fully_obs}, seed={self.seed})")
        self.env = nasim.make_benchmark(
            scenario,
            seed=self.seed,
            fully_obs=fully_obs,
            flat_actions=True,
            flat_obs=True
        )
        self.action_list = self.env.action_space.actions
        
        self.total_sensitive_hosts = len(self.env.network.sensitive_addresses)
        self.total_hosts = len(self.env.network.address_space)
        
        obs, _ = self.env.reset()
        self.state_dim = obs.shape[0]
        self.agent = LLM4TeachAgent(
            state_dim=self.state_dim,
            action_list=self.action_list,
            hidden_dim=hidden_dim,
            num_options=5,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        
        self.llm_advisor = None  # default; set below if LLM is enabled
        if self.use_llm:
            llm_client = None
            
            if self.use_local_llama:
                if LLAMA_LOCAL_AVAILABLE:
                    model_path = self.llama_model
                    if not Path(model_path).is_absolute():
                        model_path = str(Path(__file__).parent.parent / model_path)
                    
                    logger.info(
                        f"Attempting to initialize local Llama client "
                        f"(model={model_path}, 4bit={self.llama_use_4bit})..."
                    )
                    try:
                        llm_client = LocalLlama31(
                            model_id=model_path,
                            use_4bit=self.llama_use_4bit,
                        )
                        logger.info("Local Llama client initialized successfully.")
                    except Exception as e:
                        logger.error(f"Failed to initialize local Llama: {e}")
                        import traceback
                        logger.error(f"Full traceback:\n{traceback.format_exc()}")
                else:
                    logger.warning("Local Llama not available. Install torch, transformers, bitsandbytes.")
            
            if llm_client is None:
                logger.warning("No LLM client available. LLM teacher disabled.")

            if llm_client is None:
                self.use_llm = False
                self.llm_advisor = None
            else:
                self.llm_advisor = LLM4TeachAdvisor(
                    llm_client=llm_client,
                    use_cache=True,
                    force_call=force_llm,
                    use_history=self.use_history_mechanism,
                    use_avoidlist=self.use_avoidlist,
                    use_compact_prompt=self.use_compact_prompt,
                )
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoints will be saved to: {self.save_dir}")
        if self.step_log_dir:
            self.step_log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Per-step logs will be saved to: {self.step_log_dir}")
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.lambda_values = []
        self.metrics_history = []
        
        self.detailed_metrics = {
            'success_rate': [],
            'hosts_compromised': [],
            'sensitive_compromised': [],
            'discovery_rate': [],
            'compromise_rate': [],
            'error_counts': [],
            'option_distribution': [],
        }

    def _format_readable_observation(self, host_obs, aux_obs) -> str:
        """Build a text table similar to nasim.demo render_readable()."""
        aux_headers = list(aux_obs.keys())
        aux_table = PrettyTable(aux_headers)
        aux_table.add_row([str(aux_obs[k]) for k in aux_headers])

        host_table_str = "No hosts"
        if host_obs:
            host_headers = list(host_obs[0].keys())
            host_table = PrettyTable(host_headers)
            for h in host_obs:
                host_table.add_row([str(h[k]) for k in host_headers])
            host_table_str = str(host_table)

        return f"{aux_table}\n\n{host_table_str}"

    def summarize_state(self, state: np.ndarray):

        state_shape = self.env.current_state.shape()
        num_hosts = state_shape[0]
        features_per_host = state_shape[1]

        expected_obs_rows = num_hosts + 1
        expected_total_size = expected_obs_rows * features_per_host

        assert state.size == expected_total_size, (
            f"Observation size mismatch: got {state.size}, "
            f"expected {expected_total_size} "
            f"(({num_hosts}+1) hosts x {features_per_host} features)"
        )

        if len(state.shape) == 1:
            state_2d = state.reshape(expected_obs_rows, features_per_host)
        else:
            state_2d = state

        obs_obj = Observation.from_numpy(state_2d, state_shape)
        host_obs, aux_obs = obs_obj.get_readable()

        total_hosts = len(host_obs)
        
        sensitive_targets = []
        compromised_hosts = []
        available_targets = []
        unexplored = []
        
        subnet_status = {}
        subnet_hosts = {}
        
        for idx, h in enumerate(host_obs):
            addr = self.env.network.address_space[idx]
            
            if not self.fully_obs and not h['Discovered']:
                unexplored.append(addr)
                continue
            
            subnet_id = addr[0] if isinstance(addr, tuple) else None
            if subnet_id not in subnet_status:
                subnet_status[subnet_id] = {
                    'total': 0, 
                    'compromised': 0, 
                    'sensitive': 0,
                    'reachable': 0
                }
                subnet_hosts[subnet_id] = {
                    'compromised': [],
                    'sensitive': [],
                    'reachable': [],
                    'unreachable': []
                }
            subnet_status[subnet_id]['total'] += 1
            
            services = [k for k, v in h.items() if k not in {
                "Address", "Compromised", "Reachable", "Discovered",
                "Sensitive", "Discovery Value", "Access"
            } and bool(v)]
            
            host_desc = f"subnet {subnet_id}, host {addr[1]}" if isinstance(addr, tuple) else str(addr)
            
            if h['Compromised']:
                access_level = "ROOT" if h['Access'] >= 2 else "USER"
                compromised_hosts.append(f"({host_desc}) - {access_level} access")
                subnet_status[subnet_id]['compromised'] += 1
                subnet_hosts[subnet_id]['compromised'].append((addr, access_level, services))
            elif h['Sensitive']:
                status = "REACHABLE" if h['Reachable'] else "NOT REACHABLE"
                sensitive_targets.append(f"({host_desc}) [SENSITIVE TARGET, {status}, services: {services if services else 'none'}]")
                subnet_status[subnet_id]['sensitive'] += 1
                if h['Reachable']:
                    subnet_status[subnet_id]['reachable'] += 1
                    subnet_hosts[subnet_id]['sensitive'].append((addr, True, services))
                else:
                    subnet_hosts[subnet_id]['sensitive'].append((addr, False, services))
            elif h['Reachable'] and not h['Compromised']:
                available_targets.append(f"({host_desc}) [services: {services if services else 'none'}]")
                subnet_status[subnet_id]['reachable'] += 1
                subnet_hosts[subnet_id]['reachable'].append((addr, services))
            else:
                subnet_hosts[subnet_id]['unreachable'].append((addr, services))
        
        lines = []
        
        num_compromised = len(compromised_hosts)
        num_sensitive_remaining = len(sensitive_targets)
        total_sensitive = sum(status['sensitive'] for status in subnet_status.values())
        lines.append(f"PROGRESS: {num_compromised}/{total_hosts} hosts compromised | {total_sensitive - num_sensitive_remaining}/{total_sensitive} sensitive targets secured")
        
        lines.append("")
        lines.append("=== NETWORK TOPOLOGY ===")
        for subnet_id in sorted(subnet_status.keys()):
            status = subnet_status[subnet_id]
            hosts = subnet_hosts[subnet_id]
            
            if status['compromised'] == status['total']:
                subnet_line = f"SUBNET {subnet_id}: FULLY CONTROLLED ({status['total']} hosts)"
            elif status['compromised'] > 0:
                subnet_line = f"SUBNET {subnet_id}: PARTIALLY CONTROLLED ({status['compromised']}/{status['total']} hosts)"
            else:
                subnet_line = f"SUBNET {subnet_id}: NOT COMPROMISED ({status['total']} hosts)"
            
            if status['sensitive'] > 0:
                subnet_line += f" - Contains {status['sensitive']} SENSITIVE TARGET(S)"
            
            lines.append(subnet_line)
            
            if hosts['compromised']:
                for addr, access, svcs in hosts['compromised']:
                    lines.append(f"  [C] {addr}: COMPROMISED ({access} access) - services: {svcs if svcs else 'none'}")
            
            if hosts['sensitive']:
                for addr, reachable, svcs in hosts['sensitive']:
                    reach_status = "REACHABLE" if reachable else "NOT REACHABLE (need pivot)"
                    lines.append(f"  [S] {addr}: SENSITIVE TARGET - {reach_status} - services: {svcs if svcs else 'none'}")
            
            if hosts['reachable']:
                for addr, svcs in hosts['reachable'][:2]: 
                    lines.append(f"  [R] {addr}: reachable - services: {svcs if svcs else 'none'}")
                if len(hosts['reachable']) > 2:
                    lines.append(f"      + {len(hosts['reachable']) - 2} more reachable hosts")
            
            if hosts['unreachable']:
                lines.append(f"  - {len(hosts['unreachable'])} host(s) not yet reachable (need pivot/scan)")
        
        lines.append("")
        
        if num_sensitive_remaining > 0:
            lines.append(f"=== MISSION STATUS: {num_sensitive_remaining} SENSITIVE TARGET(S) REMAINING ===")
            for target in sensitive_targets[:3]:  
                lines.append(f"  > {target}")
            if len(sensitive_targets) > 3:
                lines.append(f"  > + {len(sensitive_targets) - 3} more")
        else:
            lines.append("=== MISSION STATUS: ALL SENSITIVE TARGETS SECURED ===")
        
        lines.append("")
        
        lines.append("")
        
        if not aux_obs['Success']:
            if aux_obs['Connection Error']:
                lines.append("LAST ACTION: Failed (connection error - target not reachable, need to pivot via compromised host)")
            elif aux_obs['Permission Error']:
                lines.append("LAST ACTION: Failed (permission denied - need privilege escalation)")
            else:
                lines.append("LAST ACTION: Failed (exploit/action unsuccessful - try different approach)")
        else:
            lines.append("LAST ACTION: Success")
        
        lines.append("")
        if num_sensitive_remaining > 0:
            unreachable_subnets = []
            for subnet_id, hosts in subnet_hosts.items():
                for addr, reachable, _ in hosts['sensitive']:
                    if not reachable:
                        unreachable_subnets.append(subnet_id)
                        break
            
            if unreachable_subnets:
                lines.append(f"NEXT PRIORITY: Pivot to subnet(s) {unreachable_subnets} to reach {num_sensitive_remaining} sensitive target(s)")
            else:
                lines.append(f"NEXT PRIORITY: Exploit {num_sensitive_remaining} reachable sensitive target(s) to complete mission")
        elif num_compromised < total_hosts:
            lines.append("NEXT PRIORITY: Complete network compromise (all hosts)")
        else:
            lines.append("MISSION COMPLETE: All hosts compromised")

        return "\n".join(lines), host_obs, aux_obs

    _OS_ABBREV = {"linux": "lx", "windows": "win", "macos": "mac"}
    _HOST_META = {"Address", "Compromised", "Reachable", "Discovered",
                  "Sensitive", "Discovery Value", "Access"}

    def compact_state_summary(self, host_obs: list, aux_obs: dict) -> str:
        """Compact ~40-token state representation for the LLM teacher.

        Format (generics across all NASim scenarios)::

            0c/8h | 0/2s | FAIL-conn
            S1(1h): (1,0)R[http,ssh]
            S2*(1h): (2,0)S:!R[ssh,ftp,tomcat]
            S3(5h): 5?
            S4*(1h): (4,0)S:!R[ssh,ftp,tomcat]

        Legend (also in system prompt):
          Xc/Yh  = X of Y hosts compromised
          X/Ys   = X of Y sensitive targets secured
          OK / FAIL / FAIL-conn / FAIL-perm  = last action result
          S1*(Nh) = subnet 1, N hosts, * = contains sensitive target
          C:ROOT / C:USER  = compromised (access level)
          R[svcs] = reachable host,  !R[svcs] = NOT reachable (need pivot)
          S:R / S:!R = sensitive target, reachable / not reachable
          N?   = N unreachable/undiscovered hosts
        """
        if host_obs is None or aux_obs is None:
            return "OBS_ERR"

        if not aux_obs.get('Success', True):
            if aux_obs.get('Connection Error', False):
                la = "FAIL-conn"
            elif aux_obs.get('Permission Error', False):
                la = "FAIL-perm"
            else:
                la = "FAIL"
        else:
            la = "OK"

        subnet_data: dict = {}
        num_compromised = 0
        total_sensitive = 0
        sensitive_remaining = 0
        total_hosts = len(host_obs)

        for idx, h in enumerate(host_obs):
            addr = self.env.network.address_space[idx]
            sid = addr[0] if isinstance(addr, tuple) else 0
            hid = addr[1] if isinstance(addr, tuple) else idx

            if not self.fully_obs and not h.get('Discovered', False):
                if sid not in subnet_data:
                    subnet_data[sid] = {'total': 0, 'sensitive': 0, 'entries': []}
                subnet_data[sid]['total'] += 1
                subnet_data[sid]['entries'].append('?')
                continue

            if sid not in subnet_data:
                subnet_data[sid] = {'total': 0, 'sensitive': 0, 'entries': []}
            subnet_data[sid]['total'] += 1

            svcs = [
                self._OS_ABBREV.get(k, k)
                for k, v in h.items()
                if k not in self._HOST_META and bool(v)
            ]
            svc_str = "[" + ",".join(svcs) + "]" if svcs else ""
            tag = f"({sid},{hid})"

            if h.get('Compromised', False):
                num_compromised += 1
                acc = "ROOT" if h.get('Access', 0) >= 2 else "USER"
                subnet_data[sid]['entries'].append(f"{tag}C:{acc}{svc_str}")
            elif h.get('Sensitive', False):
                total_sensitive += 1
                sensitive_remaining += 1
                subnet_data[sid]['sensitive'] += 1
                reach = "R" if h.get('Reachable', False) else "!R"
                subnet_data[sid]['entries'].append(f"{tag}S:{reach}{svc_str}")
            elif h.get('Reachable', False):
                subnet_data[sid]['entries'].append(f"{tag}R{svc_str}")
            else:
                subnet_data[sid]['entries'].append(f"{tag}?")

        sensitive_secured = total_sensitive - sensitive_remaining
        lines = [f"{num_compromised}c/{total_hosts}h | {sensitive_secured}/{total_sensitive}s | {la}"]

        for sid in sorted(subnet_data.keys()):
            d = subnet_data[sid]
            star = "*" if d['sensitive'] > 0 else ""
            visible = [e for e in d['entries'] if not e.endswith('?') and e != '?']
            n_unknown = sum(1 for e in d['entries'] if e.endswith('?') or e == '?')
            parts = visible[:]
            if n_unknown:
                parts.append(f"{n_unknown}?")
            lines.append(f"S{sid}{star}({d['total']}h): {' '.join(parts) if parts else '?'}")

        return "\n".join(lines)

    def collect_rollout(self, max_steps: int = 1000, log_fh: Optional[TextIO] = None) -> Tuple[
        List[np.ndarray], List[int], List[int], np.ndarray,
        np.ndarray, np.ndarray
    ]:
        obs, _ = self.env.reset()
        
        states = []
        option_ids = []
        actions = []
        rewards = []
        dones = []
        teacher_probs_list = []
        
        done = False
        step = 0
        episode_reward = 0

        if self.use_llm and self.llm_advisor is not None:
            self.llm_advisor.reset_episode()

        visited_states = set()
        discovered_subnets = set()
        compromised_subnets = set()
        action_history = []
        host_failure_counts = {}

        prev_option_id: int = None
        prev_action_desc: str = None

        while not done and step < max_steps:
            state_summary, host_obs_readable, aux_obs_readable = self.summarize_state(obs)
            
            should_call_llm = False
            if self.use_llm and self.llm_advisor is not None:
                if self.llm_call_frequency == "every_step":
                    should_call_llm = True
                elif self.llm_call_frequency == "cached":
                    should_call_llm = True
                elif self.llm_call_frequency == "reduced":
                    should_call_llm = (step % 5 == 0)
            
            if should_call_llm:
                if self.use_compact_prompt:
                    llm_input = self.compact_state_summary(host_obs_readable, aux_obs_readable)
                else:
                    llm_input = state_summary
                
                teacher_probs = self.llm_advisor.score_options(
                    llm_input,
                    last_option_id=prev_option_id,
                    last_action_desc=prev_action_desc,
                )
            else:
                teacher_probs = np.ones(5, dtype=np.float32) / 5.0

            mix_weight = float(np.clip(self.current_lambda * self.llm_mix_weight, 0.0, 1.0))
            option_id = self.agent.choose_option(
                obs, training=True,
                teacher_probs=teacher_probs if self.use_llm else None,
                mix_weight=mix_weight,
            )
            
            candidates = list(range(len(self.action_list)))
            action_idx = self.agent.select_action_from_option(obs, option_id, candidates)

            prev_option_id = option_id
            prev_action_desc = str(self.action_list[action_idx])

            next_obs, reward, done, truncated, info = self.env.step(action_idx)

            episode_end = bool(done or truncated)
            
            shaped_reward = reward
            
            action_obj = self.action_list[action_idx]
            target_host = getattr(action_obj, 'target', None)
            
            if reward == -1:
                connection_err = info.get('connection_error', False)
                if connection_err:
                    shaped_reward = -0.2
                else:
                    shaped_reward = -0.3
                    
                    if target_host is not None:
                        host_failure_counts[target_host] = host_failure_counts.get(target_host, 0) + 1
                        if host_failure_counts[target_host] > 3:
                            shaped_reward -= min(0.5, (host_failure_counts[target_host] - 3) * 0.1)
            
            elif reward >= 0:
                try:
                    _, curr_hosts, _ = self.summarize_state(obs)
                    _, next_hosts, _ = self.summarize_state(next_obs)

                    def get_subnet(host_addr):
                        if isinstance(host_addr, tuple) and len(host_addr) >= 2:
                            return host_addr[0]
                        return None

                    if curr_hosts and next_hosts:
                        newly_affected_host = None
                        sensitive_multiplier = 1.0

                        curr_discovered = sum(1 for h in curr_hosts if h.get('Discovered', False))
                        next_discovered = sum(1 for h in next_hosts if h.get('Discovered', False))
                        if next_discovered > curr_discovered:
                            num_new = next_discovered - curr_discovered
                            shaped_reward += 4 * num_new
                            shaped_reward += 2 * num_new

                            for i, (curr_h, next_h) in enumerate(zip(curr_hosts, next_hosts)):
                                if not curr_h.get('Discovered', False) and next_h.get('Discovered', False):
                                    newly_affected_host = i
                                    break

                        curr_compromised = sum(1 for h in curr_hosts if h.get('Compromised', False))
                        next_compromised = sum(1 for h in next_hosts if h.get('Compromised', False))
                        if next_compromised > curr_compromised:
                            num_new_comp = next_compromised - curr_compromised
                            shaped_reward += 8 * num_new_comp

                            for i, (curr_h, next_h) in enumerate(zip(curr_hosts, next_hosts)):
                                if not curr_h.get('Compromised', False) and next_h.get('Compromised', False):
                                    newly_affected_host = i
                                    if next_h.get('Sensitive', False):
                                        sensitive_multiplier = 1.5
                                    break

                        curr_access = sum(h.get('Access', 0) for h in curr_hosts)
                        next_access = sum(h.get('Access', 0) for h in next_hosts)
                        if next_access > curr_access:
                            shaped_reward += 6 * (next_access - curr_access)

                        if newly_affected_host is not None:
                            host_addr = self.env.network.address_space[newly_affected_host]
                            subnet = get_subnet(host_addr)

                            if subnet is not None:
                                if next_discovered > curr_discovered and subnet not in discovered_subnets:
                                    shaped_reward += 10
                                    discovered_subnets.add(subnet)

                                if next_compromised > curr_compromised and subnet not in compromised_subnets:
                                    shaped_reward += 16
                                    compromised_subnets.add(subnet)

                                subnet_hosts = [i for i, addr in enumerate(self.env.network.address_space)
                                               if get_subnet(addr) == subnet]
                                if subnet_hosts:
                                    all_comp = all(next_hosts[i].get('Compromised', False) for i in subnet_hosts)
                                    any_not_comp_before = any(not curr_hosts[i].get('Compromised', False) for i in subnet_hosts)
                                    if all_comp and any_not_comp_before:
                                        shaped_reward += 12

                        if len(action_history) > 0:
                            for past_step, past_option, past_target in action_history[-5:]:
                                if (past_option == 0 and option_id == 1 and
                                    past_target is not None and target_host == past_target and
                                    next_compromised > curr_compromised):
                                    shaped_reward += 6

                                if (past_option == 1 and option_id == 2 and
                                    past_target is not None and target_host == past_target and
                                    next_access > curr_access):
                                    shaped_reward += 8

                        if option_id == 4 and newly_affected_host is not None:
                            new_subnet = get_subnet(self.env.network.address_space[newly_affected_host])
                            if new_subnet is not None and new_subnet not in discovered_subnets:
                                shaped_reward += 14

                        if sensitive_multiplier > 1.0:
                            shaped_reward = reward + (shaped_reward - reward) * sensitive_multiplier

                except Exception as e:
                    logger.debug(f"Reward shaping failed: {e}")
            
            state_hash = hash(obs.tobytes())
            if state_hash not in visited_states:
                visited_states.add(state_hash)
                shaped_reward += 1
            
            action_history.append((step, option_id, target_host))
            
            states.append(obs.copy())
            option_ids.append(option_id)
            actions.append(action_idx)
            rewards.append(shaped_reward)
            dones.append(1.0 if episode_end else 0.0)
            teacher_probs_list.append(teacher_probs)
            
            episode_reward += shaped_reward
            
            if log_fh is not None:
                action_desc = str(self.action_list[action_idx]) if self.action_list else str(action_idx)
                probs_fmt = ", ".join(f"{p:.3f}" for p in teacher_probs)
                readable_tables = ""
                if host_obs_readable is not None and aux_obs_readable is not None:
                    readable_tables = self._format_readable_observation(
                        host_obs_readable, aux_obs_readable
                    )
                else:
                    readable_tables = "Observation render unavailable."

                log_fh.write(
                    f"Step {step + 1}\n"
                    f"Summary: {state_summary}\n"
                    f"Action (option {option_id}, idx {action_idx}): {action_desc}\n"
                    f"Observation:\n{readable_tables}\n"
                    f"Reward={reward:.3f}\n"
                    f"Done={done}\n"
                    f"Step limit reached={truncated}\n"
                    f"Teacher probs: [{probs_fmt}]\n"
                    f"{'-'*60}\n\n"
                )

            obs = next_obs
            step += 1

            if episode_end:
                done = True
        
        rewards_arr = np.array(rewards, dtype=np.float32)
        dones_arr = np.array(dones, dtype=np.float32)
        teacher_probs_arr = np.array(teacher_probs_list, dtype=np.float32)
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(step)

        # Use env.goal_reached() for a reliable success signal independent of shaped rewards
        self._collect_episode_metrics(states, option_ids, actions, rewards_arr,
                                       self.env.goal_reached())
        
        return states, option_ids, actions, rewards_arr, dones_arr, teacher_probs_arr
    
    def _collect_episode_metrics(self, states, option_ids, actions, rewards, success):
        try:
            if states:
                _, final_hosts, final_aux = self.summarize_state(states[-1])

                if final_hosts:
                    num_hosts = len(final_hosts)
                    num_compromised = sum(1 for h in final_hosts if h.get('Compromised', False))
                    num_discovered = sum(1 for h in final_hosts if h.get('Discovered', False))
                    num_sensitive_comp = sum(1 for h in final_hosts
                                            if h.get('Compromised', False) and h.get('Sensitive', False))

                    self.detailed_metrics['hosts_compromised'].append(num_compromised)
                    self.detailed_metrics['discovery_rate'].append(num_discovered / max(num_hosts, 1))
                    self.detailed_metrics['compromise_rate'].append(num_compromised / max(num_hosts, 1))
                    self.detailed_metrics['sensitive_compromised'].append(num_sensitive_comp)

            self.detailed_metrics['success_rate'].append(1 if success else 0)

            error_counts = {
                'total_failures': int((rewards == -1).sum()) if len(rewards) > 0 else 0,
                'total_successes': int((rewards > 0).sum()) if len(rewards) > 0 else 0,
            }
            self.detailed_metrics['error_counts'].append(error_counts)

            if option_ids:
                option_dist = {i: option_ids.count(i) for i in range(5)}
                self.detailed_metrics['option_distribution'].append(option_dist)

        except Exception as e:
            logger.debug(f"Metrics collection failed: {e}")
    
    def train_episode(self, episode_idx: int):
        if self.lambda_mode == "fixed":
            lambda_kl = self.lambda_start * (self.lambda_decay ** episode_idx)
            self.current_lambda = lambda_kl
        else:
            lambda_kl = self.current_lambda
        self.lambda_values.append(lambda_kl)
        
        log_fh = None
        if self.step_log_dir:
            log_fh = (self.step_log_dir / f"episode_{episode_idx + 1}.txt").open("w", encoding="utf-8")
            log_fh.write(
                f"Scenario={self.scenario_name} | Episode={episode_idx + 1}/{self.num_episodes} | "
                f"lambda={lambda_kl:.4f} | use_llm={self.use_llm}\n\n"
            )
        
        try:
            t0_rollout = time.perf_counter()
            states, option_ids, actions, rewards, dones, teacher_probs = \
                self.collect_rollout(max_steps=self.episode_length, log_fh=log_fh)
            t_rollout = time.perf_counter() - t0_rollout
        finally:
            if log_fh is not None:
                log_fh.close()

        if self.use_llm and teacher_probs is not None and len(teacher_probs) > 0:
            llm_top = np.argmax(teacher_probs, axis=1)
            teacher_agree = float(np.mean(np.array(option_ids) == llm_top))
        else:
            teacher_agree = float('nan')
        
        t0_ppo = time.perf_counter()
        metrics = self.agent.update_from_rollout(
            states=states,
            option_ids=option_ids,
            actions=actions,
            rewards=rewards,
            dones=dones,
            teacher_option_probs=teacher_probs if self.use_llm else None,
            lambda_kl=lambda_kl,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            teacher_temp=self.teacher_temp,
        )
        t_ppo = time.perf_counter() - t0_ppo

        if self.use_llm and self.llm_advisor is not None:
            t_llm = self.llm_advisor.elapsed_llm_s
            self.llm_advisor.reset_timing()
            cache_stats = self.llm_advisor.get_cache_stats()
            ep_cache_hits = cache_stats['cache_hits']
            ep_cache_misses = cache_stats['cache_misses']
        else:
            t_llm = 0.0
            ep_cache_hits = 0
            ep_cache_misses = 0

        t_episode = t_rollout + t_ppo

        self.metrics_history.append(metrics)

        if self.lambda_mode == "target-kl":
            actual_kl = metrics.get('loss_kl', 0.0)
            if actual_kl > 0 and self.kl_target > 0:
                ratio = actual_kl / self.kl_target
                kl_adjusted = float(np.clip(
                    self.current_lambda * ratio,
                    self.lambda_min,
                    self.lambda_start,
                ))
            else:
                kl_adjusted = self.current_lambda
        else:
            kl_adjusted = self.current_lambda

        if self.lambda_mode in ("competence", "adaptive"):
            cr_hist = self.detailed_metrics.get('compromise_rate', [])
            if cr_hist:
                window = min(self.competence_window, len(cr_hist))
                competence = float(np.mean(cr_hist[-window:]))
            else:
                competence = 0.0
            comp_ceiling = float(np.clip(
                self.lambda_start * (1.0 - competence),
                self.lambda_min,
                self.lambda_start,
            ))
        else:
            comp_ceiling = self.current_lambda

        if self.lambda_mode == "target-kl":
            self.current_lambda = kl_adjusted
        elif self.lambda_mode == "competence":
            self.current_lambda = comp_ceiling
        elif self.lambda_mode == "adaptive":
            self.current_lambda = min(kl_adjusted, comp_ceiling)

        if self._csv_writer is not None:
            avg_r = float(np.mean(self.episode_rewards[-10:])) if len(self.episode_rewards) >= 10 else float(np.mean(self.episode_rewards))
            cr_list = self.detailed_metrics.get('compromise_rate', [])
            dr_list = self.detailed_metrics.get('discovery_rate', [])
            sr_list = self.detailed_metrics.get('success_rate', [])
            hc_list = self.detailed_metrics.get('hosts_compromised', [])
            
            guidance_metrics = {}
            if self.llm_advisor:
                guidance_metrics = self.llm_advisor.get_guidance_metrics()
            
            self._csv_writer.writerow({
                'episode':           episode_idx + 1,
                'reward':            round(self.episode_rewards[-1], 3),
                'avg_reward_10':     round(avg_r, 3),
                'length':            self.episode_lengths[-1],
                'lambda_kl':         round(lambda_kl, 6),
                'next_lambda':       round(self.current_lambda, 6),
                'hosts_compromised': hc_list[-1] if hc_list else '',
                'hosts_total':       self.total_hosts,
                'compromise_rate':   round(cr_list[-1], 4) if cr_list else '',
                'discovery_pct':     round(dr_list[-1] * 100, 1) if dr_list else '',
                'loss_actor':        round(metrics.get('loss_actor', 0.0), 4),
                'loss_critic':       round(metrics.get('loss_critic', 0.0), 2),
                'loss_kl_norm':      round(metrics.get('loss_kl_norm', 0.0), 4),
                'clip_pct':          round(metrics.get('clip_fraction', 0.0) * 100, 2),
                'teacher_agree':     round(teacher_agree * 100, 1) if not np.isnan(teacher_agree) else '',
                'success':           int(sr_list[-1]) if sr_list else '',
                't_rollout_s':       round(t_rollout, 2),
                't_ppo_s':           round(t_ppo, 2),
                't_llm_s':           round(t_llm, 2),
                't_episode_s':       round(t_episode, 2),
                'cache_hits':        ep_cache_hits,
                'cache_misses':      ep_cache_misses,
                'avoidlist_used':    guidance_metrics.get('avoidlist_used', 0),
            })
            self._csv_file.flush()

        if self.verbose:
            if (episode_idx + 1) % 10 == 0 or episode_idx < 5:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths)
                
                recent = min(10, len(self.detailed_metrics['success_rate']))
                if recent > 0:
                    success_rate = np.mean(self.detailed_metrics['success_rate'][-recent:])
                    curr_sensitive_comp = self.detailed_metrics['sensitive_compromised'][-1] if self.detailed_metrics['sensitive_compromised'] else 0
                    curr_discovery = self.detailed_metrics['discovery_rate'][-1] if self.detailed_metrics['discovery_rate'] else 0
                    
                    logger.info(
                        f"Episode {episode_idx + 1}/{self.num_episodes} | "
                        f"Reward: {self.episode_rewards[-1]:.2f} (avg: {avg_reward:.2f}) | "
                        f"Length: {self.episode_lengths[-1]:.0f} (avg: {avg_length:.1f}) | "
                        f"λ: {lambda_kl:.4f}\n"
                        f"  Hosts compromised: {curr_sensitive_comp:.0f}/{self.total_sensitive_hosts} | "
                        f"Discovery: {curr_discovery:.1%} | "
                        f"Loss (actor/kl): {metrics['loss_actor']:.3f} / {metrics['loss_kl_norm']:.3f} | "
                        f"Critic: {metrics['loss_critic']:.1f} | "
                        f"Teacher agree: {teacher_agree:.0%} | "
                        f"Clip: {metrics.get('clip_fraction', 0):.2%}\n"
                        f"  Time: rollout={t_rollout:.1f}s | ppo={t_ppo:.2f}s | llm={t_llm:.1f}s | "
                        f"cache {ep_cache_hits}H/{ep_cache_misses}M"
                    )
                else:
                    logger.info(
                        f"Episode {episode_idx + 1}/{self.num_episodes} | "
                        f"Reward: {self.episode_rewards[-1]:.2f} (avg: {avg_reward:.2f}) | "
                        f"Length: {self.episode_lengths[-1]:.0f} (avg: {avg_length:.1f}) | "
                        f"λ: {lambda_kl:.4f} | "
                        f"Loss (actor/kl): {metrics['loss_actor']:.4f} / {metrics['loss_kl_norm']:.4f}"
                    )
        
        if self.save_dir and self.ckpt_every > 0 and (episode_idx + 1) % self.ckpt_every == 0:
            ckpt_path = self.save_dir / f"ckpt_episode_{episode_idx + 1}.pt"
            self.agent.save_checkpoint(str(ckpt_path))
            logger.info(f"Saved checkpoint: {ckpt_path}")
    
    def train(self):
        """Run full training loop."""
        logger.info("=" * 60)
        logger.info("Starting LLM4Teach Training")
        logger.info("=" * 60)
        logger.info(f"Scenario: {self.scenario_name}")
        logger.info(f"Episodes: {self.num_episodes}")
        logger.info(f"Episode length: {self.episode_length}")
        logger.info(f"Fully observable: {self.fully_obs}" if self.fully_obs else f"Partial observability: {not self.fully_obs}")
        if self.lambda_mode == "fixed":
            logger.info(f"\u03bb annealing: {self.lambda_start} \u2192 {self.lambda_start * (self.lambda_decay ** self.num_episodes):.4f} (decay={self.lambda_decay})")
        else:
            logger.info(f"\u03bb mode: {self.lambda_mode} | start={self.lambda_start} | min={self.lambda_min} | kl_target={self.kl_target}")
        logger.info(f"LLM teacher: {self.use_llm} (prompt={self.prompt_variant})")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60)

        _CSV_FIELDS = [
            'episode', 'reward', 'avg_reward_10', 'length',
            'lambda_kl', 'next_lambda',
            'hosts_compromised', 'hosts_total', 'compromise_rate', 'discovery_pct',
            'loss_actor', 'loss_critic', 'loss_kl_norm', 'clip_pct', 'teacher_agree', 'success',
            't_rollout_s', 't_ppo_s', 't_llm_s', 't_episode_s', 'cache_hits', 'cache_misses',
            'avoidlist_used',
        ]
        if self.csv_log_path:
            self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self.csv_log_path, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDS)
            self._csv_writer.writeheader()
            logger.info(f"CSV log: {self.csv_log_path}")

        if self.save_dir:
            config_path = self.save_dir.parent / "config.json"
            config = {
                "scenario": self.scenario_name,
                "seed": self.seed,
                "num_episodes": self.num_episodes,
                "episode_length": self.episode_length,
                "fully_obs": self.fully_obs,
                "use_llm": self.use_llm,
                "prompt_variant": self.prompt_variant,
                "lambda_start": self.lambda_start,
                "lambda_decay": self.lambda_decay,
                "lambda_mode": self.lambda_mode,
                "kl_target": self.kl_target,
                "lambda_min": self.lambda_min,
                "competence_window": self.competence_window,
                "teacher_temp": self.teacher_temp,
                "hidden_dim": self.agent.student.hidden_dim,
                "condition": self.condition,
                "use_history_mechanism": self.use_history_mechanism,
                "use_avoidlist": self.use_avoidlist,
                "use_compact_prompt": self.use_compact_prompt,
                "llm_call_frequency": self.llm_call_frequency,
                "learning_rate": self.agent.optimizer.param_groups[0]['lr'],
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "device": self.device,
            }
            config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
            logger.info(f"Config: {config_path}")

        try:
            for episode_idx in range(self.num_episodes):
                self.train_episode(episode_idx)
        finally:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Final episode reward: {self.episode_rewards[-1]:.2f}")
        logger.info(f"Average reward (last 10 eps): {np.mean(self.episode_rewards[-10:]):.2f}")
        logger.info(f"Max reward: {np.max(self.episode_rewards):.2f}")
        
        if self.use_llm and self.llm_advisor:
            cache_stats = self.llm_advisor.get_cache_stats()
            logger.info(f"LLM cache stats: {cache_stats}")
        
        if self.detailed_metrics['sensitive_compromised']:
            avg_sensitive_comp = np.mean(self.detailed_metrics['sensitive_compromised'])
            logger.info(f"Average hosts compromised: {avg_sensitive_comp:.2f}/{self.total_sensitive_hosts}")
        if self.detailed_metrics['discovery_rate']:
            avg_discovery = np.mean(self.detailed_metrics['discovery_rate'])
            logger.info(f"Average discovery rate: {avg_discovery:.1%}")
        
        if self.save_dir:
            final_ckpt = self.save_dir / "ckpt_final.pt"
            self.agent.save_checkpoint(str(final_ckpt))
            logger.info(f"Saved final checkpoint: {final_ckpt}")
    
    def evaluate(self, num_episodes: int = 5, render: bool = False) -> float:
        eval_rewards = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0
            step = 0
            
            while not done and step < self.episode_length:
                option_id = self.agent.choose_option(obs, training=False)
                candidates = list(range(len(self.action_list)))
                action_idx = self.agent.select_action_from_option(obs, option_id, candidates)
                
                obs, reward, done, truncated, info = self.env.step(action_idx)
                ep_reward += reward
                step += 1
            
            eval_rewards.append(ep_reward)
            logger.info(f"Eval episode {ep + 1}: reward = {ep_reward:.2f}")
        
        avg_reward = np.mean(eval_rewards)
        logger.info(f"Average eval reward: {avg_reward:.2f}")
        return avg_reward


def main():
    parser = argparse.ArgumentParser(
        description="Train LLM4Teach agent on NASim environment"
    )
    parser.add_argument("--scenario", type=str, default="tiny",
                        help="NASim scenario name (default: tiny)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes (default: 100)")
    parser.add_argument("--episode-length", type=int, default=500,
                        help="Max steps per episode (default: 500)")
    parser.add_argument("--lambda-start", type=float, default=1.0,
                        help="Initial KL weight (default: 1.0)")
    parser.add_argument("--lambda-decay", type=float, default=0.99,
                        help="Lambda decay per episode (default: 0.99)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden layer dimension (default: 128)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Adam learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for SGD (default: 32)")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="PPO epochs per rollout (default: 3)")
    parser.add_argument("--save-dir", type=str, default="./llm4teach_ckpts",
                        help="Directory to save checkpoints (default: ./llm4teach_ckpts)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM teacher (train with PPO only)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for RL agent: cpu or cuda (default: cpu; LLM always uses GPU)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation instead of training")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of eval episodes (default: 5)")
    parser.add_argument("--llama-model", type=str, default="llm_weights/Qwen3-4B",
                        help="Local Llama model path (default: llm_weights/Qwen3-4B)")
    parser.add_argument("--no-llama-4bit", action="store_false", dest="llama_4bit",
                        help="Disable 4-bit quantization (default: enabled on GPU)")
    parser.set_defaults(llama_4bit=True)
    parser.add_argument("--no-step-logs", action="store_true",
                        help="Disable per-step logs (saves disk space)")
    parser.add_argument("--fully-obs", action="store_true", default=True,
                        help="Fully observable mode [default]")
    parser.add_argument("--partial-obs", action="store_false", dest="fully_obs",
                        help="Use partial observability")
    parser.add_argument(
        "--lambda-mode", type=str, default="fixed",
        choices=["fixed", "target-kl", "competence", "adaptive"],
        help="Lambda scheduling mode (default: fixed exponential decay)",
    )
    parser.add_argument("--kl-target", type=float, default=0.01,
                        help="Target KL for target-kl/adaptive modes (default: 0.01)")
    parser.add_argument("--lambda-min", type=float, default=0.005,
                        help="Floor for adaptive lambda (default: 0.005)")
    parser.add_argument("--competence-window", type=int, default=10,
                        help="Rolling window for competence-based lambda (default: 10)")
    parser.add_argument("--llm-mix-weight", type=float, default=0.3,
                        help="Max fraction of LLM in option sampling (0=pure student, 1=pure LLM). Decays with lambda. Default: 0.3")
    parser.add_argument("--teacher-temp", type=float, default=2.0,
                        help="Temperature for softening teacher distribution in KL loss (1.0=no softening, 2.0=default)")
    
    parser.add_argument("--condition", type=str, default="",
                        help="Condition label written to config.json (auto-detected from --no-llm if blank)")
    parser.add_argument("--no-history", action="store_false", dest="use_history",
                        help="Disable history mechanism (old behavior: no action log)")
    parser.add_argument("--no-avoidlist", action="store_false", dest="use_avoidlist",
                        help="Disable avoidlist (don't track failed action pairs)")
    parser.add_argument("--verbose-prompt", action="store_false", dest="use_compact_prompt",
                        help="Use verbose prompts instead of compact nomad-style")
    parser.add_argument("--llm-call-freq", type=str, default="every_step",
                        choices=["every_step", "cached", "reduced"],
                        help="LLM call frequency (default: every_step)")
    parser.add_argument("--ckpt-every", type=int, default=0,
                        help="Save an intermediate checkpoint every N episodes "
                             "(0 = only the final checkpoint; keeps disk usage low)")
    parser.set_defaults(use_history=True, use_avoidlist=True, use_compact_prompt=True)

    args = parser.parse_args()

    use_llm = not args.no_llm
    step_log_dir = None if args.no_step_logs else str(Path(args.save_dir).parent / "step_logs")
    if use_llm:
        logger.info(f"LLM teacher: {args.llama_model} (4bit={args.llama_4bit})")
    
    trainer = LLM4TeachTrainer(
        scenario=args.scenario,
        seed=args.seed,
        num_episodes=args.episodes,
        episode_length=args.episode_length,
        lambda_start=args.lambda_start,
        lambda_decay=args.lambda_decay,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.save_dir,
        step_log_dir=step_log_dir,
        ckpt_every=args.ckpt_every,
        use_llm=use_llm,
        force_llm=False,
        prompt_variant="structured",
        verbose=True,
        llama_model=args.llama_model,
        use_local_llama=True,
        llama_use_4bit=args.llama_4bit,
        fully_obs=args.fully_obs,
        deterministic_torch=False,
        lambda_mode=args.lambda_mode,
        kl_target=args.kl_target,
        lambda_min=args.lambda_min,
        competence_window=args.competence_window,
        llm_mix_weight=args.llm_mix_weight,
        teacher_temp=args.teacher_temp,
        csv_log=None,
        use_history_mechanism=args.use_history,
        use_avoidlist=args.use_avoidlist,
        use_compact_prompt=args.use_compact_prompt,
        llm_call_frequency=args.llm_call_freq,
        condition=args.condition,
    )
    
    if args.eval:
        logger.info(f"Running evaluation ({args.eval_episodes} episodes)...")
        trainer.evaluate(num_episodes=args.eval_episodes)
    else:
        logger.info("Starting training...")
        trainer.train()


if __name__ == "__main__":
    main()
