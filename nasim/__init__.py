import gymnasium as gym
from gymnasium.envs.registration import register

from nasim.envs import NASimEnv
from nasim.stochastic_envs import StochNASimEnv
from nasim.scenarios.benchmark import AVAIL_BENCHMARKS
from nasim.scenarios import \
    make_benchmark_scenario, load_scenario, generate_scenario


__all__ = ['make_benchmark', 'load', 'generate']


def make_benchmark(scenario_name,
                   seed=None,
                   fully_obs=False,
                   flat_actions=True,
                   flat_obs=True,
                   render_mode=None):

    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs,
                  "render_mode": render_mode}
    scenario = make_benchmark_scenario(scenario_name, seed)
    return NASimEnv(scenario, **env_kwargs)


def load(path,
         fully_obs=False,
         flat_actions=True,
         flat_obs=True,
         name=None,
         render_mode=None):
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs,
                  "render_mode": render_mode}
    scenario = load_scenario(path, name=name)
    return NASimEnv(scenario, **env_kwargs)


def generate(num_hosts,
             num_services,
             fully_obs=False,
             flat_actions=True,
             flat_obs=True,
             render_mode=None,
             **params):

    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs,
                  "render_mode": render_mode}
    scenario = generate_scenario(num_hosts, num_services, **params)
    return NASimEnv(scenario, **env_kwargs)


def _register(id, entry_point, kwargs, nondeterministic, force=True):

    if id in gym.envs.registry:
        if not force:
            return
        del gym.envs.registry[id]
    register(
        id=id,
        entry_point=entry_point,
        kwargs=kwargs,
        nondeterministic=nondeterministic
    )


for benchmark in AVAIL_BENCHMARKS:
    for fully_obs in [True, False]:
        name = ''.join([g.capitalize() for g in benchmark.split("-")])
        if not fully_obs:
            name = f"{name}PO"

        _register(
            id=f"{name}-v0",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": True,
                "flat_obs": True
            },
            nondeterministic=True
        )

        _register(
            id=f"{name}2D-v0",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": True,
                "flat_obs": False
            },
            nondeterministic=True
        )

        _register(
            id=f"{name}VA-v0",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": False,
                "flat_obs": True
            },
            nondeterministic=True
        )

        _register(
            id=f"{name}2DVA-v0",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": False,
                "flat_obs": False
            },
            nondeterministic=True
        )

# Regsiter NASimGenEnv
for fully_obs in [True, False]:
    if not fully_obs:
        name = "StochPO"
    else:
        name = "Stoch"
    _register(
        id=f"{name}-v0",
        entry_point='nasim.stochastic_envs:StochNASimEnv',
        kwargs={
            "fully_obs": fully_obs,
            "flat_actions": True,
            "flat_obs": True
        },
        nondeterministic=True
    )

    _register(
        id=f"{name}2D-v0",
        entry_point='nasim.stochastic_envs:StochNASimEnv',
        kwargs={
            "fully_obs": fully_obs,
            "flat_actions": True,
            "flat_obs": False
        },
        nondeterministic=True
    )

__version__ = "0.14.0"
