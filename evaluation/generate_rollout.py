# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from typing import Dict, List

from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.ppo import PPO

from evaluation.seeds import PUBLIC_SEEDS
from rubiks_cube.env import RubiksCube, create_flattened_env, dump_cube
from rubiks_cube.flat_action_wrapper import unflatten_action
from training.configs import (
    AGENT_NAME,
    CUSTOM_MODEL_CONFIG,
    EASY_ENV_CONFIG,
    FLATTEN_ACTIONS,
    HARD_ENV_CONFIG,
    MEDIUM_ENV_CONFIG,
    get_config,
)
from training.registry import _ray, register

create_env = (
    create_flattened_env
    if FLATTEN_ACTIONS
    else lambda env_config: RubiksCube(**env_config)
)


def generate_single_rollout(
    env_config: Dict,
    seed: int,
    agent: Algorithm,
    results_path: str,
    explore: bool = False,
) -> None:
    """Write rollout to file (all obs and all actions)"""
    env = create_env(env_config=env_config)
    env.set_seed(seed=seed)
    observation = env.reset()
    done = False
    with open(results_path, "a") as file:
        file.write(f"{dump_cube(observation['cube'])}\n")
    while not done:
        action = agent.compute_single_action(observation=observation, explore=explore)
        with open(results_path, "a") as file:
            if FLATTEN_ACTIONS:
                unflattened_action = unflatten_action(
                    action=action, factor_sizes=env.factor_sizes
                )
                file.write(f"{unflattened_action[0]}{unflattened_action[1]}\n")
            else:
                file.write(f"{action[0]}{action[1]}\n")
        observation, reward, done, info = env.step(action)
        with open(results_path, "a") as file:
            file.write(f"{dump_cube(observation['cube'])}\n")


def main_rollout(
    seeds: List[int],
    checkpoint_path: str,
    results_path: str,
    agent_name: str = AGENT_NAME,
) -> None:
    """Main script for generating a set of rollouts from a trained model"""
    register(agent_name=agent_name)
    with _ray():
        for env_config, name in zip(
            [EASY_ENV_CONFIG, MEDIUM_ENV_CONFIG, HARD_ENV_CONFIG],
            ["easy", "medium", "hard"],
        ):
            print(f"Generating rollouts for {name} env config...")
            config = get_config(
                env_config=env_config,
                model_config=CUSTOM_MODEL_CONFIG,
                agent_name=agent_name,
            )
            agent = PPO(AlgorithmConfig.from_dict(config))
            agent.restore(checkpoint_path)
            for seed in seeds:
                with open(results_path, "a") as file:
                    file.write(f"Start {name} {seed}\n")
                generate_single_rollout(
                    env_config=env_config,
                    seed=seed,
                    agent=agent,
                    results_path=results_path,
                    explore=False,
                )
                with open(results_path, "a") as file:
                    file.write(f"End {name} {seed}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout")
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Local path to checkpoint"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Local path to write the results to",
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default="PPO",
        help="Name of agent (default PPO)",
    )
    args = parser.parse_args()
    main_rollout(
        seeds=PUBLIC_SEEDS,
        checkpoint_path=args.checkpoint_path,
        results_path=args.results_path,
        agent_name=args.agent_name,
    )
