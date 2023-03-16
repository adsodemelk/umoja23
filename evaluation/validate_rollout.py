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
from collections import defaultdict
from typing import Dict, List

import numpy as np
import tqdm

from evaluation.seeds import PUBLIC_SEEDS
from rubiks_cube.env import RubiksCube
from rubiks_cube.utils import is_solved
from training.configs import (
    CONFIG_WEIGHTINGS,
    EASY_ENV_CONFIG,
    HARD_ENV_CONFIG,
    MEDIUM_ENV_CONFIG,
)


def validate_and_score_rollouts(
    env_configs: List[Dict],
    env_config_names: List[str],
    results_path: str,
    public_seeds: List[int],
) -> Dict[str, float]:
    """Make sure that given file represents a set of valid rollouts, and compute the final scores for public seeds"""
    results = defaultdict(lambda: defaultdict(lambda: False))
    seed = None
    expecting_new_obs = False
    expecting_updated_obs = False
    expecting_new_action = False
    experiencing_error = False
    action = None
    done = False
    num_lines = sum(1 for _ in open(results_path, "r"))
    for env_config, name in zip(env_configs, env_config_names):
        print(f"Validating rollouts on {name} difficulty...")
        env = RubiksCube(**env_config)
        with open(results_path, "r") as file:
            for line in tqdm.tqdm(file, total=num_lines):
                if line.startswith("Start"):
                    seed = int(line.split()[-1])
                    name = line.split()[-2]
                    expecting_new_obs = True
                    expecting_new_action = False
                    expecting_updated_obs = False
                    experiencing_error = False
                    done = False
                elif line.startswith("End") and not experiencing_error:
                    try:
                        end_seed = int(line.split()[-1])
                        end_name = line.split()[-2]
                        if seed == end_seed and name == end_name and done:
                            results[name][seed] = is_solved(env.cube)
                    except Exception:
                        experiencing_error = True
                        done = True
                        continue
                elif expecting_new_obs:
                    try:
                        env.set_seed(seed=seed)
                        _ = env.reset()
                        assert env.get_state() == line.strip()
                        experiencing_error = False
                        expecting_new_action = True
                        expecting_new_obs = False
                        expecting_updated_obs = False
                    except Exception:
                        experiencing_error = True
                        done = True
                        continue
                elif not experiencing_error and expecting_new_action:
                    try:
                        action = (int(line.strip()[0]), int(line.strip()[1]))
                        assert env.action_space.contains(action)
                        expecting_new_action = False
                        expecting_updated_obs = True
                        expecting_new_obs = False
                    except Exception:
                        experiencing_error = True
                        done = True
                        continue
                elif not experiencing_error and expecting_updated_obs:
                    try:
                        _, _, done, _ = env.step(action=action)
                        assert env.get_state() == line.strip()
                        experiencing_error = False
                        expecting_new_action = True
                        expecting_new_obs = False
                        expecting_updated_obs = False
                    except Exception:
                        experiencing_error = True
                        done = True
                        continue
    final_results_public = defaultdict(lambda: 0.0)
    for k, v in results.items():
        total_solved_public = 0
        for seed in public_seeds:
            if v[seed]:
                total_solved_public += 1
        final_results_public[k] = total_solved_public / len(public_seeds)
    return final_results_public


def main_validation(results_path: str, public_seeds: List[int]) -> float:
    scores_per_env_config_public = validate_and_score_rollouts(
        env_configs=[EASY_ENV_CONFIG, MEDIUM_ENV_CONFIG, HARD_ENV_CONFIG],
        env_config_names=["easy", "medium", "hard"],
        results_path=results_path,
        public_seeds=public_seeds,
    )
    total_score_public = 0
    for config_name, config_weight in CONFIG_WEIGHTINGS.items():
        total_score_public += scores_per_env_config_public[config_name] * config_weight
    total_weight = np.sum(list(CONFIG_WEIGHTINGS.values()))
    return total_score_public / total_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation and scoring")
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Local path to read results from",
    )
    args = parser.parse_args()
    public_score = main_validation(
        results_path=args.results_path,
        public_seeds=PUBLIC_SEEDS,
    )
    print(f"Public score: {public_score}")
