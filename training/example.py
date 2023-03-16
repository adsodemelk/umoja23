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
from typing import Dict

from ray import tune

from training.configs import AGENT_NAME, CUSTOM_MODEL_CONFIG, get_config
from training.registry import _ray, register


def main_training(
    step_limit: int,
    reward_function_type: str,
    num_scrambles_on_reset: int,
    model_config: Dict,
    agent_name: str,
    num_iterations: int,
    restore_path: str,
) -> None:
    """Main training script that generates a checkpoint"""
    env_config = {
        "step_limit": step_limit,
        "reward_function_type": reward_function_type,
        "num_scrambles_on_reset": num_scrambles_on_reset,
    }
    config = get_config(
        env_config=env_config,
        model_config=model_config,
        agent_name=agent_name,
    )
    register(agent_name=agent_name)
    with _ray():
        results = tune.run(
            agent_name,
            config=config,
            stop={"training_iteration": num_iterations},
            checkpoint_freq=1,
            checkpoint_at_end=True,
            keep_checkpoints_num=5,
            restore=restore_path,
        )
    print(f"Results are saved to path {list(results.trial_dataframes.keys())[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--step_limit",
        type=int,
        default=10,
        help="Maximum length of an episode allowed",
    )
    parser.add_argument(
        "--reward_function_type",
        type=str,
        default="sparse",
        help="Rewards given for transitions - by default 1 if cube is solved, else 0",
    )
    parser.add_argument(
        "--num_scrambles_on_reset",
        type=int,
        default=1,
        help="Number of scrambles applied to cube whenever the env is reset",
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default=AGENT_NAME,
        help="Name of agent (default PPO)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of iterations to train for (default 1)",
    )
    parser.add_argument(
        "--restore_path",
        type=str,
        required=False,
        default=None,
        help="Path to restore from (if given)",
    )

    args = parser.parse_args()
    main_training(
        step_limit=args.step_limit,
        reward_function_type=args.reward_function_type,
        num_scrambles_on_reset=args.num_scrambles_on_reset,
        model_config=CUSTOM_MODEL_CONFIG,
        agent_name=args.agent_name,
        num_iterations=args.num_iterations,
        restore_path=args.restore_path,
    )
