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

from typing import Dict

from ray.rllib.algorithms.ppo import PPOConfig

AGENT_NAME = "PPO"
FLATTEN_ACTIONS = False


EASY_ENV_CONFIG = {
    "step_limit": 10,
    "reward_function_type": "sparse",
    "num_scrambles_on_reset": 2,
}

MEDIUM_ENV_CONFIG = {
    "step_limit": 10,
    "reward_function_type": "sparse",
    "num_scrambles_on_reset": 8,
}

HARD_ENV_CONFIG = {
    "step_limit": 10,
    "reward_function_type": "sparse",
    "num_scrambles_on_reset": 20,
}

CUSTOM_MODEL_CONFIG = {
    "cube_embed_dim": 4,
    "step_count_embed_dim": 4,
    "dense_layer_dims": [32, 64],
}


def get_config(
    env_config: Dict,
    model_config: Dict,
    agent_name: str,
) -> Dict:
    if agent_name == "PPO":
        config = PPOConfig().to_dict()
    else:
        raise ValueError(
            f"Unexpected agent name {agent_name}, have registered {AGENT_NAME}"
        )
    if not FLATTEN_ACTIONS:
        config["model"]["custom_action_dist"] = "factorised_action_dist"
    config["model"]["custom_model"] = "custom_model"
    config["model"]["custom_model_config"] = {**env_config, **model_config}
    config["env"] = "rubiks_cube_env"
    config["env_config"] = env_config
    return config
