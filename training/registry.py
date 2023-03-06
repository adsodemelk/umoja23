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

from contextlib import contextmanager

import ray
from ray import tune
from ray.rllib.models import ModelCatalog

from rubiks_cube.env import RubiksCube, create_flattened_env
from training.configs import AGENT_NAME, FLATTEN_ACTIONS
from training.PPO_models import (
    FactorisedActionDistribution,
    FactorisedPPOModel,
    FlatPPOModel,
)


def register(
    agent_name: str = AGENT_NAME,
) -> None:
    if agent_name != "PPO":
        raise ValueError(f"Unexpected agent name {agent_name}")
    if FLATTEN_ACTIONS:
        tune.register_env("rubiks_cube_env", create_flattened_env)
        ModelCatalog.register_custom_model("custom_model", FlatPPOModel)
    else:
        tune.register_env(
            "rubiks_cube_env", lambda env_config: RubiksCube(**env_config)
        )
        ModelCatalog.register_custom_model("custom_model", FactorisedPPOModel)
        ModelCatalog.register_custom_action_dist(
            "factorised_action_dist", FactorisedActionDistribution
        )


@contextmanager
def _ray():
    ray.init(local_mode=True)
    try:
        yield
    finally:
        ray.shutdown()
