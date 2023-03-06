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

from typing import List

import gym
import numpy as np


def unflatten_action(action: int, factor_sizes: List[int]):
    """Translate discrete action into tuple of discrete actions"""
    unflattened_action = []
    for size in factor_sizes:
        action, remainder = divmod(action, size)
        unflattened_action.append(remainder)
    return unflattened_action


class FlatteningActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Tuple)
        self.factor_sizes = [space.n for space in env.action_space.spaces]
        self.action_space = gym.spaces.Discrete(np.prod(self.factor_sizes))

    def action(self, action: int):
        return unflatten_action(action=action, factor_sizes=self.factor_sizes)
