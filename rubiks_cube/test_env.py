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

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pytest_mock

from rubiks_cube.constants import CubeMovementAmount, Face
from rubiks_cube.env import RubiksCube


def check_env_does_not_smoke(
    env: RubiksCube,
) -> None:
    """Run an episode of the environment, with a step function to check no errors occur."""
    observation_space = env.observation_space
    action_space = env.action_space
    observation = env.reset()
    assert observation_space.contains(observation)
    episode_length = 0
    done = False
    while not done:
        action = action_space.sample()
        assert action_space.contains(action)
        observation, reward, done, info = env.step(action)
        assert observation_space.contains(observation)
        assert isinstance(reward, float)
        episode_length += 1
        if episode_length > env.step_limit:
            # Exit condition to make sure tests don't enter infinite loop, should not be hit
            raise Exception("Entered infinite loop")


def test_flatten(rubiks_cube_env: RubiksCube) -> None:
    """Test that flattening and unflattening actions are inverse to each other"""
    flat_actions = np.arange(len(Face) * len(CubeMovementAmount), dtype=np.int16)
    faces = np.arange(len(Face), dtype=np.int16)
    amounts = np.arange(len(CubeMovementAmount), dtype=np.int16)
    unflat_actions = np.stack(
        [
            np.repeat(faces, len(CubeMovementAmount)),
            np.concatenate([amounts for _ in range(len(Face))]),
        ]
    )
    assert np.array_equal(
        unflat_actions, rubiks_cube_env._unflatten_action(flat_actions)
    )
    assert np.array_equal(flat_actions, rubiks_cube_env._flatten_action(unflat_actions))


def test_scramble_on_reset(
    rubiks_cube_env: RubiksCube, expected_scramble_result: np.ndarray
) -> None:
    """Test that the environment reset is performing correctly when given a particular scramble
    (chosen manually)"""
    flat_sequence = np.array(
        [0, 14, 16, 2, 10, 6, 3, 7, 13, 11, 4, 0, 15], dtype=np.int16
    )
    cube = rubiks_cube_env._scramble_solved_cube(flat_actions_in_scramble=flat_sequence)
    assert np.array_equal(expected_scramble_result, cube)


def test_rubiks_cube_env_reset(rubiks_cube_env: RubiksCube) -> None:
    """Validates the reset of the environment."""
    observation = rubiks_cube_env.reset()
    assert observation["step_count"] == 0


def test_rubiks_cube_env_step(rubiks_cube_env: RubiksCube) -> None:
    """Validates the step of the environment."""
    observation = rubiks_cube_env.reset()
    action = rubiks_cube_env.action_space.sample()
    next_observation, reward, done, info = rubiks_cube_env.step(action)

    # Check that the observation has changed
    assert not np.array_equal(next_observation["cube"], observation["cube"])
    assert next_observation["step_count"] == 1

    next_next_observation, reward, done, info = rubiks_cube_env.step(action)

    # Check that the state has changed
    assert not np.array_equal(next_next_observation["cube"], next_observation["cube"])
    assert next_next_observation["step_count"] == 2


def test_rubiks_cube_env_does_not_smoke() -> None:
    """Test that we can run an episode without any errors."""
    check_env_does_not_smoke(env=RubiksCube(step_limit=10))


def test_rubiks_cube_env_render(
    monkeypatch: pytest.MonkeyPatch, rubiks_cube_env: RubiksCube
) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    _ = rubiks_cube_env.reset()
    rubiks_cube_env.render()
    rubiks_cube_env.close()
    action = rubiks_cube_env.action_space.sample()
    _ = rubiks_cube_env.step(action)
    rubiks_cube_env.render()
    rubiks_cube_env.close()


@pytest.mark.parametrize("step_limit", [3, 4, 5])
def test_rubiks_cube_env_done(step_limit: int) -> None:
    """Test that the done signal is sent correctly"""
    env = RubiksCube(step_limit=step_limit)
    _ = env.reset()
    action = env.action_space.sample()
    episode_length = 0
    done = False
    while not done:
        _, _, done, _ = env.step(action)
        episode_length += 1
        if episode_length > 10:
            # Exit condition to make sure tests don't enter infinite loop, should not be hit
            raise Exception("Entered infinite loop")
    assert episode_length == step_limit


def test_rubiks_cube_animation(
    rubiks_cube_env: RubiksCube, mocker: pytest_mock.MockerFixture
) -> None:
    """Check that the animation method creates the animation correctly."""
    states = mocker.MagicMock()
    animation = rubiks_cube_env.animation(states)
    assert isinstance(animation, matplotlib.animation.Animation)
