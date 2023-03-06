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

import sys
from typing import Dict, List, Optional, Sequence, Tuple

import gym
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np

from rubiks_cube.constants import (
    CUBE_MOVE_AMOUNT_INDICES,
    DEFAULT_STICKER_COLOURS,
    OPPOSITE_FACES,
    CubeMovementAmount,
    Face,
)
from rubiks_cube.flat_action_wrapper import FlatteningActionWrapper
from rubiks_cube.reward_functions import RewardFunction, SparseRewardFunction, State
from rubiks_cube.utils import generate_all_moves, is_solved, make_solved_cube


class RubiksCube(gym.Env):
    """A numpy implementation of the Rubik's Cube.

    - observation: Observation
        - cube: numpy array (int8) of shape (6, 3, 3):
            each cell contains the index of the corresponding colour of the sticker in the scramble.
        - step_count: numpy array (int32):
            specifies how many timesteps have elapsed since environment reset

    - action:
        multi discrete array containing the move to perform (face and direction)

    - reward::
        by default, 1 if cube is solved or otherwise 0

    - episode termination:
        if either the cube is solved or a horizon is reached
    """

    def __init__(
        self,
        step_limit: int = 200,
        reward_function_type: str = "sparse",
        num_scrambles_on_reset: int = 100,
        sticker_colours: Optional[list] = None,
    ):
        if step_limit <= 0:
            raise ValueError(
                f"The step_limit must be positive, but received step_limit={step_limit}"
            )
        if num_scrambles_on_reset < 0:
            raise ValueError(
                f"The num_scrambles_on_reset must be non-negative, "
                f"but received num_scrambles_on_reset={num_scrambles_on_reset}"
            )
        self.step_limit = step_limit
        self.reward_function = self.create_reward_function(
            reward_function_type=reward_function_type
        )
        self.num_scrambles_on_reset = num_scrambles_on_reset
        self.sticker_colours_cmap = matplotlib.colors.ListedColormap(
            sticker_colours if sticker_colours else DEFAULT_STICKER_COLOURS
        )
        self.num_actions = len(Face) * len(CubeMovementAmount)
        self.all_moves = generate_all_moves()

        self.figure_name = "3x3x3 Rubik's Cube"
        self.figure_size = (6.0, 6.0)
        self.observation_space = gym.spaces.Dict(
            {
                "cube": gym.spaces.Box(
                    shape=(len(Face), 3, 3),
                    dtype=np.int8,
                    low=0,
                    high=len(Face) - 1,
                ),
                "step_count": gym.spaces.Box(
                    shape=(1,),
                    dtype=np.int32,
                    low=0,
                    high=self.step_limit,
                ),
            }
        )
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(len(Face)),
                gym.spaces.Discrete(len(CubeMovementAmount)),
            ]
        )
        self.cube = None
        self.timestep = None

    def set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            np.random.seed(seed=seed)

    def get_state(self) -> str:
        return dump_cube(cube=self.cube)

    @classmethod
    def create_reward_function(cls, reward_function_type: str) -> RewardFunction:
        if reward_function_type == "sparse":
            return SparseRewardFunction()
        else:
            raise ValueError(
                f"Unexpected value for reward_function_type, got {reward_function_type}. "
                f"Possible values: 'sparse'"
            )

    def _unflatten_action(self, action: int) -> np.ndarray:
        """Turn a flat action (index into the sequence of all moves) into a tuple:
            - face (0-5). This indicates the face on which the layer will turn.
            - amount (0-2). This indicates the amount of turning (see below).

        Convention:
        0 = up face
        1 = front face
        2 = right face
        3 = back face
        4 = left face
        5 = down face
        All read in reading order when looking directly at face

        To look directly at the faces:
        UP: LEFT face on the left and BACK face pointing up
        FRONT: LEFT face on the left and UP face pointing up
        RIGHT: FRONT face on the left and UP face pointing up
        BACK: RIGHT face on the left and UP face pointing up
        LEFT: BACK face on the left and UP face pointing up
        DOWN: LEFT face on the left and FRONT face pointing up

        Turning amounts are when looking directly at a face:
        0 = clockwise turn
        1 = anticlockwise turn
        2 = half turn
        """
        face, amount = np.divmod(action, len(CubeMovementAmount))
        return np.stack([face, amount], axis=0)

    def _flatten_action(self, action: np.ndarray) -> int:
        """Inverse of the _flatten_action method"""
        face, amount = action
        return face * len(CubeMovementAmount) + amount

    def _rotate_cube(self, cube: np.ndarray, flat_action: int) -> np.ndarray:
        """Apply a flattened action (index into the sequence of all moves) to a cube"""
        moved_cube = self.all_moves[flat_action](cube)
        return moved_cube

    def _scramble_solved_cube(self, flat_actions_in_scramble: np.ndarray) -> np.ndarray:
        """Return a scrambled cube according to a given sequence of flat actions"""
        cube = make_solved_cube()
        for flat_action in flat_actions_in_scramble:
            cube = self._rotate_cube(cube=cube, flat_action=flat_action)
        return cube

    def _generate_flat_actions_for_scramble(self, num_scrambles: int) -> np.ndarray:
        valid_faces = list(Face)
        prev_face = None
        actions = []
        for _ in range(num_scrambles):
            selected_face = np.random.choice(valid_faces)
            selected_cube_movement_amount = np.random.choice(CubeMovementAmount)
            action = np.array(
                [
                    selected_face.value,
                    CUBE_MOVE_AMOUNT_INDICES[selected_cube_movement_amount],
                ]
            )
            actions.append(self._flatten_action(action=action))
            if prev_face and OPPOSITE_FACES[prev_face] == selected_face:
                valid_faces = [
                    x for x in Face if (x != selected_face and x != prev_face)
                ]
            else:
                valid_faces = [x for x in Face if x != selected_face]
            prev_face = selected_face

        return np.array(actions, dtype=np.int16)

    def reset(self, **kwargs) -> Dict:
        """Resets the environment.

        Returns:
            observation: Observation corresponding to the new state of the environment
        """
        flat_actions_in_scramble = self._generate_flat_actions_for_scramble(
            num_scrambles=self.num_scrambles_on_reset
        )
        self.cube = self._scramble_solved_cube(
            flat_actions_in_scramble=flat_actions_in_scramble
        )
        self.step_count = 0
        observation = {
            "cube": self.cube.copy(),
            "step_count": np.array([self.step_count], dtype=np.int32),
        }
        return observation

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Run one timestep of the environment's dynamics.

        Args:
            action: Array containing the face to move and the amount
                to move by.

        Returns:
            next_observation: Observation corresponding to the next state of the environment
        """
        self.cube = self._rotate_cube(
            cube=self.cube, flat_action=self._flatten_action(action)
        )
        self.step_count += 1
        next_observation = {
            "cube": self.cube.copy(),
            "step_count": np.array([self.step_count], dtype=np.int32),
        }
        reward = self.reward_function(
            state=State(cube=self.cube, step_count=self.step_count)
        )
        solved = is_solved(self.cube)
        done = (self.step_count >= self.step_limit) | solved
        return next_observation, reward, done, {}

    def render(self, **kwargs) -> None:
        """Render frames of the environment for a given state using matplotlib.
        :param **kwargs:
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        self._draw(ax, self.cube)
        self._update_display(fig)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        plt.close(self.figure_name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        exists = plt.fignum_exists(self.figure_name)
        if exists:
            fig = plt.figure(self.figure_name)
            ax = fig.get_axes()
        else:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=self.figure_size)
            fig.suptitle(self.figure_name)
            ax = ax.flatten()
            plt.tight_layout()
            plt.axis("off")
            if not plt.isinteractive():
                fig.show()
        return fig, ax

    def _draw(self, ax: List[plt.Axes], cube: np.ndarray) -> None:
        i = 0
        for face in Face:
            ax[i].clear()
            ax[i].set_title(label=f"{face}")
            ax[i].set_xticks(np.arange(-0.5, 2, 1))
            ax[i].set_yticks(np.arange(-0.5, 2, 1))
            ax[i].tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                labeltop=False,
                labelright=False,
            )
            ax[i].imshow(
                cube[i],
                cmap=self.sticker_colours_cmap,
                vmin=0,
                vmax=len(Face) - 1,
            )
            ax[i].grid(color="black", linestyle="-", linewidth=2)
            i += 1

    def _update_display(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if "google.colab" in sys.modules:
                plt.show(self.figure_name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            # Block for 0.5 seconds.
            fig.canvas.start_event_loop(0.5)

    def _clear_display(self) -> None:
        if "google.colab" in sys.modules:
            import IPython.display

            IPython.display.clear_output(True)

    def animation(
        self,
        cubes: Sequence[np.ndarray],
        interval: int = 200,
        blit: bool = False,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            cubes: sequence of cubes to render corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            blit: whether to use blitting, which optimises the animation by only re-drawing
                pieces of the plot that have changed. Defaults to False.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=self.figure_size)
        fig.suptitle(self.figure_name)
        plt.tight_layout()
        ax = ax.flatten()
        plt.close(fig)

        def animate(cube_index: int) -> None:
            cube = cubes[cube_index]
            self._draw(ax, cube)

        anim = matplotlib.animation.FuncAnimation(
            fig,
            animate,
            frames=len(cubes),
            blit=blit,
            interval=interval,
        )
        return anim


def dump_cube(cube: np.ndarray) -> str:
    """Dump the cube to string, for use in evaluation"""
    result = ""
    for i in range(len(Face)):
        for j in range(3):
            for k in range(3):
                result += str(cube[i, j, k])
    return result


def create_flattened_env(env_config: Dict) -> gym.Env:
    env = RubiksCube(**env_config)
    env = FlatteningActionWrapper(env)
    return env
