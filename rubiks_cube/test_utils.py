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

from typing import Callable

import numpy as np
import pytest

from rubiks_cube.constants import CubeMovementAmount, Face
from rubiks_cube.reward_functions import SparseRewardFunction, State
from rubiks_cube.utils import (
    generate_all_moves,
    generate_back_move,
    generate_down_move,
    generate_front_move,
    generate_left_move,
    generate_right_move,
    generate_up_move,
    make_solved_cube,
)

# 3x3x3 moves, for testing purposes
up_move = generate_up_move(CubeMovementAmount.CLOCKWISE)
up_move_inverse = generate_up_move(CubeMovementAmount.ANTI_CLOCKWISE)
up_move_half_turn = generate_up_move(CubeMovementAmount.HALF_TURN)
front_move = generate_front_move(CubeMovementAmount.CLOCKWISE)
front_move_inverse = generate_front_move(CubeMovementAmount.ANTI_CLOCKWISE)
front_move_half_turn = generate_front_move(CubeMovementAmount.HALF_TURN)
right_move = generate_right_move(CubeMovementAmount.CLOCKWISE)
right_move_inverse = generate_right_move(CubeMovementAmount.ANTI_CLOCKWISE)
right_move_half_turn = generate_right_move(CubeMovementAmount.HALF_TURN)
back_move = generate_back_move(CubeMovementAmount.CLOCKWISE)
back_move_inverse = generate_back_move(CubeMovementAmount.ANTI_CLOCKWISE)
back_move_half_turn = generate_back_move(CubeMovementAmount.HALF_TURN)
left_move = generate_left_move(CubeMovementAmount.CLOCKWISE)
left_move_inverse = generate_left_move(CubeMovementAmount.ANTI_CLOCKWISE)
left_move_half_turn = generate_left_move(CubeMovementAmount.HALF_TURN)
down_move = generate_down_move(CubeMovementAmount.CLOCKWISE)
down_move_inverse = generate_down_move(CubeMovementAmount.ANTI_CLOCKWISE)
down_move_half_turn = generate_down_move(CubeMovementAmount.HALF_TURN)


@pytest.mark.parametrize(
    "move, inverse_move",
    [
        (up_move, up_move_inverse),
        (front_move, front_move_inverse),
        (right_move, right_move_inverse),
        (back_move, back_move_inverse),
        (left_move, left_move_inverse),
        (down_move, down_move_inverse),
    ],
)
def test_inverses(
    differently_stickered_cube: np.ndarray,
    move: Callable[[np.ndarray], np.ndarray],
    inverse_move: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Test that applying a move followed by its inverse leads back to the original"""
    cube = move(differently_stickered_cube)
    cube = inverse_move(cube)
    assert np.array_equal(cube, differently_stickered_cube)


@pytest.mark.parametrize(
    "move, half_turn_move",
    [
        (up_move, up_move_half_turn),
        (front_move, front_move_half_turn),
        (right_move, right_move_half_turn),
        (back_move, back_move_half_turn),
        (left_move, left_move_half_turn),
        (down_move, down_move_half_turn),
    ],
)
def test_half_turns(
    differently_stickered_cube: np.ndarray,
    move: Callable[[np.ndarray], np.ndarray],
    half_turn_move: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Test that 2 applications of a move followed by its half turn leads back to the original"""
    cube = move(differently_stickered_cube)
    cube = move(cube)
    cube = half_turn_move(cube)
    assert np.array_equal(cube, differently_stickered_cube)


def test_solved_reward(
    solved_cube: np.ndarray, differently_stickered_cube: np.ndarray
) -> None:
    """Test that the cube fixtures have the expected rewards"""
    solved_state = State(cube=solved_cube, step_count=np.int32(0))
    differently_stickered_state = State(
        cube=differently_stickered_cube, step_count=np.int32(0)
    )
    assert np.equal(SparseRewardFunction()(solved_state), 1.0)
    assert np.equal(SparseRewardFunction()(differently_stickered_state), 0.0)


@pytest.mark.parametrize("move", generate_all_moves())
def test_moves_nontrivial(
    solved_cube: np.ndarray,
    differently_stickered_cube: np.ndarray,
    move: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Test that all moves leave the cube in a non-solved state"""
    move_solved_cube = move(solved_cube)
    move_solved_state = State(cube=move_solved_cube, step_count=np.int32(0))
    assert np.equal(SparseRewardFunction()(move_solved_state), 0.0)
    assert (
        np.not_equal(solved_cube, move_solved_cube).sum()
        == (len(Face) - 2) * solved_cube.shape[-1]
    )
    for face in Face:
        assert (
            np.equal(move_solved_cube, face.value).sum()
            == solved_cube.shape[-1] * solved_cube.shape[-1]
        )
    moved_differently_stickered_cube = move(differently_stickered_cube)
    num_face_impacted_cubies = 8
    num_non_face_impacted_cubies = (len(Face) - 2) * 3
    assert (
        np.not_equal(differently_stickered_cube, moved_differently_stickered_cube).sum()
        == num_non_face_impacted_cubies + num_face_impacted_cubies
    )
    assert np.array_equal(
        differently_stickered_cube[
            :,
            1,
            1,
        ],
        moved_differently_stickered_cube[
            :,
            1,
            1,
        ],
    )


@pytest.mark.parametrize(
    "first_move, second_move",
    [(up_move, down_move), (right_move, left_move), (front_move, back_move)],
)
def test_commuting_moves(
    differently_stickered_cube: np.ndarray,
    first_move: Callable[[np.ndarray], np.ndarray],
    second_move: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Check that moves that should commute, do in fact commute
    (on a differently stickered cube)"""
    first_then_second = second_move(first_move(differently_stickered_cube))
    second_then_first = first_move(second_move(differently_stickered_cube))
    assert np.array_equal(first_then_second, second_then_first)


@pytest.mark.parametrize(
    "first_move, second_move",
    [
        (up_move, front_move),
        (up_move, right_move),
        (up_move, back_move),
        (up_move, left_move),
        (front_move, right_move),
        (front_move, left_move),
        (front_move, down_move),
        (right_move, back_move),
        (right_move, down_move),
        (back_move, left_move),
        (back_move, down_move),
        (left_move, down_move),
    ],
)
def test_non_commuting_moves(
    solved_cube: np.ndarray,
    first_move: Callable[[np.ndarray], np.ndarray],
    second_move: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Check that moves that should not commute, do not (on a solved cube)"""
    first_then_second = second_move(first_move(solved_cube))
    second_then_first = first_move(second_move(solved_cube))
    assert ~np.array_equal(first_then_second, second_then_first)


def test_checkerboard() -> None:
    """Check that the checkerboard scramble gives the expected result"""
    cube = make_solved_cube()
    all_moves = generate_all_moves()
    for index in [2, 17, 5, 11, 8, 14]:
        cube = all_moves[index](cube)
    opposite_face = [Face.DOWN, Face.BACK, Face.LEFT, Face.FRONT, Face.RIGHT, Face.UP]
    for face in Face:
        expected_result = np.concatenate(
            [
                np.array([face.value, opposite_face[face.value].value])
                for _ in range((3 * 3) // 2)
            ]
            + [np.array([face.value])]
        ).reshape(3, 3)
        assert np.array_equal(cube[face.value], expected_result)


def test_manual_scramble(
    solved_cube: np.ndarray, expected_scramble_result: np.ndarray
) -> None:
    """Testing a particular scramble manually.
    Scramble chosen to have all faces touched at least once"""
    scramble = [
        up_move,
        left_move_half_turn,
        down_move_inverse,
        up_move_half_turn,
        back_move_inverse,
        right_move,
        front_move,
        right_move_inverse,
        left_move_inverse,
        back_move_half_turn,
        front_move_inverse,
        up_move,
        down_move,
    ]
    cube = solved_cube
    for move in scramble:
        cube = move(cube)
    assert np.array_equal(expected_scramble_result, cube)
