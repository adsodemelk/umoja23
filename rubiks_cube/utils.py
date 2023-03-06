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

from typing import Any, Callable, List

import numpy as np

from rubiks_cube.constants import CubeMovementAmount, Face

# Convention:
# 0 = up face
# 1 = front face
# 2 = right face
# 3 = back face
# 4 = left face
# 5 = down face
# All read in reading order when looking directly at face
# To look directly at the faces:
# UP: LEFT face on the left and BACK face pointing up
# FRONT: LEFT face on the left and UP face pointing up
# RIGHT: FRONT face on the left and UP face pointing up
# BACK: RIGHT face on the left and UP face pointing up
# LEFT: BACK face on the left and UP face pointing up
# DOWN: LEFT face on the left and FRONT face pointing up

# Turn amounts (eg clockwise) are when looking directly at the face


def make_solved_cube() -> np.ndarray:
    return np.stack([face.value * np.ones((3, 3), dtype=np.int8) for face in Face])


def is_solved(cube: np.ndarray) -> bool:
    max_sticker_by_side = np.max(cube, axis=(-1, -2))
    min_sticker_by_side = np.min(cube, axis=(-1, -2))
    return bool(np.array_equal(max_sticker_by_side, min_sticker_by_side))


def sparse_reward_function(state: Any) -> np.ndarray:
    solved = is_solved(state.cube)
    return np.float32(solved)


def do_rotation(
    cube: np.ndarray,
    face: Face,
    amount: CubeMovementAmount,
    adjacent_faces: np.ndarray,
    adjacent_faces_columns: np.ndarray,
    adjacent_faces_rows: np.ndarray,
) -> np.ndarray:
    """Perform the rotation, given information about which pieces move.

    Args:
        cube: the unrotated cube.
        face: which face rotates when the layer is moved.
        amount: how much to rotate by.
        adjacent_faces: array of shape (4,) indicating which faces are adjacent to the rotated
            face, in the order in which a clockwise move would be performed.
        adjacent_faces_columns: array of shape (12,) indicating for each adjacent face the column
            indices of the stickers that will move on the adjacent faces.
            For example the first 4 entries are the column indices passed through (in the order in
            which a clockwise turn would be performed) on the first adjacent face, the next 4
            correspond to the second adjacent face, and so on.
        adjacent_faces_rows: as above but for the rows.

    Returns:
        moved_cube: the rotated cube.
    """
    moved_cube = cube.copy()
    moved_cube[face.value] = np.rot90(cube[face.value], k=-amount.value)
    adjacent_faces = np.repeat(adjacent_faces, 3)
    moved_cube[adjacent_faces, adjacent_faces_rows, adjacent_faces_columns] = np.roll(
        cube[adjacent_faces, adjacent_faces_rows, adjacent_faces_columns],
        shift=3 * amount.value,
    )
    return moved_cube


def generate_up_move(amount: CubeMovementAmount) -> Callable[[np.ndarray], np.ndarray]:
    def up_move_function(cube: np.ndarray) -> np.ndarray:
        adjacent_faces = np.array(
            [Face.FRONT.value, Face.LEFT.value, Face.BACK.value, Face.RIGHT.value]
        )
        adjacent_faces_columns = np.concatenate(
            [
                np.arange(3),
                np.arange(3),
                np.arange(3),
                np.arange(3),
            ]
        )
        adjacent_faces_rows = np.concatenate(
            [
                np.repeat(0, 3),
                np.repeat(0, 3),
                np.repeat(0, 3),
                np.repeat(0, 3),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.UP,
            amount=amount,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return up_move_function


def generate_front_move(
    amount: CubeMovementAmount,
) -> Callable[[np.ndarray], np.ndarray]:
    def front_move_function(cube: np.ndarray) -> np.ndarray:
        adjacent_faces = np.array(
            [Face.UP.value, Face.RIGHT.value, Face.DOWN.value, Face.LEFT.value]
        )
        adjacent_faces_columns = np.concatenate(
            [
                np.arange(3),
                np.repeat(0, 3),
                np.flip(np.arange(3)),
                np.repeat(2, 3),
            ]
        )
        adjacent_faces_rows = np.concatenate(
            [
                np.repeat(2, 3),
                np.arange(3),
                np.repeat(0, 3),
                np.flip(np.arange(3)),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.FRONT,
            amount=amount,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return front_move_function


def generate_right_move(
    amount: CubeMovementAmount,
) -> Callable[[np.ndarray], np.ndarray]:
    def right_move_function(cube: np.ndarray) -> np.ndarray:
        adjacent_faces = np.array(
            [Face.UP.value, Face.BACK.value, Face.DOWN.value, Face.FRONT.value]
        )
        adjacent_faces_columns = np.concatenate(
            [
                np.repeat(2, 3),
                np.repeat(0, 3),
                np.repeat(2, 3),
                np.repeat(2, 3),
            ]
        )
        adjacent_faces_rows = np.concatenate(
            [
                np.flip(np.arange(3)),
                np.arange(3),
                np.flip(np.arange(3)),
                np.flip(np.arange(3)),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.RIGHT,
            amount=amount,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return right_move_function


def generate_back_move(
    amount: CubeMovementAmount,
) -> Callable[[np.ndarray], np.ndarray]:
    def back_move_function(cube: np.ndarray) -> np.ndarray:
        adjacent_faces = np.array(
            [Face.UP.value, Face.LEFT.value, Face.DOWN.value, Face.RIGHT.value]
        )
        adjacent_faces_columns = np.concatenate(
            [
                np.flip(np.arange(3)),
                np.repeat(0, 3),
                np.arange(3),
                np.repeat(2, 3),
            ]
        )
        adjacent_faces_rows = np.concatenate(
            [
                np.repeat(0, 3),
                np.arange(3),
                np.repeat(2, 3),
                np.flip(np.arange(3)),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.BACK,
            amount=amount,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return back_move_function


def generate_left_move(
    amount: CubeMovementAmount,
) -> Callable[[np.ndarray], np.ndarray]:
    def left_move_function(cube: np.ndarray) -> np.ndarray:
        adjacent_faces = np.array(
            [Face.UP.value, Face.FRONT.value, Face.DOWN.value, Face.BACK.value]
        )
        adjacent_faces_columns = np.concatenate(
            [
                np.repeat(0, 3),
                np.repeat(0, 3),
                np.repeat(0, 3),
                np.repeat(2, 3),
            ]
        )
        adjacent_faces_rows = np.concatenate(
            [
                np.arange(3),
                np.arange(3),
                np.arange(3),
                np.flip(np.arange(3)),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.LEFT,
            amount=amount,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return left_move_function


def generate_down_move(
    amount: CubeMovementAmount,
) -> Callable[[np.ndarray], np.ndarray]:
    def down_move_function(cube: np.ndarray) -> np.ndarray:
        adjacent_faces = np.array(
            [Face.FRONT.value, Face.RIGHT.value, Face.BACK.value, Face.LEFT.value]
        )
        adjacent_faces_columns = np.concatenate(
            [
                np.arange(3),
                np.arange(3),
                np.arange(3),
                np.arange(3),
            ]
        )
        adjacent_faces_rows = np.concatenate(
            [
                np.repeat(2, 3),
                np.repeat(2, 3),
                np.repeat(2, 3),
                np.repeat(2, 3),
            ]
        )
        return do_rotation(
            cube=cube,
            face=Face.DOWN,
            amount=amount,
            adjacent_faces=adjacent_faces,
            adjacent_faces_columns=adjacent_faces_columns,
            adjacent_faces_rows=adjacent_faces_rows,
        )

    return down_move_function


def generate_all_moves() -> List[Callable[[np.ndarray], np.ndarray]]:
    return [
        f(amount)
        for f in [
            generate_up_move,
            generate_front_move,
            generate_right_move,
            generate_back_move,
            generate_left_move,
            generate_down_move,
        ]
        for amount in CubeMovementAmount
    ]
