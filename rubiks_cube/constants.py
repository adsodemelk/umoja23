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

from enum import Enum

DEFAULT_STICKER_COLOURS = ["white", "green", "red", "blue", "orange", "yellow"]


class Face(Enum):
    UP = 0
    FRONT = 1
    RIGHT = 2
    BACK = 3
    LEFT = 4
    DOWN = 5


class CubeMovementAmount(Enum):
    CLOCKWISE = 1
    ANTI_CLOCKWISE = -1
    HALF_TURN = 2


OPPOSITE_FACES = {
    Face.UP: Face.DOWN,
    Face.DOWN: Face.UP,
    Face.FRONT: Face.BACK,
    Face.BACK: Face.FRONT,
    Face.RIGHT: Face.LEFT,
    Face.LEFT: Face.RIGHT,
}

CUBE_MOVE_AMOUNT_INDICES = {
    CubeMovementAmount.CLOCKWISE: 0,
    CubeMovementAmount.ANTI_CLOCKWISE: 1,
    CubeMovementAmount.HALF_TURN: 2,
}
