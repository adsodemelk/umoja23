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

from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_tfp

from rubiks_cube.constants import CubeMovementAmount, Face

_, tf, _ = try_import_tf()
tfp = try_import_tfp()
tfkl = tf.keras.layers


class FlatPPOModel(TFModelV2):
    """A model that takes the observation (cube and step_count) to logits for the
    flattened action space, as well as a value for the critic training.
    This model uses a shared encoding - a potential path for investigating could be
    whether separating into two models helps"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, None, model_config, name)
        cube_input = tfkl.Input(
            shape=(len(Face), 3, 3),
            dtype=tf.int32,
            name="cube_input",
        )
        step_count_input = tfkl.Input(
            shape=(1,), dtype=tf.float32, name="step_count_input"
        )
        embedded_cube = tfkl.Embedding(
            len(Face), model_config["custom_model_config"]["cube_embed_dim"]
        )(cube_input)
        flattened_embedding = tfkl.Flatten()(embedded_cube)
        step_count_embedding = tfkl.Dense(
            model_config["custom_model_config"]["step_count_embed_dim"]
        )(step_count_input / model_config["custom_model_config"]["step_limit"])
        dense_layer_output = tfkl.concatenate(
            [flattened_embedding, step_count_embedding], axis=-1
        )
        for output_dim in model_config["custom_model_config"]["dense_layer_dims"]:
            dense_layer_output = tfkl.Dense(output_dim, activation="relu")(
                dense_layer_output
            )
        logits = tfkl.Dense(action_space.n)(dense_layer_output)
        value = tfkl.Dense(1)(dense_layer_output)
        self.custom_model = tf.keras.Model(
            [cube_input, step_count_input],
            [logits, value],
            name="custom_model",
        )
        self.custom_model.summary()

    def forward(self, input_dict, state, seq_lens):
        logits, self._value = self.custom_model(
            [input_dict["obs"]["cube"], input_dict["obs"]["step_count"]]
        )
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


class FactorisedPPOModel(TFModelV2):
    """A model similarly to the above but which parametrises 2 distributions: P(face) and
    P(amount | face). To do this we compute an encoding (used also for the value function)
    and then project this for logits for the face selection; for selecting the amount we
    concatenate the encoding with (a one hot encoding of) the selected face.
    This model also requires us to use the factorised action distribution below."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, None, model_config, name)
        cube_input = tfkl.Input(
            shape=(len(Face), 3, 3),
            dtype=tf.int32,
            name="cube_input",
        )
        step_count_input = tfkl.Input(
            shape=(1,), dtype=tf.float32, name="step_count_input"
        )
        selected_face_input = tfkl.Input(
            shape=(), dtype=tf.int32, name="selected_face_input"
        )
        encoding_input = tfkl.Input(
            shape=(model_config["custom_model_config"]["dense_layer_dims"][-1],),
            dtype=tf.float32,
        )
        embedded_cube = tfkl.Embedding(
            len(Face), model_config["custom_model_config"]["cube_embed_dim"]
        )(cube_input)
        flattened_embedding = tfkl.Flatten()(embedded_cube)
        step_count_embedding = tfkl.Dense(
            model_config["custom_model_config"]["step_count_embed_dim"],
            activation="relu",
        )(step_count_input / model_config["custom_model_config"]["step_limit"])
        dense_layer_output = tfkl.concatenate(
            [flattened_embedding, step_count_embedding], axis=-1
        )
        for output_dim in model_config["custom_model_config"]["dense_layer_dims"]:
            dense_layer_output = tfkl.Dense(output_dim, activation="relu")(
                dense_layer_output
            )
        value = tfkl.Dense(1)(dense_layer_output)
        self.encoding_model = tf.keras.Model(
            [cube_input, step_count_input],
            [dense_layer_output, value],
            name="encoding_model",
        )
        face_logits = tfkl.Dense(len(Face))(encoding_input)
        self.face_selection_model = tf.keras.Model(
            encoding_input, face_logits, name="face_selection_model"
        )
        face_action_one_hot = tf.one_hot(
            tf.cast(selected_face_input, tf.int32),
            len(Face),
            axis=-1,
        )
        cube_movement_amount_logits = tfkl.Dense(len(CubeMovementAmount))(
            tf.concat([encoding_input, face_action_one_hot], axis=-1)
        )
        self.cube_movement_amount_selection_model = tf.keras.Model(
            [encoding_input, selected_face_input],
            cube_movement_amount_logits,
            name="cube_movement_amount_selection_model",
        )
        self.encoding_model.summary()
        self.face_selection_model.summary()
        self.cube_movement_amount_selection_model.summary()
        total_params = (
            self.encoding_model.count_params()
            + self.face_selection_model.count_params()
            + self.cube_movement_amount_selection_model.count_params()
        )
        print(f"Total parameters in model: {total_params}")

    def forward(self, input_dict, state, seq_lens):
        encoding, self._value = self.encoding_model(
            [input_dict["obs"]["cube"], input_dict["obs"]["step_count"]]
        )
        return encoding, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


class FactorisedActionDistribution(TFActionDistribution):
    """A factorised action distribution, for use with the factorised model above."""

    def __init__(self, inputs, model):
        self.inputs = inputs
        self.model = model
        self.face_selection_distribution = self._face_selection_distribution()
        super().__init__(inputs=inputs, model=model)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config["custom_model_config"]["dense_layer_dims"][-1]

    def logp(self, x):
        """Compute the log probability as the sum of the log probabilities for the two
        factors, as P(x, y) = P(x) * P(y | x)"""
        if isinstance(x, tuple):
            selected_face = tf.cast(x[0], tf.float32)
            selected_cube_movement_amount = tf.cast(x[1], tf.float32)
        else:
            selected_face, selected_cube_movement_amount = tf.split(
                x, num_or_size_splits=2, axis=-1
            )
            selected_face = tf.squeeze(selected_face, axis=-1)
            selected_cube_movement_amount = tf.squeeze(
                selected_cube_movement_amount, axis=-1
            )

        face_logps = self.face_selection_distribution.log_prob(selected_face)
        amount_logps = self._cube_movement_amount_distribution(
            selected_face=selected_face
        ).log_prob(selected_cube_movement_amount)
        return face_logps + amount_logps

    def entropy(self):
        """Similarly to the above calculation for log probability, this method calculates
        the entropy of a factorised distribution. This is used in entropy regularisation
        for PPO."""
        selected_face = self.face_selection_distribution.sample()
        amount_distribution = self._cube_movement_amount_distribution(
            selected_face=selected_face
        )
        return (
            self.face_selection_distribution.entropy() + amount_distribution.entropy()
        )

    def kl(self, other):
        """Similarly to the above but for the KL divergence."""
        selected_face = self.face_selection_distribution.sample()
        amount_distribution = self._cube_movement_amount_distribution(
            selected_face=selected_face
        )
        other_amount_distribution = other._cube_movement_amount_distribution(
            selected_face=selected_face
        )
        return self.face_selection_distribution.kl_divergence(
            other.face_selection_distribution
        ) + amount_distribution.kl_divergence(other_amount_distribution)

    def _face_selection_distribution(self):
        """This can be computed on init since it does not depend on sampled values"""
        logits = self.model.face_selection_model(self.inputs)
        return tfp.distributions.Categorical(logits)

    def _cube_movement_amount_distribution(self, selected_face):
        """The marginal distribution depends on the face selected"""
        logits = self.model.cube_movement_amount_selection_model(
            [self.inputs, selected_face]
        )
        return tfp.distributions.Categorical(logits)

    def _build_sample_op(self):
        """Method used by RLLIB to sample an action from the policy"""
        selected_face = self.face_selection_distribution.sample()
        amount_distribution = self._cube_movement_amount_distribution(
            selected_face=selected_face
        )
        selected_cube_movement_amount = amount_distribution.sample()
        return (
            tf.cast(selected_face, tf.int64),
            tf.cast(selected_cube_movement_amount, tf.int64),
        )

    def deterministic_sample(self):
        """Method for greedy action selection"""
        selected_face = self.face_selection_distribution._mode()
        amount_distribution = self._cube_movement_amount_distribution(
            selected_face=selected_face
        )
        selected_cube_movement_amount = amount_distribution._mode()
        return (
            tf.cast(selected_face, tf.int64),
            tf.cast(selected_cube_movement_amount, tf.int64),
        )
