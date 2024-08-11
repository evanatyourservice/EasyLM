# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Schedule-Free wrapper for faster training & removes the need for lr decay.

Modified to not need base optimizer."""

from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax.schedules import _schedule
from optax.transforms import _adding


class ScheduleFreeState(NamedTuple):
    """State for schedule_free."""

    b1: chex.Array
    weight_sum: chex.Array
    step_count: chex.Array
    max_lr: chex.Array
    z: base.Params


def schedule_free_eval_params(state: ScheduleFreeState, params: base.Params):
    return jax.tree_util.tree_map(
        lambda yi, zi: (yi - (1.0 - state.b1) * zi) / state.b1, params, state.z
    )


def schedule_free(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    weight_lr_power: float = 2.0,
    state_dtype=jnp.float32,
) -> base.GradientTransformationExtraArgs:
    def init_fn(params: base.Params) -> ScheduleFreeState:
        if b1 == 0:
            raise ValueError(
                "The current implementation of schedule_free requires b1 > 0."
            )
        z = jax.tree_util.tree_map(lambda t: t.astype(state_dtype), params)
        return ScheduleFreeState(
            b1=jnp.array([b1], dtype=jnp.float32),
            weight_sum=jnp.zeros([], dtype=jnp.float32),
            step_count=jnp.ones([], dtype=jnp.int32),
            max_lr=jnp.zeros([], dtype=jnp.float32),
            z=z,
        )

    def update_fn(
        base_updates: base.Updates,
        state: ScheduleFreeState,
        params: base.Params,
    ):
        lr = learning_rate
        if callable(learning_rate):
            lr = learning_rate(state.step_count)
        max_lr = jnp.maximum(state.max_lr, lr)

        next_step_count = state.step_count + 1

        weight = max_lr**weight_lr_power
        next_total_weight = state.weight_sum + weight
        # We add this to avoid NaNs in the case of a small learning rate.
        ck = jnp.where(
            jnp.logical_or(jnp.isnan(weight), jnp.isnan(next_total_weight)),
            jnp.full(weight.shape, jnp.nan),
            jnp.nan_to_num(weight / next_total_weight, nan=0.0, posinf=jnp.inf),
        )

        z = jax.tree_util.tree_map(
            lambda pi, ui: jnp.asarray(pi + ui).astype(jnp.asarray(pi).dtype),
            state.z,
            base_updates,
        )

        # Important: recompute x to both save memory and maintain accurate x seq
        # especially if y is modified by another transform wrapped on top.
        prev_x = jax.tree_util.tree_map(
            lambda yi, zi: (yi - (1.0 - b1) * zi) / b1, params, state.z
        )

        x = jax.tree_util.tree_map(
            lambda xi, zi: (1.0 - ck) * xi + ck * zi,
            prev_x,
            z,
        )
        new_params = jax.tree_util.tree_map(
            lambda xi, zi: b1 * xi + (1.0 - b1) * zi,
            x,
            z,
        )
        updates = jax.tree_util.tree_map(lambda npi, pi: npi - pi, new_params, params)

        next_state = ScheduleFreeState(
            b1=jnp.array([b1], dtype=jnp.float32),
            weight_sum=next_total_weight,
            step_count=next_step_count,
            max_lr=max_lr,
            z=z,
        )

        return updates, next_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
