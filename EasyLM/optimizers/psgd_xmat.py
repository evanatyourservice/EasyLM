from functools import partial
from typing import Any, Optional, Union, Callable, NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey
import optax
from optax import tree_utils as otu
from optax._src import base, transform, clipping
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain

from EasyLM.optimizers.utils import add_eps, apply_momentum
from EasyLM.optimizers.schedule_free import schedule_free, schedule_free_eval_params


class PSGDXMatState(NamedTuple):
    count: jax.Array
    key: PRNGKey
    mu: Optional[base.Updates]
    a: jax.Array
    b: jax.Array
    nu: Optional[base.Updates]


def scale_by_xmat(
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = True,
    gradient_clip: Optional[float] = None,
    step_normalizer_order: str = "2nd",
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
    normalize: bool = True,
    adaptive: bool = True,
    b2: float = 0.95,
) -> base.GradientTransformationExtraArgs:
    """
    Implements XMat PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        gradient_clip: optional float, global gradient norm clipping.
        step_normalizer_order: str, '1st' or '2nd'.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: optional float, initial scale for the preconditioner.
        seed: Optional PRNGKey, random seed.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
            Defaults to the same dtype as the parameters.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.
        normalize: bool, whether to normalize the preconditioned grads to unit norm.
        adaptive: bool, layer-wise adaptive second moment.
        b2: float, beta2 for adaptive second moment.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    def init_fn(params):
        key = seed if seed is not None else jax.random.PRNGKey(36)

        # momentum
        mu = None
        if b1 > 0:
            print("PSGD: Using momentum.")
            mu = otu.tree_zeros_like(params, mu_dtype)

        # preconditioner
        a = otu.tree_ones_like(params, precond_dtype)
        b = otu.tree_zeros_like(params, precond_dtype)

        # layer-wise second moment
        nu = None
        if adaptive:
            nu = jax.tree.map(lambda x: jnp.zeros([], dtype=precond_dtype), params)

        # initial state
        return PSGDXMatState(
            count=jnp.zeros([], jnp.int32), key=key, mu=mu, a=a, b=b, nu=nu
        )

    def update_fn(
        updates: base.Updates,
        state: PSGDXMatState,
        params: base.Params = None,
        Hvp: Optional[base.Updates] = None,
        vector: Optional[base.Updates] = None,
        update_preconditioner: Optional[bool] = None,
    ):
        del params
        # use hessian preconditioning if hessian provided
        # otherwise use gg^T whitening type preconditioning
        hessian_based_preconditioning = Hvp is not None
        if hessian_based_preconditioning and (
            vector is None or update_preconditioner is None
        ):
            raise ValueError(
                "If using Hessian-based preconditioning, must also pass in random vector and "
                "update_preconditioner to PSGD's update function. See README for more info."
            )

        # cast grads if precond_dtype is set
        updates = otu.tree_cast(updates, precond_dtype)
        if hessian_based_preconditioning:
            Hvp = otu.tree_cast(Hvp, precond_dtype)
            vector = otu.tree_cast(vector, precond_dtype)

        count_inc = safe_int32_increment(state.count)
        key = state.key

        precond_lr_in = precond_lr
        if isinstance(precond_lr, Callable):
            precond_lr_in = precond_lr(count_inc)

        def _update_precond(key: PRNGKey, a, b, Hvs, vs, count):
            def init_a(a, h, v):
                if precond_init_scale is not None:
                    init_scale = precond_init_scale
                else:
                    if hessian_based_preconditioning:
                        init_scale = (
                            jnp.sum(jnp.square(v)) / jnp.sum(jnp.square(h))
                        ) ** 0.25
                    else:
                        init_scale = (h.size / jnp.sum(jnp.square(h))) ** 0.25
                return a * init_scale

            # init a
            a = jax.lax.cond(
                count == 0,
                lambda a, h, v: jax.tree.map(init_a, a, h, v),
                lambda a, h, v: a,
                a,
                Hvs,
                vs,
            )

            # update preconditioner
            new_a = []
            new_b = []
            for a, b, v, h in zip(a, b, vs, Hvs):
                a, b = _update_precond_Xmat_math_(
                    a, b, v, h, precond_lr_in, step_normalizer_order, precision
                )
                new_a.append(a)
                new_b.append(b)

            return key, new_a, new_b

        def _dont_update_precond(key, a, b, Hvs, vs, count):
            return key, a, b

        if not hessian_based_preconditioning:
            # update cond and vector not passed in, create here
            key, subkey = jax.random.split(key)
            update_preconditioner = (
                jax.random.uniform(subkey) < preconditioner_update_probability
            )
            key, subkey = jax.random.split(key)
            # TODO sharding
            vector = otu.tree_random_like(
                subkey, updates, partial(jax.random.rademacher, dtype=precond_dtype)
            )
            # use grads as Hvp
            Hvp = updates

        flat_a, a_struct = jax.tree.flatten(state.a)
        flat_b, b_struct = jax.tree.flatten(state.b)
        flat_h, _ = jax.tree.flatten(Hvp)
        flat_v, _ = jax.tree.flatten(vector)
        key, a, b = jax.lax.cond(
            update_preconditioner,
            _update_precond,
            _dont_update_precond,
            key,
            flat_a,
            flat_b,
            flat_h,
            flat_v,
            state.count,
        )
        a = a_struct.unflatten(a)
        b = b_struct.unflatten(b)

        # preconditioning
        updates = jax.tree.map(_precond_grad_Xmat_math, a, b, updates)

        if normalize:
            # normalize to unit norm
            global_norm = add_eps(optax.global_norm(updates))
            updates = jax.tree.map(lambda x: x / global_norm, updates)
        elif gradient_clip:
            updates, _ = clipping.clip_by_global_norm(gradient_clip).update(
                updates, base.EmptyState
            )

        # momentum
        mu = None
        momentum_updates = updates
        if state.mu is not None:
            momentum_updates, mu = apply_momentum(
                updates, state.mu, count_inc, b1, nesterov
            )

        # layer-wise second moment
        nu = None
        if adaptive:
            nu = jax.tree.map(
                lambda nu, update: b2 * nu + (1 - b2) * jnp.mean(update**2),
                state.nu,
                updates,
            )
            nu_hat = jax.tree.map(lambda nu: nu / (1 - b2**count_inc), nu)
            updates = jax.tree.map(
                lambda x, nu: x / (jnp.sqrt(nu) + 1e-8), momentum_updates, nu_hat
            )

        mu = otu.tree_cast(mu, mu_dtype)
        a = otu.tree_cast(a, precond_dtype)
        b = otu.tree_cast(b, precond_dtype)
        nu = otu.tree_cast(nu, precond_dtype)
        state = PSGDXMatState(count=count_inc, key=key, mu=mu, a=a, b=b, nu=nu)
        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def xmat(
    learning_rate: Union[float, Callable[[int], float]] = 0.01,
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = True,
    gradient_clip: Optional[float] = None,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    step_normalizer_order: str = "2nd",
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
    normalize: bool = True,
    adaptive: bool = True,
    b2: float = 0.99,
) -> base.GradientTransformationExtraArgs:
    """
    Implements XMat PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate for the optimizer.
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        gradient_clip: optional float, global gradient norm clipping.
        weight_decay: float, weight decay.
        mask: optional mask for weight decay.
        step_normalizer_order: str, '1st' or '2nd'.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: optional float, initial scale for the preconditioner.
        seed: Optional PRNGKey, random seed.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
            Defaults to the same dtype as the parameters.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.
        normalize: bool, whether to normalize the preconditioned grads to unit norm.
        adaptive: bool, layer-wise adaptive second moment.
        b2: float, beta2 for adaptive second moment.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    opt = [
        scale_by_xmat(
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            nesterov=nesterov,
            gradient_clip=gradient_clip,
            step_normalizer_order=step_normalizer_order,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            seed=seed,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precision=precision,
            normalize=normalize,
            adaptive=adaptive,
            b2=b2,
        )
    ]
    if weight_decay > 0:
        opt.append(transform.add_decayed_weights(weight_decay, mask=mask))
    opt.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*opt)


def flip(x):
    return jnp.flip(x, list(np.arange(x.ndim)))


def _update_precond_Xmat_math_(a, b, v, h, precond_lr, step_normalizer, precision):
    """
    Update preconditioner Q = diag(a) + adiag(b) with (vector, Hessian-vector product) = (v, h).
    """
    with jax.default_matmul_precision(precision):
        Qh = a * h + b * flip(h)
        aflip, bflip = flip(a), flip(b)
        invQtv = (aflip * v - bflip * flip(v)) / (a * aflip - b * bflip)

        u, v = Qh * Qh, invQtv * invQtv
        nablaA = u - v
        nablaB = Qh * flip(Qh) - invQtv * flip(invQtv)
        q, r = jnp.divmod(len(nablaB), 2)
        nablaB = jnp.where(r == 1, nablaB.at[q].set(0), nablaB)

        a_update = nablaA * a + nablaB * bflip
        b_update = nablaA * b + nablaB * aflip

        # weight decay
        a_update += 1e-4 * a
        b_update += 1e-4 * b

        # lr
        if step_normalizer == "2nd":
            mu = precond_lr / add_eps(jnp.max(u + v))
        else:
            mu = precond_lr / add_eps(
                jnp.maximum(jnp.max(jnp.abs(nablaA)), jnp.max(jnp.abs(nablaB)))
            )

        a_update *= -mu
        b_update *= -mu

        return a_update, b_update


def _precond_grad_Xmat_math(a, b, g):
    """
    Preconditioning gradient g with Q = diag(a) + adiag(b).

    All variables here are either matrices or column vectors.
    """
    ab = a * b
    return (a * a + flip(b * b)) * g + (ab + flip(ab)) * flip(g)


