import pprint
from functools import partial
import multiprocessing as mp
from multiprocessing import Queue, Process

from flax.traverse_util import _sorted_items, flatten_dict, _get_params_dict
from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax import traverse_util
from flax.training.train_state import TrainState
import optax
from transformers import AutoTokenizer

from EasyLM.data_adjusted_2 import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG,
    JaxDistributedConfig,
    next_rng,
    match_partition_rules,
    cross_entropy_loss_and_accuracy,
    global_norm,
    get_float_dtype_by_name,
    set_random_seed,
    average_metrics,
    make_shard_and_gather_fns,
    with_sharding_constraint,
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfigurator,
    FlaxLLaMAForCausalLMModule,
)
from EasyLM.optimizers.utils import hessian_helper

import datasets.config as ds_config


mp.set_start_method("spawn", force=True)  # jax-friendly

ds_config.STREAMING_READ_MAX_RETRIES = 86400 // 5  # Retry for 24 hours.
ds_config.STREAMING_READ_RETRY_INTERVAL = 5


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim="1,-1,1",
    dtype="fp32",
    param_dtype="fp32",
    total_steps=10000,
    load_llama_config="",
    update_llama_config="",
    load_checkpoint="",
    load_dataset_state="",
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    calc_hessian=False,
    update_prob=0.1,
    l2_reg=0.0,
    tokenizer="openlm-research/open_llama_3b_v2",
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfigurator.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


def _keep_full(dataset, queue: Queue):
    import datasets.config as ds_config
    ds_config.STREAMING_READ_MAX_RETRIES = 86400 // 5  # Retry for 24 hours.
    ds_config.STREAMING_READ_RETRY_INTERVAL = 5

    for item in dataset:
        queue.put(item)
    queue.put(None)


def prefetch(dataset, n_prefetch):
    queue = Queue(maxsize=n_prefetch)
    p = Process(target=_keep_full, args=(dataset, queue))
    p.daemon = True
    p.start()

    while True:
        item = queue.get()
        if item is None:
            break
        yield item


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)

    if FLAGS.load_dataset_state != "":
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    dataset_prefetch = prefetch(dataset, FLAGS.log_freq + 1)

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_dataset = prefetch(eval_dataset, FLAGS.eval_steps + 1)
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)

    model = FlaxLLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )

    # only lets through kernel weights for weight decay
    _kernels = traverse_util.ModelParamTraversal(lambda p, _: "kernel" in p)

    def _kernel_mask(params):
        all_false = jax.tree.map(lambda _: False, params)
        out = _kernels.update(lambda _: True, all_false)
        return out

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer, weight_decay_mask=_kernel_mask
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch, hess_rng):
        rng_generator = JaxRNG(rng)
        # batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def loss_and_accuracy(params, rngs):
            logits = model.apply(
                params,
                batch["input_tokens"],
                deterministic=False,
                rngs=rngs,
            ).logits
            loss, acc = cross_entropy_loss_and_accuracy(
                logits, batch["target_tokens"], batch["loss_masks"]
            )
            orig_loss = loss

            # palm style z-loss
            z_loss = (
                1e-4 * jnp.square(jax.scipy.special.logsumexp(logits, axis=-1)).mean()
            )
            loss += z_loss

            # l2 regularization
            if FLAGS.l2_reg > 0:
                to_l2 = []
                for key, value in _sorted_items(flatten_dict(_get_params_dict(params))):
                    path = "/" + "/".join(key)
                    if "kernel" in path:
                        to_l2.append(jnp.linalg.norm(value))
                l2_loss = jnp.linalg.norm(jnp.array(to_l2))
                loss += FLAGS.l2_reg * l2_loss

            return loss, (orig_loss, acc)

        rngs = rng_generator(LLaMAConfigurator.rng_keys())
        if FLAGS.calc_hessian:
            hess_rng, subkey = jax.random.split(hess_rng)
            loss_out, grads, hvp, vector, update_precond = hessian_helper(
                subkey,
                train_state.step,
                loss_and_accuracy,
                train_state.params,
                loss_fn_extra_args=(rngs,),
                has_aux=True,
                preconditioner_update_probability=FLAGS.update_prob,
            )
            _, (loss, accuracy) = loss_out

            updates, opt_state = optimizer.update(
                grads,
                train_state.opt_state,
                train_state.params,
                Hvp=hvp,
                vector=vector,
                update_preconditioner=update_precond,
            )
            new_params = optax.apply_updates(train_state.params, updates)

            train_state = train_state.replace(
                step=train_state.step + 1,
                params=new_params,
                opt_state=opt_state,
            )
        else:
            grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
            (_, (loss, accuracy)), grads = grad_fn(train_state.params, rngs)
            train_state = train_state.apply_gradients(grads=grads)

        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info["learning_rate_schedule"](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics, hess_rng

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        # batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            train_state.params,
            batch["input_tokens"],
            deterministic=True,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch["target_tokens"], batch["loss_masks"]
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())

    total_params = sum(
        [np.prod(x.shape) for x in jax.tree.leaves(train_state_shapes.params)]
    )
    print(f"Total number of parameters: {total_params}")

    print("Model shapes:")
    pprint.pprint(
        jax.tree.map(lambda x: x.shape, train_state_shapes), width=120, compact=True
    )

    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), train_state_shapes
    )

    pprint.pprint(train_state_partition, width=120)

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer,
        logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn, in_shardings=PS(), out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params,),
        out_shardings=train_state_partition,
        donate_argnums=(0,),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS(("dp", "fsdp")), PS()),
        out_shardings=(train_state_partition, PS(), PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS(("dp", "fsdp"))),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != "":
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()
        hess_rng = jax.random.PRNGKey(0)

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset_prefetch):
            train_state, sharded_rng, metrics, hess_rng = sharded_train_step(
                train_state, sharded_rng, batch, hess_rng
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if (
                FLAGS.save_milestone_freq > 0
                and (step + 1) % FLAGS.save_milestone_freq == 0
            ):
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
