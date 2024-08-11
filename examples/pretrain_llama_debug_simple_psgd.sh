#! /bin/bash

umask 000

python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --dtype='fp32' \
    --total_steps=20000 \
    --log_freq=50 \
    --save_model_freq=1000000 \
    --save_milestone_freq=1000000 \
    --eval_steps 5 \
    --calc_hessian=True \
    --update_prob=0.1 \
    --l2_reg=0.0001 \
    --load_llama_config='3b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --optimizer.type='psgd' \
    --optimizer.psgd_optimizer.weight_decay=0.0001 \
    --optimizer.psgd_optimizer.lr=0.1 \
    --optimizer.psgd_optimizer.lr_warmup_steps=512 \
    --optimizer.psgd_optimizer.lr_decay_steps=20000 \
    --optimizer.psgd_optimizer.b1=0.0 \
    --optimizer.psgd_optimizer.b2=0.95 \
    --optimizer.psgd_optimizer.clip_gradient=1.0 \
    --optimizer.psgd_optimizer.nesterov=True \
    --optimizer.psgd_optimizer.precond_update_probability=0.1 \
    --optimizer.psgd_optimizer.precond_lr=0.1 \
    --optimizer.psgd_optimizer.precond_init_scale=0.0 \
    --optimizer.psgd_optimizer.max_size_triangular=64 \
    --optimizer.psgd_optimizer.max_skew_triangular=32 \
    --optimizer.psgd_optimizer.normalize=True \
    --optimizer.psgd_optimizer.adaptive=True \
    --optimizer.psgd_optimizer.bf16_momentum=True \
    --optimizer.psgd_optimizer.bf16_preconditioner=False \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.huggingface_dataset.path='HuggingFaceFW/fineweb-edu' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.seq_length=2048 \
    --train_dataset.huggingface_dataset.batch_size=64 \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset.huggingface_dataset.name='sample-100BT' \
    --eval_dataset.type='huggingface' \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.huggingface_dataset.path='HuggingFaceFW/fineweb-edu' \
    --eval_dataset.huggingface_dataset.streaming=True \
    --eval_dataset.huggingface_dataset.seq_length=2048 \
    --eval_dataset.huggingface_dataset.batch_size=64 \
    --eval_dataset.huggingface_dataset.split='train' \
    --eval_dataset.huggingface_dataset.name='sample-10BT' \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="open_llama_evan" \
    --logger.output_dir="$HOME/bucket/evan_llm/output_dir" \
    --logger.wandb_dir="$HOME/bucket/evan_llm/wandb_dir" \
|& tee $HOME/output.txt