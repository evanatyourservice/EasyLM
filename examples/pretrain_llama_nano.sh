#! /bin/bash

umask 000

python -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,1,1' \
    --dtype='fp32' \
    --total_steps=100000 \
    --log_freq=50 \
    --save_model_freq=1000000 \
    --save_milestone_freq=1000000 \
    --eval_steps 5 \
    --load_llama_config='3b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.01 \
    --optimizer.adamw_optimizer.lr=0.001 \
    --optimizer.adamw_optimizer.lr_warmup_steps=512 \
    --optimizer.adamw_optimizer.lr_decay_steps=100000 \
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
    --logger.output_dir="gs://uscentral1stuff/checkpoints" \
    --logger.wandb_dir="/dev/shm/wandb" \
|& tee $HOME/output.txt