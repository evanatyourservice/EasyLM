#! /bin/bash

umask 000

python -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,1,1' \
    --dtype='bfloat16' \
    --total_steps=200000 \
    --log_freq=25 \
    --save_model_freq=1000000 \
    --save_milestone_freq=1000000 \
    --eval_steps 2 \
    --load_llama_config='3b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --optimizer.type='psgd' \
    --optimizer.psgd_optimizer.weight_decay=0.1 \
    --optimizer.psgd_optimizer.lr=1e-3 \
    --optimizer.psgd_optimizer.end_lr=3e-12 \
    --optimizer.psgd_optimizer.lr_warmup_steps=512 \
    --optimizer.psgd_optimizer.lr_decay_steps=200000 \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.huggingface_dataset.path='HuggingFaceFW/fineweb-edu' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.seq_length=2048 \
    --train_dataset.huggingface_dataset.batch_size=64 \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset.huggingface_dataset.name='sample-100BT' \
    --train_dataset.huggingface_dataset.cache_dir="$HOME/bucket/evan_llm/fw_100b" \
    --train_dataset.huggingface_dataset.queue_size=30 \
    --eval_dataset.type='huggingface' \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.huggingface_dataset.path='HuggingFaceFW/fineweb-edu' \
    --eval_dataset.huggingface_dataset.streaming=True \
    --eval_dataset.huggingface_dataset.seq_length=2048 \
    --eval_dataset.huggingface_dataset.batch_size=64 \
    --eval_dataset.huggingface_dataset.split='train' \
    --eval_dataset.huggingface_dataset.name='sample-10BT' \
    --eval_dataset.huggingface_dataset.cache_dir="$HOME/bucket/evan_llm/fw_10b" \
    --eval_dataset.huggingface_dataset.queue_size=5 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="open_llama_evan" \
    --logger.output_dir="$HOME/bucket/evan_llm/output_dir" \
    --logger.wandb_dir="$HOME/bucket/evan_llm/wandb_dir" \
|& tee $HOME/output.txt