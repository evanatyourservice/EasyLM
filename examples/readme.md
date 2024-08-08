# How to use the example script

Navagate to tpu engine page and use gloud api to lauch...

Please replace with your own wanbdb key in command

```bash 
gcloud compute tpus tpu-vm ssh LLaMA --zone us-central2-b --worker=all --command 'export WANDB_API_KEY='3dbb68cc5c35d09ec37a06b3ea87fc10131ea5af'  && cd EasyLM-SMD && ./examples/pretrain_llama_3b.sh'
```
