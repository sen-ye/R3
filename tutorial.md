## Training script
We provide a training script in `scripts/train_rl.sh`. You can modify the script to suit your own needs. We detail how to train R3 on various datasets below.

The hyperparameters are defined as follows:
```text
kl_weight_text: KL divergence weight for text reasoning tokens
kl_weight_image: KL divergence weight for image tokens in Mix-GRPO
tune_text_cot: Whether to tune the text reasoning tokens using GRPO
tune_image_cot: Whether to tune the image reasoning tokens using Mix-GRPO
group_size: Group size for a prompt
policy_group_size: how many GPUs constitute a policy group (per GPU batch size = group_size // policy_group_size)
max_output_token_n_gen: maximum number of text reasoning tokens in the reasoning stage
max_output_token_n_edit: maximum number of text reasoning tokens in the reflect stage
num_timesteps_gen: number of diffusion timesteps in the reasoning stage
num_timesteps_edit: number of diffusion timesteps in the refine(edit) stage
enable_sde_gen: enable stochastic diffusion sampling for the reasoning stage
enable_sde_edit: enable stochastic diffusion sampling for the refine(edit) stage
```


## GenEval dataset
### Reward Server setup
We adopt the GenEval reward server provided by Flow-GRPO. Please follow the instructions in [Flow-GRPO GenEval Server](https://github.com/yifan123/reward-server) to setup the reward server on your own server.

### Training script
Set the following parameters in the training script:
```bash
train_data_path="./data/rl_train/train_geneval.jsonl"
val_data_path="./data/rl_test/test_geneval.jsonl"
dataset_name="geneval"
reward_fn="geneval"
reward_server_urls="192.168.1.100" # set to your own reward server IP, if have multiple IPs, use comma to seperate
reward_server_port="1001,1002" # set to your own reward server port, if have multiple ports, use comma to seperate
```

## GenEval++ dataset
### Reward Server setup
We adopt Qwen2.5-VL-72B-Instruct-AWQ as the reward model. Please download the model on huggingface [Qwen2.5-VL-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ) and deploy it using [vllm](https://github.com/vllm-project/vllm).

### Training script
Set the following parameters in the training script:
```bash
train_data_path="./data/rl_train/train_geneval++.jsonl"
val_data_path="./data/rl_test/test_geneval++.jsonl"
dataset_name="geneval_plus"
reward_fn="geneval_plus"
reward_server_urls="192.168.1.100" # set to your own reward server IP,  if have multiple IPs, use comma to seperate
reward_server_port="1001,1002" # set to your own reward server port, if have multiple ports, use comma to seperate
```


## TIIF dataset
### Reward Server setup
We adopt Qwen2.5-VL-72B-Instruct-AWQ as the reward model. Please download the model on huggingface [Qwen2.5-VL-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ) and deploy it using [vllm](https://github.com/vllm-project/vllm).

### Training script
Set the following parameters in the training script:
```bash
train_data_path="./data/rl_train/train_tiif.jsonl"
val_data_path="./data/rl_test/test_tiif.jsonl"
dataset_name="tiif"
reward_fn="tiif"
reward_server_urls="192.168.1.100" # set to your own reward server IP, if have multiple IPs, use comma to seperate
reward_server_port="1001,1002" # set to your own reward server port, if have multiple ports, use comma to seperate
```