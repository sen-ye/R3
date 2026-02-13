#!/bin/bash
export num_nodes=1
export node_rank=0
export master_addr="localhost"
export master_port=29509

# Model paths - replace with your actual paths
export BAGEL_ROOT=./ckpt/BAGEL-7B-MoT

export llm_path="${BAGEL_ROOT}/llm_config.json"
export vit_path="${BAGEL_ROOT}/vit_config.json"
export vae_path="${BAGEL_ROOT}/ae.safetensors"
# Output paths - will be suffixed with dataset name in training scripts
export output_path="output"
export resume_from="${BAGEL_ROOT}"
export use_flex=True

# Training command
run_name="online_rl_train"
mkdir -p ${output_path}/${run_name}
std_out_path=${output_path}/${run_name}/std_out.txt
std_err_path=${output_path}/${run_name}/std_err.txt
# checkpoint and rollout results will be saved to mydir/run_name/checkpoints and mydir/run_name/
mydir=${output_path}/${run_name}
lr=5e-6
debug=False
# data
train_data_path="./data/rl_train/train_geneval++.jsonl"
val_data_path="./data/rl_test/test_geneval++.jsonl"
dataset_name="geneval_plus"
# training related
num_shard=8
save_every=50
eval_freq=50
total_steps=1000
max_num_tokens=32768
ce_weight=1.0
mse_weight=2.0
kl_weight_text=0.001
kl_weight_image=0.001
tune_text_cot=True
tune_image_cot=True
format_reward_weight=1.0
pack_start_round_idx=0
# rollout related
rounds=2
group_size=16 # per card batch size = group_size // policy_group_size
policy_group_size=4 # for distributed rollout
think_mode=True
max_output_token_n_gen=196
max_output_token_n_edit=384
rollout_vis_step=20
text_temperature=0.8
top_k=50
num_timesteps_gen=15
num_timesteps_edit=18
eta_mode="monotonic"
constant_eta_gen=0.7
constant_eta_edit=1.0
cfg_text_scale=4.0
cfg_img_scale=1.5
cfg_renorm_type_gen="global"
cfg_renorm_type_edit="text_channel"
enable_sde_gen=True
enable_sde_edit=True
# reweard server related
reward_fn="geneval_plus"
reward_server_urls="29.191.192.70"
reward_server_port="1001,1002,1003,1004,1005,1006,1007,1008"
api_key="EMPTY"
reward_model_name="Qwen2.5-VL-72B-Instruct-AWQ"
client_type="openai"
# freeze related
freeze_vit=True
freeze_und=False
freeze_text_transformer=False
freeze_diffusion_transformer=False
warmup_steps=30
# load pretrained model
resume_model_only=True
finetune_from_ema=True
ref_model_path=$resume_from

torchrun \
    --nnodes=$num_nodes \
    --node_rank=$node_rank \
    --nproc_per_node=8 \
    --master_addr=$master_addr \
    --master_port=$master_port \
    train/online_rl.py \
    --dataset_name ${dataset_name} \
    ${llm_path:+--llm_path $llm_path} \
    ${vae_path:+--vae_path $vae_path} \
    ${vit_path:+--vit_path $vit_path} \
    ${use_flex:+--use_flex $use_flex} \
    --train_data_path ${train_data_path} \
    --val_data_path ${val_data_path} \
    --debug ${debug} \
    --mydir $mydir \
    --results_dir ${output_path}/${run_name} \
    --checkpoint_dir ${output_path}/${run_name}/checkpoints \
    ${resume_from:+--resume_from $resume_from} \
    --resume_model_only ${resume_model_only} \
    --finetune_from_ema ${finetune_from_ema} \
    --llm_qk_norm True \
    --tie_word_embeddings False \
    --layer_module "Qwen2MoTDecoderLayer" \
    --max_num_tokens 32768 \
    --max_output_token_n_gen ${max_output_token_n_gen} \
    --max_output_token_n_edit ${max_output_token_n_edit} \
    --rounds ${rounds} \
    --group_size ${group_size} \
    --policy_group_size ${policy_group_size} \
    --think_mode ${think_mode} \
    --max_num_tokens ${max_num_tokens} \
    --reward_fn ${reward_fn} \
    --reward_server_urls ${reward_server_urls} \
    --reward_server_port ${reward_server_port} \
    --max_latent_size 64 \
    --wandb_name ${run_name} \
    --wandb_offline 0 \
    --num_shard ${num_shard} \
    --total_steps ${total_steps} \
    --save_every ${save_every} \
    --ce_weight ${ce_weight} \
    --mse_weight ${mse_weight} \
    --kl_weight_text ${kl_weight_text} \
    --kl_weight_image ${kl_weight_image} \
    --freeze_vit ${freeze_vit} \
    --freeze_text_transformer ${freeze_text_transformer} \
    --freeze_und ${freeze_und} \
    --freeze_diffusion_transformer ${freeze_diffusion_transformer} \
    --tune_text_cot ${tune_text_cot} \
    --tune_image_cot ${tune_image_cot} \
    --rollout_vis_step ${rollout_vis_step} \
    --text_temperature ${text_temperature} \
    --top_k ${top_k} \
    --num_timesteps_gen ${num_timesteps_gen} \
    --num_timesteps_edit ${num_timesteps_edit} \
    --cfg_text_scale ${cfg_text_scale} \
    --cfg_img_scale ${cfg_img_scale} \
    --cfg_renorm_type_gen ${cfg_renorm_type_gen} \
    --cfg_renorm_type_edit ${cfg_renorm_type_edit} \
    --lr ${lr} \
    --warmup_steps ${warmup_steps} \
    --eps 1e-8 \
    --log_every 1 \
    --beta2 0.95 \
    --format_reward_weight ${format_reward_weight} \
    --eta_mode ${eta_mode} \
    --constant_eta_gen ${constant_eta_gen} \
    --constant_eta_edit ${constant_eta_edit} \
    --pack_start_round_idx ${pack_start_round_idx} \
    --enable_sde_gen ${enable_sde_gen} \
    --enable_sde_edit ${enable_sde_edit} \
    --eval_freq ${eval_freq} \
    --client_type ${client_type} \
    --reward_api_key ${api_key} \
    --reward_model_name ${reward_model_name} 1>${std_out_path} 2>${std_err_path}