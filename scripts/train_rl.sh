#!/bin/bash
set -euo pipefail

num_nodes="${NNODES:-1}"
node_rank="${NODE_RANK:-0}"
nproc_per_node="${NPROC_PER_NODE:-8}"
master_addr="${MASTER_ADDR:-localhost}"
master_port="${MASTER_PORT:-29509}"

# Model paths - replace with your actual paths
export BAGEL_ROOT="${MODEL_PATH:-./ckpt/BAGEL-7B-MoT}"
llm_path="${LLM_PATH:-${BAGEL_ROOT}/llm_config.json}"
vit_path="${VIT_PATH:-${BAGEL_ROOT}/vit_config.json}"
vae_path="${VAE_PATH:-${BAGEL_ROOT}/ae.safetensors}"

# Output paths
# Rollout logs and visualizations will be saved under:
#   ${LOG_DIR}/${WANDB_PROJECT}/${EXP_NAME}
# Checkpoints will be saved under:
#   ${LOG_DIR}/${WANDB_PROJECT}/${EXP_NAME}/ckpts
# Local wandb files will be saved under:
#   ${LOG_DIR}/${WANDB_PROJECT}/${EXP_NAME}/wandb
#
# Training command
# `RUN_NAME` is kept for backward compatibility and will also be used as the
# default value of `EXP_NAME` / `WANDB_NAME`.
run_name="${RUN_NAME:-online_rl_train}"
exp_name="${EXP_NAME:-${run_name}}"
wandb_name="${WANDB_NAME:-${run_name}}"
wandb_project="${WANDB_PROJECT:-R3}"
log_dir="${LOG_DIR:-./output}"
exp_dir="${log_dir}/${wandb_project}/${exp_name}"
export WANDB_DIR="${WANDB_DIR:-${exp_dir}/wandb}"
mkdir -p "${exp_dir}" "${WANDB_DIR}"
std_out_path="${exp_dir}/std_out.txt"
std_err_path="${exp_dir}/std_err.txt"

# data
train_data_path="${TRAIN_DATA_PATH:-./data/rl_train/train_geneval++.jsonl}"
val_data_path="${VAL_DATA_PATH:-./data/rl_test/test_geneval++.jsonl}"
dataset_name="${DATASET_NAME:-geneval_plus}"

# training related
lr="${LR:-5e-6}"
debug="${DEBUG:-False}"
num_shard="${NUM_SHARD:-8}"
save_every="${SAVE_EVERY:-50}"
eval_freq="${EVAL_FREQ:-50}"
total_steps="${TOTAL_STEPS:-1000}"
ce_weight="${CE_WEIGHT:-1.0}"
mse_weight="${MSE_WEIGHT:-2.0}"
kl_weight_text="${KL_WEIGHT_TEXT:-0.001}"
kl_weight_image="${KL_WEIGHT_IMAGE:-0.001}"
tune_text_cot="${TUNE_TEXT_COT:-True}"
tune_image_cot="${TUNE_IMAGE_COT:-True}"
format_reward_weight="${FORMAT_REWARD_WEIGHT:-1.0}"
pack_start_round_idx="${PACK_START_ROUND_IDX:-0}"
warmup_steps="${WARMUP_STEPS:-10}"

# rollout related
rounds="${ROUNDS:-2}"
group_size="${GROUP_SIZE:-16}" # per card batch size = group_size // policy_group_size
policy_group_size="${POLICY_GROUP_SIZE:-4}" # for distributed rollout
think_mode="${THINK_MODE:-True}"
max_output_token_n_gen="${MAX_OUTPUT_TOKEN_N_GEN:-196}"
max_output_token_n_edit="${MAX_OUTPUT_TOKEN_N_EDIT:-384}"
text_temperature="${TEXT_TEMPERATURE:-0.9}"
top_k="${TOP_K:-50}"
num_timesteps_gen="${NUM_TIMESTEPS_GEN:-16}"
num_timesteps_edit="${NUM_TIMESTEPS_EDIT:-21}"
eta_mode="${ETA_MODE:-monotonic}"
constant_eta_gen="${CONSTANT_ETA_GEN:-0.7}"
constant_eta_edit="${CONSTANT_ETA_EDIT:-1.0}"
cfg_text_scale="${CFG_TEXT_SCALE:-4.0}"
cfg_img_scale="${CFG_IMG_SCALE:-1.5}"
cfg_renorm_type_gen="${CFG_RENORM_TYPE_GEN:-global}"
cfg_renorm_type_edit="${CFG_RENORM_TYPE_EDIT:-text_channel}"
enable_sde_gen="${ENABLE_SDE_GEN:-True}"
enable_sde_edit="${ENABLE_SDE_EDIT:-True}"

# reward server related
reward_fn="${REWARD_FN:-geneval_plus}"
reward_server_urls="${REWARD_SERVER_URLS:-127.0.0.1}"
reward_server_port="${REWARD_SERVER_PORT:-8008,8009}"
reward_api_key="${REWARD_API_KEY:-EMPTY}"
reward_model_name="${REWARD_MODEL_NAME:-Qwen2.5-VL-72B-Instruct-AWQ}"
client_type="${CLIENT_TYPE:-openai}"

# If you use an external OpenAI-compatible endpoint directly, set:
#   REWARD_SERVER_PORT=-1
#   REWARD_SERVER_URLS=<base_url>

# freeze related
freeze_vit="${FREEZE_VIT:-True}"
freeze_und="${FREEZE_UND:-False}"

# load pretrained model
resume_from="${RESUME_FROM:-${BAGEL_ROOT}}"
resume_model_only="${RESUME_MODEL_ONLY:-True}"
finetune_from_ema="${FINETUNE_FROM_EMA:-True}"

torchrun \
    --nnodes="${num_nodes}" \
    --node_rank="${node_rank}" \
    --nproc_per_node="${nproc_per_node}" \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    train/online_rl.py \
    --dataset_name "${dataset_name}" \
    --llm_path "${llm_path}" \
    --vae_path "${vae_path}" \
    --vit_path "${vit_path}" \
    --train_data_path "${train_data_path}" \
    --val_data_path "${val_data_path}" \
    --debug "${debug}" \
    --log_dir "${log_dir}" \
    --wandb_project "${wandb_project}" \
    --exp_name "${exp_name}" \
    --resume_from "${resume_from}" \
    --resume_model_only "${resume_model_only}" \
    --finetune_from_ema "${finetune_from_ema}" \
    --llm_qk_norm True \
    --tie_word_embeddings False \
    --layer_module "Qwen2MoTDecoderLayer" \
    --max_output_token_n_gen "${max_output_token_n_gen}" \
    --max_output_token_n_edit "${max_output_token_n_edit}" \
    --rounds "${rounds}" \
    --group_size "${group_size}" \
    --policy_group_size "${policy_group_size}" \
    --think_mode "${think_mode}" \
    --reward_fn "${reward_fn}" \
    --reward_server_urls "${reward_server_urls}" \
    --reward_server_port "${reward_server_port}" \
    --max_latent_size 64 \
    --wandb_name "${wandb_name}" \
    --wandb_offline 0 \
    --num_shard "${num_shard}" \
    --total_steps "${total_steps}" \
    --save_every "${save_every}" \
    --ce_weight "${ce_weight}" \
    --mse_weight "${mse_weight}" \
    --kl_weight_text "${kl_weight_text}" \
    --kl_weight_image "${kl_weight_image}" \
    --freeze_vit "${freeze_vit}" \
    --freeze_und "${freeze_und}" \
    --tune_text_cot "${tune_text_cot}" \
    --tune_image_cot "${tune_image_cot}" \
    --text_temperature "${text_temperature}" \
    --top_k "${top_k}" \
    --num_timesteps_gen "${num_timesteps_gen}" \
    --num_timesteps_edit "${num_timesteps_edit}" \
    --cfg_text_scale "${cfg_text_scale}" \
    --cfg_img_scale "${cfg_img_scale}" \
    --cfg_renorm_type_gen "${cfg_renorm_type_gen}" \
    --cfg_renorm_type_edit "${cfg_renorm_type_edit}" \
    --lr "${lr}" \
    --warmup_steps "${warmup_steps}" \
    --eps 1e-8 \
    --log_every 1 \
    --beta2 0.95 \
    --format_reward_weight "${format_reward_weight}" \
    --eta_mode "${eta_mode}" \
    --constant_eta_gen "${constant_eta_gen}" \
    --constant_eta_edit "${constant_eta_edit}" \
    --pack_start_round_idx "${pack_start_round_idx}" \
    --enable_sde_gen "${enable_sde_gen}" \
    --enable_sde_edit "${enable_sde_edit}" \
    --eval_freq "${eval_freq}" \
    --client_type "${client_type}" \
    --reward_api_key "${reward_api_key}" \
    --reward_model_name "${reward_model_name}" \
    --lr_scheduler constant \
    --max_grad_norm 1.0 \
    >"${std_out_path}" 2>"${std_err_path}"
