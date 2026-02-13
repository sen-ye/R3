# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import datetime
import pytz
import torch
import torch.distributed as dist
from typing import List, Dict, Any, Union, Tuple, Optional
import html
from PIL import Image


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir


def compute_advantages_over_policy_groups(rewards, policy_group):
    # 确保rewards是浮点数类型
    rewards = rewards.float()    
    # 收集所有rank的统计信息
    world_size = dist.get_world_size(policy_group)
    group_rewards = [torch.zeros_like(rewards) for _ in range(world_size)]
    dist.all_gather(group_rewards, rewards, group=policy_group)
    group_rewards = torch.cat(group_rewards, dim=0)
    group_mean = group_rewards.mean(dim=0)
    group_std = group_rewards.std(dim=0)
    advantage = (rewards - group_mean) / (group_std + 1e-6)
    return advantage, group_mean, group_std


def compute_advantages_for_multi_round_rollout(
    per_sample_rewards: List[List[float]], 
    current_policy_group: Optional[int] = None,
    # 第一轮配置
    first_subtract_mean: bool = True,
    first_divide_std: bool = False,
    first_scaler: float = 1.0,
    # 后续轮次配置
    later_operations: str = "subtract_prev,subtract_mean",
    later_scaler: float = 1.0,
) -> Tuple[List[List[float]], Dict[str, float]]:
    """
    多轮rollout RL advantage计算函数
    
    Args:
        per_sample_rewards: 每个样本的多轮rewards，形状为 [num_samples, num_rounds]
        current_policy_group: 分布式训练的policy group，用于advantage计算中的group均值
        first_subtract_mean: 第一轮是否减均值
        first_divide_std: 第一轮是否除标准差
        first_scaler: 第一轮是在advantage上的缩放系数
        later_operations: 后续轮次的操作序列，用逗号分隔，可选操作：
                         - "subtract_prev": 减去上一轮reward
                         - "subtract_mean": 减去当前轮group均值
                         - "divide_std": 除以当前轮group标准差
                         例如: "subtract_prev,subtract_mean" 或 "subtract_mean,subtract_prev"
    
    Returns:
        per_sample_advantages: 每个样本的多轮advantages
        improvement_stats: 包含每轮全局平均reward和相对提升的字典
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_sample_advantages = []
    num_rounds = len(per_sample_rewards[0]) if per_sample_rewards else 0
    
    # 解析后续轮次的操作序列
    operations = [op.strip() for op in later_operations.split(',') if op.strip()]
    
    # 存储每轮的全局平均reward（用于logging和计算提升）
    global_round_mean_rewards = []
    improvement_stats = {
        "round_mean_rewards": {},
        "round_improvements": {},
        "total_improvement": 0.0
    }
    
    def compute_group_std(data_tensor, group=None):
        """计算group标准差，通过all_gather所有数据"""
        if group is not None:
            # 获取group大小
            group_size = dist.get_world_size(group)
            
            # all_gather所有rank的数据
            gathered_data = [torch.zeros_like(data_tensor) for _ in range(group_size)]
            dist.all_gather(gathered_data, data_tensor, group=group)
            
            # 拼接所有数据
            all_data = torch.cat(gathered_data, dim=0)
            
            # 计算真正的全局标准差
            return all_data.std()
        else:
            # 没有group，只计算本rank的标准差
            return data_tensor.std()
    
    # 处理第一轮
    if num_rounds > 0:
        first_round_rewards = torch.tensor(
            [sample_reward[0] for sample_reward in per_sample_rewards], 
            device=device, dtype=torch.float
        )
        
        # 计算全局平均reward（用于logging）
        first_round_global_mean = first_round_rewards.mean()
        # 全局均值总是在所有rank上计算
        if dist.is_initialized():
            dist.all_reduce(first_round_global_mean, op=dist.ReduceOp.AVG)
        
        global_round_mean_rewards.append(first_round_global_mean.item())
        improvement_stats["round_mean_rewards"]["round_0"] = first_round_global_mean.item()
        
        first_round_advantages = first_round_rewards.clone()
        
        # 减group均值（用于advantage计算）
        if first_subtract_mean:
            group_mean = first_round_rewards.mean()
            # group均值只在指定的policy group内计算
            if current_policy_group is not None:
                dist.all_reduce(group_mean, op=dist.ReduceOp.AVG, group=current_policy_group)
            first_round_advantages = first_round_advantages - group_mean
        
        # 除group标准差
        std = compute_group_std(first_round_advantages, current_policy_group)
        round_0_global_mean_std = std.clone()
        dist.all_reduce(round_0_global_mean_std, op=dist.ReduceOp.AVG)
        improvement_stats["round_0_global_mean_std"] = round_0_global_mean_std.item()
        if first_divide_std:
            first_round_advantages = first_round_advantages / (std + 1e-6)
        
        first_round_advantages_list = first_round_advantages.cpu().tolist()
        for adv in first_round_advantages_list:
            per_sample_advantages.append([adv * first_scaler])
    
    # 处理后续轮次
    for round_idx in range(1, num_rounds):
        current_round_rewards = torch.tensor(
            [per_sample_rewards[sample_idx][round_idx] for sample_idx in range(len(per_sample_rewards))],
            device=device, dtype=torch.float
        )
        
        # 计算全局平均reward（用于logging和计算提升）
        current_round_global_mean = current_round_rewards.mean()
        # 全局均值总是在所有rank上计算
        if dist.is_initialized():
            dist.all_reduce(current_round_global_mean, op=dist.ReduceOp.AVG)
        
        global_round_mean_rewards.append(current_round_global_mean.item())
        
        round_key = f"round_{round_idx}"
        improvement_stats["round_mean_rewards"][round_key] = current_round_global_mean.item()
        
        # 计算相对于上一轮的全局提升
        if len(global_round_mean_rewards) >= 2:
            improvement = global_round_mean_rewards[-1] - global_round_mean_rewards[-2]
            improvement_stats["round_improvements"][f"round_{round_idx}_vs_{round_idx - 1}"] = improvement

        
        # 初始化当前轮的advantages
        round_advantages = current_round_rewards.clone()
        
        # 按配置顺序执行操作
        for op in operations:
            if op == "subtract_prev":
                # 减去上一轮reward
                prev_round_rewards = torch.tensor(
                    [per_sample_rewards[sample_idx][round_idx - 1] for sample_idx in range(len(per_sample_rewards))],
                    device=device, dtype=torch.float
                )
                round_advantages = round_advantages - prev_round_rewards
                
            elif op == "subtract_mean":
                # 减去当前轮的group均值（用于advantage计算）
                group_mean = round_advantages.mean()
                # group均值只在指定的policy group内计算
                if current_policy_group is not None:
                    dist.all_reduce(group_mean, op=dist.ReduceOp.AVG, group=current_policy_group)
                round_advantages = round_advantages - group_mean
                
            elif op == "divide_std":
                # 除以group标准差（用于advantage计算）
                std = compute_group_std(current_round_rewards, current_policy_group)
                round_advantages = round_advantages / (std + 1e-6)

            elif op == "subtract_mean_global":
                # 减去全局均值（用于advantage计算）
                global_mean = round_advantages.mean()
                if dist.is_initialized():
                    dist.all_reduce(global_mean, op=dist.ReduceOp.AVG)
                round_advantages = round_advantages - global_mean
        
        # 添加到结果中
        round_advantages_list = round_advantages.cpu().tolist()
        for sample_idx, adv in enumerate(round_advantages_list):
            per_sample_advantages[sample_idx].append(adv * later_scaler)
    
    # 计算总的全局提升（最后一轮相对于第一轮）
    if len(global_round_mean_rewards) >= 2:
        improvement_stats["total_improvement"] = global_round_mean_rewards[-1] - global_round_mean_rewards[0]
    
    return per_sample_advantages, improvement_stats


class EvalStats:
    def __init__(self, basic_info: List[dict] = None, track_keys: List[str] = None):
        self.basic_info = basic_info
        self.track_keys = track_keys
        self.track_stats = {}
        self.track_stats_count = {}
        self.mean_stats = {}
        if track_keys is not None:
            for key in track_keys:
                self.track_stats[key] = 0
                self.track_stats_count[key] = 0

    def update(self, key: str, value: Union[float, int]):
        if key in self.track_keys:
            self.track_stats[key] += float(value)
            self.track_stats_count[key] += 1

    def aggregate(self, normalize=True, log_overall=True):
        mean_score = 0
        for key in self.track_keys:
            self.track_stats[key] = torch.tensor(self.track_stats[key], device="cuda", dtype=torch.float).mean()
            self.track_stats_count[key] = torch.tensor(self.track_stats_count[key], device="cuda", dtype=torch.float).mean()
            dist.all_reduce(self.track_stats[key], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.track_stats_count[key], op=dist.ReduceOp.SUM)
            if normalize and self.track_stats_count[key] > 0:
                self.mean_stats[f"{key}"] = self.track_stats[key] / self.track_stats_count[key]
            else:
                self.mean_stats[f"{key}"] = self.track_stats[key]
            mean_score += self.mean_stats[key]
        if log_overall:
            self.mean_stats["Overall"] = mean_score / len(self.track_keys)
        return self.mean_stats
    
    def to_dict(self):
        return_dict = {}
        if self.basic_info is not None:
            return_dict.update({k: v for k, v in self.basic_info})
        if self.mean_stats is not None:
            return_dict.update(self.mean_stats)
        return return_dict





def save_result_as_html(
    outputs: List[Dict[str, Any]],
    curr_step: int,
    results_dir: str,
    mode: str = "train",
    consolidate: bool = True,
):
    """
    Save intermediate results for the current rank as a robust HTML file and
    consolidates all reports on rank 0.

    Args:
        outputs (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
            represents an item to display. Values can be text, numbers, or PIL Images.
        curr_step (int): The current step in the process (e.g., training step).
        results_dir (str): The main directory to save results in.
        mode (str, optional): The mode, e.g., 'train', 'val', 'test'. Defaults to "train".
    """
    # Get rank and world size from torch.distributed, or default for non-distributed scenarios
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Create a directory for the current step and mode, only on rank 0
    output_dir = os.path.join(results_dir, f"{mode}_step_{curr_step}")
    if rank == 0 and consolidate:
        os.makedirs(output_dir, exist_ok=True)
    
    # Synchronize all processes to ensure the directory is created before proceeding
    if dist.is_initialized() and consolidate:
        dist.barrier()

    # Process outputs to save images and prepare data for HTML
    processed_outputs = []
    for i, item_dict in enumerate(outputs):
        processed_dict = {}
        for key, value in item_dict.items():
            # Sanitize key for use in filenames
            sanitized_key = "".join(c for c in key if c.isalnum() or c in ('_', '-')).rstrip()
            if not sanitized_key:
                sanitized_key = "unnamed_key" # Fallback for empty keys
            
            if isinstance(value, Image.Image):
                # Save image and store its relative path
                img_filename = f"rank{rank}_item{i}_{sanitized_key}.png"
                img_path = os.path.join(output_dir, img_filename)
                value.save(img_path)
                processed_dict[key] = os.path.basename(img_path)
            else:
                # Keep other data types as they are
                processed_dict[key] = value
        processed_outputs.append(processed_dict)

    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Results - {mode.capitalize()} - Rank {rank} - Step {curr_step}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #212529; }}
        .header {{ text-align: center; border-bottom: 2px solid #dee2e6; padding-bottom: 20px; margin-bottom: 40px; position: relative; }}
        .header h1 {{ font-size: 2.5em; color: #343a40; }}
        .header p {{ font-size: 1.2em; color: #6c757d; }}
        .navigation-hint {{ position: absolute; top: 10px; right: 10px; font-size: 0.9em; color: #6c757d; background: #e9ecef; padding: 5px 10px; border-radius: 4px; }}
        .item-container {{ margin-bottom: 40px; border: 1px solid #dee2e6; padding: 25px; border-radius: 8px; background-color: #ffffff; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }}
        .item-container h2 {{ font-size: 1.8em; color: #495057; border-bottom: 1px solid #e9ecef; padding-bottom: 10px; margin-top: 0; }}
        .kv-pair {{ display: grid; grid-template-columns: 200px 1fr; gap: 15px; align-items: start; padding: 10px 0; border-bottom: 1px solid #f1f3f5; }}
        .kv-pair:last-child {{ border-bottom: none; }}
        .key {{ font-weight: bold; color: #007bff; }}
        .value img {{ max-width: 100%; max-height: 400px; border-radius: 4px; border: 1px solid #dee2e6; cursor: pointer; transition: transform 0.2s; }}
        .value img:hover {{ transform: scale(1.05); }}
        .value pre {{ background-color: #e9ecef; padding: 15px; border-radius: 4px; white-space: pre-wrap; word-break: break-all; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="navigation-hint">Use ← → keys to navigate</div>
        <h1>{mode.capitalize()} Results</h1>
        <p>Step: {curr_step} | Rank: {rank} / {world_size - 1}</p>
    </div>
    
    <script>
        document.addEventListener('keydown', function(event) {{
            const currentRank = {rank};
            const worldSize = {world_size};
            
            if (event.key === 'ArrowLeft') {{
                // Go to previous rank
                const prevRank = currentRank === 0 ? worldSize - 1 : currentRank - 1;
                const prevFile = `results_rank_${{prevRank}}.html`;
                window.location.href = prevFile;
            }} else if (event.key === 'ArrowRight') {{
                // Go to next rank
                const nextRank = currentRank === worldSize - 1 ? 0 : currentRank + 1;
                const nextFile = `results_rank_${{nextRank}}.html`;
                window.location.href = nextFile;
            }}
        }});
    </script>
"""
    
    if not processed_outputs:
        html_content += "<p>No output data provided.</p>"
    else:
        for i, p_dict in enumerate(processed_outputs):
            html_content += f'<div class="item-container"><h2>Item {i + 1}</h2>'
            for key, value in p_dict.items():
                html_content += '<div class="kv-pair">'
                html_content += f'<div class="key">{html.escape(str(key))}:</div>'
                
                # Check original type to decide how to render
                original_value = outputs[i].get(key)
                if isinstance(original_value, Image.Image):
                    # 'value' here is the path
                    html_content += f'<div class="value"><img src="{html.escape(value)}" alt="{html.escape(str(key))}" onclick="this.requestFullscreen()"></div>'
                else:
                    html_content += f'<div class="value"><pre>{html.escape(str(value))}</pre></div>'
                
                html_content += '</div>'
            html_content += '</div>'
        
    html_content += "</body></html>"
    
    # Save HTML file
    html_filename = f"results_rank_{rank}.html"
    html_file_path = os.path.join(output_dir, html_filename)
    try:
        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except IOError as e:
        print(f"Error writing HTML file for rank {rank}: {e}")

    # Synchronize all ranks to ensure all individual reports are written
    if dist.is_initialized() and consolidate:
        dist.barrier()

    # On rank 0, consolidate all reports into a single index.html
    if rank == 0 and consolidate:
        consolidate_html_reports(
            results_dir=results_dir,
            curr_step=curr_step,
            world_size=world_size,
            mode=mode,
        )


def consolidate_html_reports(
    results_dir: str,
    curr_step: int,
    world_size: int,
    mode: str = "train",
):
    """
    Consolidates HTML reports from all ranks into a single index.html file.
    This function should only be executed on rank 0.
    """
    # Guard to ensure this only runs on the main process
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    base_dir = os.path.join(results_dir, f"{mode}_step_{curr_step}")

    # Find all individual HTML report files that match the expected format
    report_files = []
    for rank in range(world_size):
        filepath = os.path.join(base_dir, f"results_rank_{rank}.html")
        if os.path.exists(filepath):
            report_files.append(os.path.basename(filepath))

    if not report_files:
        print(f"No HTML reports found in {base_dir} to consolidate.")
        return

    # Sort by rank number to ensure correct ordering (e.g. 2 before 10)
    sorted_report_files = sorted(report_files, key=lambda f: int(f.replace('results_rank_', '').replace('.html', '')))

    # Generate a main index.html file
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Consolidated Report - {mode.capitalize()} - Step {curr_step}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f8f9fa; color: #212529; }}
        .header {{ background-color: #343a40; color: white; padding: 20px 40px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 2.2em; }}
        .header p {{ margin: 5px 0 0; font-size: 1.1em; opacity: 0.8; }}
        .nav {{ background-color: #fff; padding: 15px 40px; border-bottom: 1px solid #dee2e6; position: sticky; top: 0; z-index: 1000; }}
        .nav ul {{ list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 20px; }}
        .nav a {{ text-decoration: none; color: #007bff; font-weight: 500; }}
        .nav a:hover {{ color: #0056b3; }}
        .container {{ padding: 40px; }}
        .grid-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 40px; }}
        .rank-frame {{ border: 1px solid #dee2e6; border-radius: 8px; background: #ffffff; box-shadow: 0 4px 8px rgba(0,0,0,0.05); overflow: hidden; display: flex; flex-direction: column; }}
        .rank-frame h2 {{ margin: 0; padding: 15px 20px; background: #f1f3f5; border-bottom: 1px solid #dee2e6; font-size: 1.2em; }}
        .rank-frame a {{ text-decoration: none; color: inherit; }}
        .rank-frame iframe {{ width: 100%; height: 80vh; border: none; flex-grow: 1; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Consolidated Report</h1>
        <p>{mode.capitalize()} | Step {curr_step}</p>
    </div>
    
    <nav class="nav">
        <ul>
"""
    # Add navigation links
    for fname in sorted_report_files:
        try:
            rank_num_str = fname.replace('results_rank_', '').replace('.html', '')
            int(rank_num_str)  # Validate that it's a number
        except (ValueError, IndexError):
            continue # Skip files with unexpected names
        html_content += f'<li><a href="#{rank_num_str}">Rank {rank_num_str}</a></li>'

    html_content += """
        </ul>
    </nav>

    <div class="container">
        <div class="grid-container">
"""
    # Add iframes for each report
    for fname in sorted_report_files:
        try:
            rank_num_str = fname.replace('results_rank_', '').replace('.html', '')
            int(rank_num_str) # Validate
        except (ValueError, IndexError):
            continue
            
        html_content += f"""
        <div id="{rank_num_str}" class="rank-frame">
            <h2><a href="{html.escape(fname)}" target="_blank">Report for Rank {rank_num_str}</a></h2>
            <iframe src="{html.escape(fname)}"></iframe>
        </div>
"""
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    # Save the consolidated index file
    index_path = os.path.join(base_dir, "index.html")
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Consolidated HTML report saved to: {index_path}")
    except IOError as e:
        print(f"Error writing consolidated HTML report: {e}")