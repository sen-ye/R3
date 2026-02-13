import json
import torch
from torch.utils.data import Dataset, Sampler
    
    
class GenevalPromptDataset(Dataset):
    def __init__(self, dataset):
        self.file_path = dataset
        with open(self.file_path, 'r', encoding='utf-8') as f:
            metadatas = [json.loads(line) for line in f]

        self.metadatas = metadatas  
        self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        idx = idx % len(self.prompts)
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas
    

class TIIFDataset(Dataset):
    def __init__(self, file_path, key="short_description"):
        self.file_path = file_path
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item[key] for item in self.metadatas]
            for item in self.metadatas:
                item['prompt'] = item[key]

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        idx = idx % len(self.prompts)
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas
            

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, world_size, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # 每卡的batch大小
        self.k = k                    # 每个样本重复的次数
        self.world_size = world_size  # 总卡数
        self.rank = rank              # 当前卡编号
        self.seed = seed              # 随机种子，用于同步
        
        # 计算每个迭代需要的不同样本数
        self.total_samples = self.world_size * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{world_size}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # 不同样本数
        self.epoch=0

    def __iter__(self):
        while True:
            # 生成确定性的随机序列，确保所有卡同步
            g = torch.Generator(device="cuda")
            g.manual_seed(self.seed + self.epoch)
            # print('epoch', self.epoch)
            # 随机选择m个不同的样本
            indices = torch.randperm(len(self.dataset), generator=g, device="cuda")[:self.m].tolist()
            # print(self.rank, 'indices', indices)
            # 每个样本重复k次，生成总样本数n*b
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # 打乱顺序确保均匀分配
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g, device="cuda").tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            # print(self.rank, 'shuffled_samples', shuffled_samples)
            # 将样本分割到各个卡
            per_card_samples = []
            for i in range(self.world_size):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            # print(self.rank, 'per_card_samples', per_card_samples[self.rank])
            # 返回当前卡的样本索引
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # 用于同步不同 epoch 的随机状态