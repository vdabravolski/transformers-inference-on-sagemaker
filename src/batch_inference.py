import os
import argparse
import json
from transformers import pipeline
import torch
import torch.distributed as dist
from launcher import get_training_world
from torch.utils.data import DistributedSampler, DataLoader
from datasets import load_dataset
import wandb
from custom_data_utils import DistributedEvalSampler, DummyDataset


def _get_global_rank(local_rank):
    hosts = json.loads(os.environ["SM_HOSTS"])
    host_rank = hosts.index(os.environ["SM_CURRENT_HOST"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    
    return host_rank*num_gpus+local_rank

def _get_dataset(local_rank, data_split, cache_dir=os.environ["SM_INPUT_DIR"], dummy_dataset=False, download_mode="reuse_dataset_if_exists"):
    if dummy_dataset:
        return DummyDataset()
    
    return load_dataset("amazon_polarity", cache_dir=cache_dir, download_mode=download_mode, split=data_split)

def _setup_wandb(args):
    if args.wandb_api_key is not None:
        os.environ["WANDB_API_KEY"]=args.wandb_api_key
        os.environ["WANDB_PROJECT"]=args.wandb_project
        wandb.init()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--inference-batch", type=int, default=16)
    parser.add_argument("--data-split", type=str, default='test')
    parser.add_argument("--dummy-dataset", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], 
                        default=False, help="whether to use dummy dataset or not")
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()
    
    print(f"Running inference in process with global_rank={_get_global_rank(args.local_rank)} and local_rank={args.local_rank}")
    
    world = get_training_world()
    global_rank = _get_global_rank(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', 
                                         init_method=f'tcp://{world["master_addr"]}:{world["master_port"]}', 
                                         rank=global_rank, 
                                         world_size=world["size"])
    
    # Downloading model and tokenizer in one process. 
    if args.local_rank==0:
        _setup_wandb(args)
        print("downloading model/tokenizer in local_rank=0 process")
        inference_pipeline = pipeline('sentiment-analysis', device=args.local_rank)
        dataset = _get_dataset(args.local_rank, args.data_split, dummy_dataset=args.dummy_dataset)

    # Other have to wait
    torch.distributed.barrier()
    if args.local_rank!=0:
        print("Initializing pre-downloaded model/tokenizer in local_rank!=0 processes")
        inference_pipeline = pipeline('sentiment-analysis', device=args.local_rank)
        dataset = _get_dataset(args.local_rank, args.data_split, dummy_dataset=args.dummy_dataset, download_mode="reuse_cache_if_exists",)
    
    sampler = DistributedEvalSampler(dataset, num_replicas=world["size"], rank=global_rank) if world["size"]>1 else None
    loader = DataLoader(dataset, batch_size=args.inference_batch, shuffle=(sampler is None), sampler=sampler)
    
    # Synchronizing across all processes before starting inference
    torch.distributed.barrier()
    for batch_id, data in enumerate(loader):
        model_output = inference_pipeline(data['content'])
        print(f"Ran inference for batch id={batch_id} in process with global_rank={global_rank}.")
    
if __name__ == "__main__":
    main()