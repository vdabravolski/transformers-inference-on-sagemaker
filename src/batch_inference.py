import os
import argparse
import json
from transformers import pipeline
import torch
import torch.distributed as dist
from launcher import get_training_world
from torch.utils.data import DistributedSampler, DataLoader
from datasets import load_dataset

def _get_global_rank(local_rank):
    hosts = json.loads(os.environ["SM_HOSTS"])
    host_rank = hosts.index(os.environ["SM_CURRENT_HOST"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    
    return host_rank*num_gpus+local_rank
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--inference-batch", type=int, default=16)
    parser.add_argument("--data-split", type=str, default='test')
    args = parser.parse_args()
    
    print(f"Running inference stub in global rank={_get_global_rank(args.local_rank)}")
    
    world = get_training_world()
    global_rank = _get_global_rank(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', 
                                         init_method=f'tcp://{world["master_addr"]}:{world["master_port"]}', 
                                         rank=global_rank, 
                                         world_size=world["size"])
    
    # Downloading model and tokenizer in one process. 
    if args.local_rank==0:
        print("downloading model/tokenizer in local_rank=0 process")
        inference_pipeline = pipeline('sentiment-analysis')
        dataset = load_dataset("amazon_polarity", cache_dir=os.environ["SM_INPUT_DIR"], split=args.data_split)

    # Other have to wait
    torch.distributed.barrier()
    if args.local_rank!=0:
        print("Initializing pre-downloaded model/tokenizer in local_rank!=0 processes")
        inference_pipeline = pipeline('sentiment-analysis')
        dataset = load_dataset("amazon_polarity", cache_dir=os.environ["SM_INPUT_DIR"], download_mode="reuse_cache_if_exists", split=args.data_split)
    
    sampler = DistributedSampler(dataset) if world["size"]>1 else None
    loader = DataLoader(dataset, batch_size=args.inference_batch, shuffle=(sampler is None), sampler=sampler)
    
    for batch_id, data in enumerate(loader):
        print(f"Dealing with batch id={batch_id} for rank={global_rank}")
        model_output = inference_pipeline(data['content'])
        print(model_output)
    
if __name__ == "__main__":
    main()