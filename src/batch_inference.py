import os
import argparse
import json

def _get_global_rank(local_rank):
    hosts = json.loads(os.environ["SM_HOSTS"])
    host_rank = hosts.index(os.environ["SM_CURRENT_HOST"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    
    return host_rank*num_gpus+local_rank
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    
    
    print(f"Running inference stub in global rank={_get_global_rank(args.local_rank)}")
    
if __name__ == "__main__":
    main()