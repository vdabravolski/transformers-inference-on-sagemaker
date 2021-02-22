from argparse import ArgumentParser
import os
import json
import subprocess
import sys


def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """

    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker

    return world


if __name__ == "__main__":
    # Get initial configuration to select appropriate HuggingFace task and its configuration
    print('Starting training...')
    parser = ArgumentParser()
    parser.add_argument('--train-script', type=str, default='batch_inference.py', help='Config overrides.')
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Following arguments were not recognized and won't be used: {unknown}")

    # Derive parameters of distributed training cluster in Sagemaker
    world = get_training_world()  
              
    # Train script config
    launch_config = [ "python -m torch.distributed.launch", 
                      "--nnodes", str(world['number_of_machines']), "--node_rank", str(world['machine_rank']),
                     "--nproc_per_node", str(world['number_of_processes']), "--master_addr", world['master_addr'], 
                     "--master_port", world['master_port']]
 
    train_config = [args.train_script]
    

    # Concat Pytorch Distributed Launch config and MMdetection config
    joint_cmd = " ".join(str(x) for x in launch_config+train_config)
    print("Following command will be executed: \n", joint_cmd)
    
    process = subprocess.Popen(joint_cmd,  stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    
    while True:
        output = process.stdout.readline()
        
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    rc = process.poll()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=joint_cmd)
        
    sys.exit(process.returncode)