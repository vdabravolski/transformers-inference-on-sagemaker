# transformers-inference-on-sagemaker
A sample repo to show how to optimize NLP inference on Amazon SageMaker

# Backlog
1. Add inference code (rank aware) to perform classification pipeline - done, with issues.
2. Add data loader (rank aware) to load the financial sentiment dataset - done, with issues.
3. Add support of SPOT instance (specifically, pay attention to checkpointing)
4. Add relevant metrics (end-to-end inference time, inference cost, ???)
5. Do testing/benchmarking on G4s (single-device and multi-device) and P3s (multi-gpu)


## Current issues:
1. Need to replace default DistributedSampler with custom implementation (e.g. [this one](https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py)) because default sampler augments batches to even out size across nodes. It's undesirable for inference.
2. According to SM Profiler Report, GPU devices are underutilized. Need to dive deeper. We are using inference pipeline and loading it to 
3. We are using DDP to spin up inference processes in multi-node/multi-gpu setup. DDP may have synchonization enforced even during forward passes. Explore [no_sync](https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py#L656) context manager to avoid it. Do we need to wrap models with DDP at all? Right now, it's not the case. 
4. Try to use WANDB for profiling and compare it to SM Profiler.