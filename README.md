# transformers-inference-on-sagemaker
A sample repo to show how to optimize NLP inference on Amazon SageMaker

# Backlog
1. Add inference code (rank aware) to perform classification pipeline (for example, pre-trained FinBERT)
2. Add data loader (rank aware) to load the financial sentiment dataset
3. Add support of SPOT instance (specifically, pay attention to checkpointing)
4. Add relevant metrics (end-to-end inference time, inference cost, ???)
5. Do testing/benchmarking on G4s (single-device and multi-device) and P3s (multi-gpu)
