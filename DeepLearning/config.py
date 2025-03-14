import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 8

L1_LAMBDA = 100
NUM_EPOCHS = 100
