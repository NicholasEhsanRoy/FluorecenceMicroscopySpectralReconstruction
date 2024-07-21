import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.utils.benchmark as benchmark
from time import time
from generator_model import Generator
import torch.profiler
from tqdm import tqdm
from utils import load_checkpoint2

cudnn.benchmark = True

# Define the function to ensure all normalization layers are in eval mode
def set_to_eval(m):
    classname = m.__class__.__name__
    if 'Norm' in classname:
        m.eval()

def set_to_half(m):
    """
    Recursively sets all submodules and parameters of the given module to half precision.
    """
    if isinstance(m, nn.Module):
        for child in m.children():
            set_to_half(child)
    if isinstance(m, nn.Parameter):
        m.data = m.data.half()
        if m.grad is not None:
            m.grad.data = m.grad.data.half()
    for param in m.parameters(recurse=False):
        param.data = param.data.half()
        if param.grad is not None:
            param.grad.data = param.grad.data.half()

# Initialize and prepare the model
generator = Generator(in_channels=1, out_channels=1, features=16).cuda()
load_checkpoint2("model.pth.tar", generator, None, "cuda") ## Change to your path to model


generator.eval()

generator.apply(set_to_eval)

# Create dummy input
dummy_input = torch.randn(1, 1, 240, 300).cuda()  

# To have cudnn.benchmark optimize algorithm for hardware
for _ in tqdm(range(100)):
    _ = generator(dummy_input)
    torch.cuda.synchronize()

# Convert to TorchScript
scripted_model = torch.jit.script(generator)
scripted_model = torch.jit.optimize_for_inference(scripted_model)



print("Warming up the model...")

with torch.no_grad():
    # Warm up the model for accurate benchmarking
    for _ in tqdm(range(100)):
        _ = scripted_model(dummy_input)
        torch.cuda.synchronize()

    times = []
    # Measuring performance (inference times)
    for _ in tqdm(range(100)):
        start = time.time()
        _ = scripted_model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

print("optimized inference times: ", times)
        
torch.jit.save(scripted_model, "output_path.ts")
