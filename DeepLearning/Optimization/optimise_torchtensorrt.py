import torch
import torch_tensorrt as trt
import torch.nn as nn

print("will output an optimized model for first float32 then float16 precision")

# Load the TorchScript model
ts_file_name = "path_to_torchscript_model.ts"
ts_model = torch.jit.load(ts_file_name)

# Example input tensor
example_input = torch.randn((1, 1, 240, 300)).cuda()

# Compile model with TensorRT for FP32 precision
trt_model_f32 = trt.compile(
    ts_model,
    inputs=[trt.Input(example_input.shape)],
    enabled_precisions={torch.float}
)

# Save the FP32 optimized model
torch.jit.save(trt_model_f32, "model_f32.ts")

print("saved trt model optimized with f32")
# Define the function to set the model to half precision
def set_to_half(m):
    """
    Recursively sets all submodules and parameters of the given module to half precision.
    This is necessary as not all parts of the generator architecture are compatible with the .half() method
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

# Apply the set_to_half function to the model
ts_model.apply(set_to_half)
ts_model.half()

# Convert example input to half precision
example_input = example_input.half()

# Compile model with TensorRT for FP16 precision
trt_model_f16 = trt.compile(
    ts_model,
    inputs=[trt.Input(example_input.shape)],
    enabled_precisions={torch.half}
)

# Save the FP16 optimized model
torch.jit.save(trt_model_f16, "model_f16.ts")

print("saved trt unet f16")
