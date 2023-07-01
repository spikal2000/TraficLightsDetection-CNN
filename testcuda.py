import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computations.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computations.")

# Test CUDA computation
a = torch.tensor([1, 2, 3], device=device)
b = torch.tensor([4, 5, 6], device=device)
c = a + b
print("Result:", c)
