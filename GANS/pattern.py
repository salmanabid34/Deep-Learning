import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Define hook function
def forward_hook(module, input, output):
    print("Forward pass - input:", input, "output:", output)

# Create model and register the hook
model = SimpleModel()
hook = model.linear.register_forward_hook(forward_hook)

# Run a forward pass
x = torch.randn(1, 10)
output = model(x)

# Remove the hook when done
hook.remove()
