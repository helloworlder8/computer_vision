import torch
import torch.nn as nn

class CLLA(nn.Module):
    def __init__(self, range, c): # 3 8
        super().__init__()
        self.c_ = c
        self.q = nn.Linear(self.c_, self.c_)
        self.k = nn.Linear(self.c_, self.c_)
        self.v = nn.Linear(self.c_, self.c_)
        self.range = range
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x1, x2):
        b1, c1, w1, h1 = x1.shape
        b2, c2, w2, h2 = x2.shape
        assert b1 == b2 and c1 == c2

        x2_ = x2.permute(0, 2, 3, 1).contiguous().unsqueeze(3)
        pad = int(self.range / 2 - 1)
        padding = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
        x1 = padding(x1)

        local = []
        for i in range(int(self.range)):
            for j in range(int(self.range)):
                tem = x1
                tem = tem[..., i::2, j::2][..., :w2, :h2].contiguous().unsqueeze(2)
                local.append(tem)
        local = torch.cat(local, 2)

        x1 = local.permute(0, 3, 4, 2, 1)

        q = self.q(x2_)
        k, v = self.k(x1), self.v(x1)

        dots = torch.sum(q * k / self.range, 4)
        irr = torch.mean(dots, 3).unsqueeze(3) * 2 - dots
        att = self.attend(irr)

        out = v * att.unsqueeze(4)
        out = torch.sum(out, 3)
        out = out.squeeze(3).permute(0, 3, 1, 2).contiguous()
        return (out + x2) / 2

def test_clla():
    # Input dimensions and parameters
    b, c, w, h = 2, 8, 32, 32  # Batch size, channels, width, height
    attention_range = 3  # Define a range for local attention

    # Initialize the CLLA module
    clla = CLLA(range=attention_range, c=c)

    # Create dummy inputs
    x1 = torch.randn(b, c, w, h, requires_grad=True)
    x2 = torch.randn(b, c, w, h, requires_grad=True)

    # Forward pass
    output = clla(x1, x2)

    # Check output shape
    assert output.shape == x2.shape, "Output shape does not match input shape"

    # Check if gradients can be computed (backward pass)
    output.mean().backward()
    assert x1.grad is not None and x2.grad is not None, "Gradients were not computed"

    print("CLLA test passed successfully!")

# Run the test
test_clla()
