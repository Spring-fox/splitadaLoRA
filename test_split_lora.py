import torch
from split_lora import SplitLoRALinear


def test_forward_shape():
    layer = SplitLoRALinear(6, 4, r=2, n_splits=2)
    x = torch.randn(3, 6)
    y = layer(x)
    assert y.shape == (3, 4)


def test_training_reduces_loss():
    torch.manual_seed(0)
    layer = SplitLoRALinear(5, 1, r=2, n_splits=5)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    x = torch.randn(20, 5)
    y = x.sum(dim=1, keepdim=True)

    with torch.no_grad():
        initial = ((layer(x) - y) ** 2).mean().item()

    for _ in range(100):
        pred = layer(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = ((layer(x) - y) ** 2).mean().item()

    assert final < initial
