from model.config import KataLMConfig
from model.transformer import KataLM


def test_forward_and_loss_cpu_shapes():
    # Shape/loss unit test only. Training must use CUDA (see test_gpu_smoke).
    cfg = KataLMConfig(vocab_size=128, block_size=32, n_layer=2, n_head=4, d_model=64, d_ff=128)
    import torch

    model = KataLM(cfg)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 128)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
