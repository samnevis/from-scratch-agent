import pytest
import torch

from model.config import KataLMConfig
from model.device import require_cuda
from model.transformer import KataLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="4060 CUDA required")
def test_gpu_one_step_no_oom():
    device = require_cuda("test")
    cfg = KataLMConfig()
    model = KataLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, cfg.vocab_size, (2, 256), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 256), device=device)
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert "4060" in torch.cuda.get_device_name(0)
