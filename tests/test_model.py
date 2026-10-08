import torch

from windllm.model import ModelConfig, RMSNorm, WindLLM, apply_rope


def small_model():
    torch.manual_seed(17)
    return WindLLM(
        ModelConfig(
            vocab_size=64,
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            hidden_dim=64,
            max_seq_len=16,
        )
    )


def test_future_tokens_cannot_change_past_logits():
    model = small_model().eval()
    ids = torch.randint(0, 64, (2, 12))
    changed = ids.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 64
    torch.testing.assert_close(model(ids)[:, :7], model(changed)[:, :7], atol=1e-6, rtol=1e-5)


def test_rope_preserves_vector_norm():
    x = torch.randn(2, 4, 10, 8)
    theta = torch.randn(10, 4)
    rotated = apply_rope(x, theta.cos(), theta.sin())
    torch.testing.assert_close(x.square().sum(-1), rotated.square().sum(-1))


def test_rmsnorm_and_weight_tying():
    x = torch.randn(2, 6, 32)
    result = RMSNorm(32, 1e-6)(x)
    torch.testing.assert_close(result.square().mean(-1), torch.ones(2, 6), atol=1e-5, rtol=1e-5)
    model = small_model()
    assert model.embedding.weight is model.lm_head.weight


def test_model_can_overfit_a_sequence():
    torch.set_num_threads(2)
    model = small_model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    targets = torch.tensor([[2, 3, 4, 5, 6]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(35):
        loss = torch.nn.functional.cross_entropy(model(ids).flatten(0, 1), targets.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
    assert model.generate(ids[:, :1], max_new_tokens=5, temperature=0).tolist() == [
        [1, 2, 3, 4, 5, 6]
    ]
