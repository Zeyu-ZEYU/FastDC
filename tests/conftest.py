import pytest
import torch


class TinyTokenizer:
    """Minimal tokenizer for tests: characters map to token ids."""

    vocab_size = 96
    all_special_ids = [0, 1]
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [2 + (ord(c) % (self.vocab_size - 2)) for c in text]
        return {"input_ids": ids}


def tiny_model(model_type: str, dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = dict(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  pad_token_id=0, eos_token_id=1, bos_token_id=None)
    if model_type in ("qwen3",):
        kwargs["head_dim"] = 16
    config = AutoConfig.for_model(model_type, **kwargs)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="sdpa").eval()
    return model


@pytest.fixture(params=["llama", "qwen3", "phi3"])
def model_type(request):
    return request.param


@pytest.fixture
def tokenizer():
    return TinyTokenizer()
