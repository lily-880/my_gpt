"""vLLM Demo 拒绝自研权重。不导入 vllm。"""

import pytest

from my_gpt.vllm_guard import assert_official_hf_model


def test_accepts_gpt2():
    assert assert_official_hf_model("gpt2") == "gpt2"


def test_rejects_pt_file():
    with pytest.raises(ValueError, match="checkpoint"):
        assert_official_hf_model("checkpoints/shakespeare_perdoc/best.pt")


def test_rejects_shakespeare_name():
    with pytest.raises(ValueError, match="莎士比亚"):
        assert_official_hf_model("shakespeare-gpt2")


def test_accepts_official_snapshot_dir(tmp_path):
    d = tmp_path / "gpt2"
    d.mkdir()
    (d / "config.json").write_text('{"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}', encoding="utf-8")
    assert assert_official_hf_model(str(d)) == str(d)
