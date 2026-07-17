"""
test_model_probe.py — 模型结构探测单测

测纯函数逻辑（不加载真实模型）：属性名别名匹配、layer_prefix 候选命中、
router 嵌套路径探测、源码 grep。端到端探测见 examples/probe_model.py 实跑。
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from lib.model_probe import (
    _detect_attr, _detect_router, _find_layer_prefix, _grep_source_attrs,
    ProbeField, ModelProbeResult, save_probe_result,
)
from lib.model_probe import _ATTN_ALIASES, _MLP_ALIASES, _ROUTER_ALIASES


# ---------- 假模型结构 ----------

class _FakeGate(nn.Module):
    pass


class _FakeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _FakeGate()
        self.experts = nn.ModuleList([nn.Linear(4, 4)])


class _FakeDenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(4, 4)


class _FakeAttn(nn.Module):
    pass


class _FakeLayer(nn.Module):
    """模拟 BailingMoeV2：attn 叫 attention，mlp 可 dense 可 MoE"""
    def __init__(self, moe=False):
        super().__init__()
        self.attention = _FakeAttn()
        self.mlp = _FakeMoE() if moe else _FakeDenseMLP()


class _FakeLayers(nn.ModuleList):
    pass


class _FakeBackbone(nn.Module):
    def __init__(self, n=4, first_moe=1):
        super().__init__()
        self.layers = _FakeLayers([
            _FakeLayer(moe=(i >= first_moe)) for i in range(n)
        ])


class _FakeCausalLM(nn.Module):
    def __init__(self, n=4, first_moe=1):
        super().__init__()
        self.model = _FakeBackbone(n, first_moe)


# ---------- 测试 ----------

class TestDetectAttr:

    def test_finds_attention_alias(self):
        layer = _FakeLayer()
        assert _detect_attr(layer, _ATTN_ALIASES) == "attention"

    def test_finds_self_attn(self):
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(1, 1)
        assert _detect_attr(L(), _ATTN_ALIASES) == "self_attn"

    def test_returns_none_when_absent(self):
        class L(nn.Module):
            pass
        assert _detect_attr(L(), _ATTN_ALIASES) is None


class TestDetectRouter:

    def test_finds_gate_under_mlp(self):
        layer = _FakeLayer(moe=True)
        assert _detect_router(layer, "mlp") == "mlp.gate"

    def test_dense_layer_has_no_router(self):
        layer = _FakeLayer(moe=False)
        assert _detect_router(layer, "mlp") is None

    def test_finds_experts_nested_router(self):
        class Exp(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = nn.Linear(1, 1)
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = Exp()
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = M()
        assert _detect_router(L(), "mlp") == "mlp.experts.router"

    def test_no_mlp_attr(self):
        class L(nn.Module):
            pass
        assert _detect_router(L(), "mlp") is None


class TestFindLayerPrefix:

    def test_finds_model_layers(self):
        m = _FakeCausalLM()
        assert _find_layer_prefix(m) == "model.layers"

    def test_returns_none_for_empty(self):
        class M(nn.Module):
            pass
        assert _find_layer_prefix(M()) is None


class TestGrepSourceAttrs:

    def test_grep_finds_self_assignments(self):
        src = (
            "class L(nn.Module):\n"
            "    def __init__(self):\n"
            "        self.attention = Attn()\n"
            "        self.mlp = MoE()\n"
            "        self.gate = Gate()\n"
            "        self.foo = bar  # not an alias\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            found = _grep_source_attrs([path])
            assert "attention" in found["attn"]
            assert "mlp" in found["mlp"]
            assert "gate" in found["router"]
            assert "foo" not in found["attn"] and "foo" not in found["mlp"]
        finally:
            os.unlink(path)

    def test_empty_files(self):
        assert _grep_source_attrs([]) == {"attn": set(), "mlp": set(), "router": set()}


class TestSaveProbeResult:

    def test_save_writes_json_and_md(self):
        result = ModelProbeResult(
            model_path="/fake", arch="FakeForCausalLM", model_type="fake",
            is_moe=True, num_layers=4,
            layer_prefix=ProbeField("layer_prefix", "model.layers", ["reflection"], "medium"),
            attn_attr=ProbeField("attn_attr", "attention", ["reflection", "source"], "high"),
            mlp_attr=ProbeField("mlp_attr", "mlp", ["reflection", "source"], "high"),
            router_attr=ProbeField("router_attr", "mlp.gate", ["reflection", "source"], "high"),
            first_moe_layer=1, all_fields_ok=True,
        )
        with tempfile.TemporaryDirectory() as d:
            jp, mp = save_probe_result(result, d)
            assert os.path.isfile(jp)
            assert os.path.isfile(mp)
            import json
            with open(jp) as f:
                data = json.load(f)
            assert data["arch"] == "FakeForCausalLM"
            assert data["attn_attr"]["value"] == "attention"
            assert data["attn_attr"]["confidence"] == "high"
            with open(mp) as f:
                md = f.read()
            assert "attention" in md
            assert "全部高置信" in md
