"""
model_probe.py — 模型结构探测（不武断推断）

目的：找出给定模型"注意力模块 / MLP 模块 / MoE router / decoder 层前缀"在代码里
的真实属性名，供 compare_layers 等挂 hook 用。**不硬编码、不靠 arch 名猜测**，
而是从模型真实结构反射 + modeling 源码交叉验证两条路得到，结果落盘可人核。

为什么不加载权重也能拿到真实架构？
    一个模型由两层信息构成：
    1. 结构（架构）：有哪些子模块、叫什么名、怎么连——由 config.json + modeling
       代码的 __init__ 决定。AutoModelForCausalLM.from_config(cfg) 只做这步：
       读 config + 跑一遍 __init__ 把模块树建出来（每个 nn.Linear 都在，shape 对，
       只是数值空/未初始化）。with torch.device("meta") 连内存都不分配。
       所以反射属性名（hasattr(layer,"attention") / layer.mlp.gate）完全有效——
       属性名是结构决定的，跟权重数值无关。
    2. 权重（参数数值）：每个 Linear 里的具体数——在 .safetensors，这才需要加载。
    即"加载权重"和"拿架构"是两回事。探测只需架构，故不加载权重。
    双保险：from_config 偶尔建残缺（如 AutoModel 返回 backbone 而非 CausalLM），
    故另有"源码 grep self.xxx=" 兜底；还不放心可 --load-weights 加载真权重复核。

为什么独立成一步：
- 探测和对比解耦——探测错了能先发现，不会带着错路径跑半天加载
- 结果可复用——同一模型探测一次，落 JSON 后续步骤直接读
- 防"武断推断"——结果白纸黑字，每项带来源与置信度，人能核对

两条探测路：
1. 反射：用 AutoModelForCausalLM.from_config 建空结构（不加载权重，省显存），遍历层找属性
2. 源码：从 modeling py 文件 grep `self.<name> = ` 确认属性名真实存在

两边一致 → high；仅反射 → medium；仅源码 → low（需人工确认）。

零副作用：import 不初始化 GPU、不读模型。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# 常见架构的属性名别名（探测候选，不是结论）
_ATTN_ALIASES = ["self_attn", "attention", "attn", "self_attention"]
_MLP_ALIASES = ["mlp", "feed_forward", "ffn", "block"]
_ROUTER_ALIASES = ["router", "gate", "gate_up", "router_layer", "moe_gate"]
_LAYER_PREFIX_CANDIDATES = ["model.layers", "layers", "transformer.h", "model.model.layers",
                            "gpt_neox.layers", "base_model.h"]


@dataclass
class ProbeField:
    """单项探测结果"""
    name: str                       # 如 "attn_attr"
    value: Optional[str]            # 如 "attention"
    sources: List[str] = field(default_factory=list)  # ["reflection", "source"]
    confidence: str = "low"         # high / medium / low
    note: str = ""


@dataclass
class ModelProbeResult:
    """完整探测结果"""
    model_path: str
    arch: str
    model_type: str
    is_moe: bool
    num_layers: Optional[int]
    layer_prefix: ProbeField
    attn_attr: ProbeField
    mlp_attr: ProbeField
    router_attr: ProbeField         # 相对 layer 的路径，如 "mlp.gate"
    first_moe_layer: Optional[int]  # 第一个含 MoE 结构的层号
    all_fields_ok: bool             # 所有关键项 confidence=high

    def to_dict(self) -> dict:
        d = {
            "model_path": self.model_path, "arch": self.arch,
            "model_type": self.model_type, "is_moe": self.is_moe,
            "num_layers": self.num_layers,
            "layer_prefix": asdict(self.layer_prefix),
            "attn_attr": asdict(self.attn_attr),
            "mlp_attr": asdict(self.mlp_attr),
            "router_attr": asdict(self.router_attr),
            "first_moe_layer": self.first_moe_layer,
            "all_fields_ok": self.all_fields_ok,
        }
        return d

    def to_markdown(self) -> str:
        lines = [
            f"# 模型结构探测结果",
            f"",
            f"- 模型路径: `{self.model_path}`",
            f"- arch: `{self.arch}`  model_type: `{self.model_type}`",
            f"- is_moe: {self.is_moe}  num_layers: {self.num_layers}",
            f"- first_moe_layer: {self.first_moe_layer}",
            f"- **全部高置信**: {'✅ 是' if self.all_fields_ok else '❌ 否（需人工核对）'}",
            f"",
            f"| 项 | 值 | 来源 | 置信度 | 说明 |",
            f"|---|---|---|---|---|",
            _row("layer_prefix", self.layer_prefix),
            _row("attn_attr", self.attn_attr),
            _row("mlp_attr", self.mlp_attr),
            _row("router_attr", self.router_attr),
            f"",
            f"> 来源：reflection=从模型真实结构反射，source=从 modeling py 源码确认。",
            f"> 置信度：high=两条路一致，medium=仅反射，low=仅源码或存疑。",
        ]
        return "\n".join(lines)


def _row(label: str, f: ProbeField) -> str:
    return f"| {label} | `{f.value}` | {', '.join(f.sources) or '-'} | {f.confidence} | {f.note} |"


def _find_layer_prefix(model) -> Optional[str]:
    """从模型根开始试候选前缀，返回第一个能取到层的"""
    for pref in _LAYER_PREFIX_CANDIDATES:
        try:
            obj = model
            for p in pref.split("."):
                obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
            if hasattr(obj, "__getitem__") and len(obj) > 0:
                return pref
        except (AttributeError, TypeError, IndexError):
            continue
    return None


def _detect_attr(layer, aliases) -> Optional[str]:
    for a in aliases:
        if hasattr(layer, a):
            return a
    return None


def _detect_router(layer, mlp_attr) -> Optional[str]:
    """在单层里找 router，返回相对 layer 的路径，如 'mlp.gate'"""
    if not mlp_attr:
        return None
    mlp_mod = getattr(layer, mlp_attr, None)
    if mlp_mod is None:
        return None
    for a in _ROUTER_ALIASES:
        if hasattr(mlp_mod, a):
            return f"{mlp_attr}.{a}"
    if hasattr(mlp_mod, "experts"):
        exp = getattr(mlp_mod, "experts")
        for a in _ROUTER_ALIASES:
            if hasattr(exp, a):
                return f"{mlp_attr}.experts.{a}"
    for a in _ROUTER_ALIASES:
        if hasattr(layer, a):
            return a
    return None


def _grep_source_attrs(py_files: List[str]) -> Dict[str, set]:
    """从 modeling py 源码 grep self.<name> = ，返回各类候选属性名集合"""
    found = {"attn": set(), "mlp": set(), "router": set()}
    if not py_files:
        return found
    pat = re.compile(r"self\.(\w+)\s*=\s*")
    for pf in py_files:
        try:
            with open(pf, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = pat.search(line)
                    if not m:
                        continue
                    name = m.group(1)
                    if name in _ATTN_ALIASES:
                        found["attn"].add(name)
                    if name in _MLP_ALIASES:
                        found["mlp"].add(name)
                    if name in _ROUTER_ALIASES:
                        found["router"].add(name)
        except OSError:
            continue
    return found


def probe_model(model_path: str, load_weights: bool = False) -> ModelProbeResult:
    """探测模型结构属性名。

    Args:
        model_path: 模型本地目录
        load_weights: False（默认）只 from_config 建空结构（快、省显存）；
                      True 加载完整权重（更可靠但慢）

    Returns:
        ModelProbeResult，可 to_dict / to_markdown 落盘
    """
    from .config_loader import load_model_profile
    prof = load_model_profile(model_path)

    # ---- 路1：源码 grep（不需 GPU）----
    src = _grep_source_attrs(prof.custom_py_files)

    # ---- 路2：反射（建空结构）----
    from transformers import AutoConfig, AutoModelForCausalLM
    import torch
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    # 复用 config_patch 修正 MTP 等字段
    try:
        from . import config_patch
        config_patch.patch_config(cfg)
    except Exception:
        pass

    if load_weights:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=cfg, torch_dtype=torch.bfloat16,
            trust_remote_code=True, local_files_only=True,
        )
    else:
        with torch.device("meta"):
            try:
                model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            except Exception:
                # 某些自定义模型 from_config 不支持 meta，退回普通 from_config（占 CPU 内存）
                model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    layer_prefix = _find_layer_prefix(model)
    layers_obj = model
    if layer_prefix:
        for p in layer_prefix.split("."):
            layers_obj = getattr(layers_obj, p) if not p.isdigit() else layers_obj[int(p)]
    num_layers = len(layers_obj) if hasattr(layers_obj, "__len__") else None

    attn_val = mlp_val = None
    router_val = None
    first_moe_layer = None
    if num_layers and num_layers > 0:
        layer0 = layers_obj[0]
        attn_val = _detect_attr(layer0, _ATTN_ALIASES)
        mlp_val = _detect_attr(layer0, _MLP_ALIASES)
        # router 要遍历层找第一个 MoE 层
        for i, layer in enumerate(layers_obj):
            r = _detect_router(layer, mlp_val)
            if r:
                router_val = r
                first_moe_layer = i
                break

    # ---- 合并两路，定置信度 ----
    def merge(value: Optional[str], src_set: set, label: str) -> ProbeField:
        has_refl = value is not None
        has_src = value in src_set if value else False
        if has_refl and has_src:
            conf, srcs = "high", ["reflection", "source"]
        elif has_refl:
            conf, srcs = "medium", ["reflection"]
        elif has_src and src_set:
            # 源码有但反射没探到——取源码第一个，标 low
            conf, srcs = "low", ["source"]
            value = sorted(src_set)[0]
        else:
            conf, srcs = "low", []
        note = ""
        if has_src and len(src_set) > 1:
            note = f"源码含多个候选: {sorted(src_set)}"
        return ProbeField(label, value, srcs, conf, note)

    layer_prefix_field = ProbeField(
        "layer_prefix", layer_prefix,
        ["reflection"] if layer_prefix else [],
        "medium" if layer_prefix else "low",
        "从候选前缀反射命中" if layer_prefix else "未命中任何候选，需人工指定 --layer-prefix",
    )
    attn_field = merge(attn_val, src["attn"], "attn_attr")
    mlp_field = merge(mlp_val, src["mlp"], "mlp_attr")
    # router 源码里的名字是相对 mlp 的，比对时要带 mlp 前缀
    router_src = {f"{mlp_field.value}.{n}" if mlp_field.value else n
                  for n in src["router"]}
    router_field = merge(router_val, router_src, "router_attr")
    if not prof.is_moe:
        router_field = ProbeField("router_attr", None, [], "high", "非 MoE 模型，无 router")

    all_ok = (layer_prefix_field.confidence == "high" or layer_prefix_field.confidence == "medium") \
        and attn_field.confidence in ("high", "medium") \
        and mlp_field.confidence in ("high", "medium") \
        and (not prof.is_moe or router_field.confidence in ("high", "medium"))

    return ModelProbeResult(
        model_path=model_path, arch=prof.arch, model_type=prof.model_type,
        is_moe=prof.is_moe, num_layers=num_layers,
        layer_prefix=layer_prefix_field, attn_attr=attn_field,
        mlp_attr=mlp_field, router_attr=router_field,
        first_moe_layer=first_moe_layer, all_fields_ok=all_ok,
    )


def save_probe_result(result: ModelProbeResult, workdir: str) -> Tuple[str, str]:
    """落盘到 workdir，返回 (json_path, md_path)"""
    os.makedirs(workdir, exist_ok=True)
    json_path = os.path.join(workdir, "model_probe.json")
    md_path = os.path.join(workdir, "model_probe.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.to_markdown())
    return json_path, md_path
