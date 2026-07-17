"""
probe_model.py — 探测模型结构属性名，落盘到临时工作目录

不武断推断 attn/mlp/router/layer_prefix 的属性名：从模型真实结构反射 +
modeling 源码交叉验证两条路得到，结果落 JSON + MD 可人核，供 compare_layers
等后续步骤读取。

== 用法 ==
    # 探测并落盘到默认临时目录
    python3 examples/probe_model.py --model /models/AntAngelMed

    # 指定工作目录（推荐，便于复用）
    python3 examples/probe_model.py --model /models/AntAngelMed --workdir /tmp/probe_antangelmed

    # 加载完整权重探测（更可靠但慢，默认只 from_config 建空结构）
    python3 examples/probe_model.py --model /models/AntAngelMed --load-weights

== 输出 ==
    <workdir>/model_probe.json   # 结构化，供程序读
    <workdir>/model_probe.md     # 人可读表格，每项带来源与置信度

== 说明 ==
    - 默认不加载权重（AutoModel.from_config + meta device），省显存省时间
    - 探测结果供 compare_layers 用 --probe-dir 读取，避免每次重探
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser(description="探测模型结构属性名并落盘")
    ap.add_argument("--model", required=True, help="模型本地路径")
    ap.add_argument("--workdir", default=None,
                    help="输出工作目录；默认 /tmp/probe_<model_basename>_<pid>")
    ap.add_argument("--load-weights", action="store_true",
                    help="加载完整权重探测（默认只 from_config 空结构）")
    args = ap.parse_args()

    if args.workdir is None:
        base = os.path.basename(os.path.normpath(args.model))
        args.workdir = os.path.join(tempfile.gettempdir(),
                                   f"probe_{base}_{os.getpid()}")

    from lib import probe_model, save_probe_result
    print(f"[probe] 探测 {args.model} ...")
    print(f"[probe] 模式: {'加载权重' if args.load_weights else 'from_config 空结构'}")
    result = probe_model(args.model, load_weights=args.load_weights)

    json_path, md_path = save_probe_result(result, args.workdir)
    print(f"\n[probe] 落盘:")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")
    print(f"\n[probe] 摘要:")
    print(f"  arch          : {result.arch}  model_type: {result.model_type}")
    print(f"  is_moe        : {result.is_moe}  num_layers: {result.num_layers}")
    print(f"  first_moe_layer: {result.first_moe_layer}")
    print(f"  layer_prefix  : {result.layer_prefix.value}  [{result.layer_prefix.confidence}]")
    print(f"  attn_attr     : {result.attn_attr.value}  [{result.attn_attr.confidence}]"
          f"  来源:{result.attn_attr.sources}")
    print(f"  mlp_attr      : {result.mlp_attr.value}  [{result.mlp_attr.confidence}]"
          f"  来源:{result.mlp_attr.sources}")
    print(f"  router_attr   : {result.router_attr.value}  [{result.router_attr.confidence}]"
          f"  来源:{result.router_attr.sources}")
    print(f"\n[probe] 全部高置信: {'✅ 是' if result.all_fields_ok else '❌ 否（请看 .md 核对）'}")
    if not result.all_fields_ok:
        print("[probe] 提示: 某项置信度低，打开 model_probe.md 看 note，必要时人工指定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
