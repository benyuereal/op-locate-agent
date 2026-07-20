# OpLocate Agent

> 把"vLLM 输出和 transformers 对不齐，定位到具体算子"——从天级压到小时级。
> 适用任意 HF/ModelScope 模型（MoE / dense）在 vLLM 上的精度异常定位。

## 快速开始

### 1. 拉项目 → 启动 Claude

```bash
git clone https://github.com/benyuereal/op-locate-agent.git
cd op-locate-agent

# 启动 Claude Code（带 op-locate skill）
claude -p "用 op-locate skill 定位 /path/to/model 的 vLLM 精度问题" \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,Skill"
```

### 2. 前提

- vLLM + transformers + torch 已装
- 模型已下载到本地，目录含 `config.json`
- **空闲卡用 `HIP_VISIBLE_DEVICES` 前置指定**（`rocm-smi` 确认）

### 3. 定位流程

```
① 逐层对比 → 找到发散层（只需一条命令，模型结构自动探测）
② 算子细化 → 定位到具体算子（在发散层内钻取）
③ 出报告   → 结论 + vLLM 源码调用链
```

**① 逐层对比**：比每层输入 hidden_states，看误差从哪层开始、如何累积。

```bash
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/compare_layers.py --model <模型路径>
```

> 自动探测模型结构（attn/mlp/router 属性名），自动采样全模型层，打印逐层 cos 衰减表 + 输出 token 对比。

**② 算子细化**：定位到发散层后，在该层内钻到具体算子。

```bash
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/compare_layers.py \
    --model <模型路径> --op <算子名> --layers <发散层号>
```

> `--op` 支持任意属性名（如 `attn`/`mlp`/`router`/`rmsnorm`），不限于三件套。
> MoE 推荐排除顺序：`router` → `mlp` → `attn`。
> `--env` 对照验证（如 `VLLM_ENABLE_MOE_FUSED_GATE=0`，默认不设）。

**③ 出报告**：定位结论 + vLLM 源码调用链落盘。

```bash
python3 examples/generate_report.py \
    --model <模型路径> \
    --compare-dir /tmp/compare_layers_xxx \   # 可多次传入聚合多轮细化
    --symptom "vLLM 输出异常描述"
```

产出 `reports/<model>_<date>/`：`report.md` + `verdict.json` + `evidence/`。

### 4. 完整示例（AntAngelMed）

```bash
# 确认空闲卡 → 逐层对比
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py \
    --model /models/AntAngelMed

# 算子细化（L1 的 mlp 算子）
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py \
    --model /models/AntAngelMed --op mlp --layers 1

# 出报告
python3 examples/generate_report.py \
    --model /models/AntAngelMed \
    --compare-dir /tmp/compare_layers_xxx \
    --symptom "vLLM 输出全 NULL"
```

## 目录结构

```
op-locate-agent/
├── examples/               # 入口脚本
│   ├── compare_layers.py   # 逐层/逐算子对比（核心，自动探测模型结构）
│   ├── generate_report.py  # 出报告
│   └── probe_model.py      # 模型结构探测（compare_layers 已内置自动探测）
├── lib/                    # 工具库
│   ├── config_loader.py    # 模型配置解析
│   ├── model_probe.py      # 模型结构反射探测
│   ├── path_resolver.py    # vLLM 代码路径定位
│   ├── hook_manager.py     # hook 注册与管理
│   ├── tensor_compare.py   # 张量对比（cos/max_abs）
│   └── platform_probe.py   # 平台信息探测
├── knowledge/              # 知识库
│   ├── precision_known_issues.md  # 已知精度问题库
│   ├── vllm_forward_paths.md      # vLLM 前向路径速查
│   └── arch_index.md             # 架构→代码路径映射
├── skills/op-locate/       # Claude Code skill
├── reports/                # 定位报告输出
└── tests/                  # 单测
```

## 设计要点

- **自动探测**：`compare_layers.py` 运行时自动反射模型结构，无需先跑 `probe_model.py`。已有缓存（`--probe-dir`）则直接复用。
- **运行时探针为准**：根因以 hook 抓取为准，不以静态分析为准。
- **不预设结论**：默认不设修复环境变量，需对照时显式开 `--env`。
- **泛化设计**：`--op` 支持任意属性名，dense/MoE 共用同一套探针体系。
- **不自动回写知识库**：人工 review 是质量门。

## 测试

```bash
python3 -m pytest tests/ -q          # 单测
python3 lib/config_loader.py <模型路径>   # lib 自检
```
