# Hailo-8 量化部署参考文档（DFC v3.33.1）

> 用途：Demo 3（RTFormer → Hailo 部署）的**权威参照**。写代码、调 `.alls`、跑量化前先查这里，**防止凭记忆瞎改 API / 参数 / 版本**。
> 适用硬件：Raspberry Pi 5 + Hailo-8 AI HAT+（26 TOPS），发送端 + 接收端各一块。
> 来源：Hailo Dataflow Compiler v3.33.1 官方文档（Developer Zone），整理日期 2026-06-08。
> 关联：上层项目上下文见 `CLAUDE.md`；本文件只管 Hailo 部署链路。

---

## 0. 一句话定位

Demo 3 的目标 = 证明 **RTFormer 的算子能过 Hailo DFC + 量化掉点可接受**。
这两件事 **100% 在 host（x86 + GPU）上完成，不需要板子**。板子（HailoRT）只在 Phase 3 测真实延迟/功耗时才需要，且那是**可选**章节。

```
本文件覆盖 = DFC 这半边（host 编译）
不覆盖     = HailoRT（板上运行，Phase 3）
```

---

## 1. 版本分线（最容易踩的坑，先记死）

| 硬件 | DFC 分支 | Model Zoo 分支 | HailoRT |
|---|---|---|---|
| **Hailo-8 / 8L（我们）** | **3.x（当前 3.33.x）** | **v2.x** | 4.x 系 |
| Hailo-10 / Hailo-15 | 5.x | master | 5.x |

- **不要用 DFC 5.2 / 5.3** —— 那是给 Hailo-10/15（如 Pi AI HAT+ 2 的 Hailo-10H）的。我们的 Hailo-8 必须用 **3.33.x**。
- **HEF ↔ HailoRT 版本要配套**：DFC 编 HEF 用的版本和板上 HailoRT 版本必须匹配（曾见 DFC 5.3 编的 HEF 在 HailoRT 4.20 上加载失败）。上板时两端锁同一 Software Suite 大版本。
- DFC 3.31.0 起 HEF 错误校验默认从 CRC32 改 XXHASH3_64 → **老 HailoRT（<4.21.0）跑不了新 DFC 编的 HEF**。上板时 HailoRT 锁 **≥4.21.0**。
- Hailo-8 的 **Ethernet 通信已弃用，只走 PCIe**（HailoRT 4.18.0 是最后支持 H8 以太网的版本）。Pi5 上 HAT 本就是 PCIe，不受影响；但这意味着 HEF 的 `should_use_sequencer` / `params_load_time_compression`（默认 True，移除 Ethernet 支持）对我们无副作用。

---

## 2. 环境（Docker 或宿主机 venv）

### 2.1 环境隔离与选型
Docker suite 是一条受支持的路径，但不是强制要求。宿主机也可以用独立 Python venv 跑部署工具；前提是该机已经从授权的 Hailo Software Suite 安装包获得 Hailo-8 对应的 DFC 3.33.x SDK 及其运行时依赖。不要从公共 PyPI 猜装或混用 DFC 5.x。

仓库的 `scripts/setup_venv.sh` 会创建 `.venv`、安装通用 ONNX/runtime 工具依赖并检查 SDK 导入。可以通过 `HAILO_WHEEL_DIR` 安装 suite 提供的 wheel，或通过 `HAILO_SITE_PACKAGES` 只读复用同 Python ABI 的既有 SDK 安装；完整用法见仓库根目录 README。训练环境（新 PyTorch）仍应与 Hailo 环境分开。

Docker 的优势仍是 DFC + HailoRT + Model Zoo 版本预先配套；若宿主机缺少授权 SDK 或其系统依赖，应使用 Docker/Suite 安装包补齐，而不是尝试用通用 pip 包替代。

### 2.2 硬件/软件要求
- Ubuntu 20.04/22.04（裸装支持 22.04/24.04；Windows 走 WSL2）。
- 16GB+ RAM（荐 32GB）。CPU 需支持 **AVX**。
- **GPU 加速量化（必配）**：Nvidia Pascal/Turing/Ampere/Ada、driver 525+、CUDA 12.5.1、cuDNN 9.10、`nvidia-docker2`。量化 emulator 评估靠它，没 GPU 极慢。
- 裸装额外 apt：`python3.X-dev python3-tk graphviz libgraphviz-dev pkg-config`；Python 3.10/3.11/3.12。

### 2.3 Docker 启动
```bash
# 从 Developer Zone 下 suite 包；PCIe 驱动仅上板需要，host 编译可跳过
unzip hailo_ai_sw_suite_<version>.zip
./hailo_ai_sw_suite_docker_run.sh            # 进容器
# 常用 flag：--resume（回到已有容器） --override（删了重建）
```
- 共享文件夹：容器内 `/local/shared_with_docker/` ↔ 容器外 `./shared_with_docker`。ONNX、calib 图、HEF 都从这里进出。
- 容器不持久，**只有共享文件夹持久**。升级 suite 前先把私有文件拷到共享文件夹。
- 验证：`pip list | grep hailo`，`hailo -h`。

### 2.4 裸装升级注意
DFC 与 HailoRT 在同 venv 时，**先升 HailoRT 再升 DFC**。旧版（≤3.14）包名变过，需先 `pip uninstall -y hailo_sdk_common hailo_sdk_client hailo_sdk_server hailo_model_optimization`。

---

## 3. 总体流程与状态机（ClientRunner）

`ClientRunner` 是贯穿全程的有状态对象，每阶段可 `save_har`/`load_har` 存档续跑。

```
Uninitialized
   │ translate_onnx_model()                     ← 阶段1 Translation（parse）
   ▼
Hailo Model ──── optimize_full_precision() ──→ FP optimized model
   │ optimize(calib)                            ← 阶段2 Optimization（含量化）   │
   ▼                                                                            │
Quantized Model ←────────────────────────────────────────────────────────────┘
   │ compile()                                  ← 阶段3 Compilation
   ▼
Compiled Model  → 导出 HEF
```

每个状态解锁的 emulator / 工具：

| 状态 | 能跑的 emulator | 验什么 |
|---|---|---|
| Hailo Model | `SDK_NATIVE` + `hailo profiler` | 算子翻译对不对 |
| FP optimized model | `SDK_FP_OPTIMIZED` | 模型改造对不对 |
| Quantized Model | `SDK_QUANTIZED` + `analyze_noise()` | 量化掉多少点 |
| Compiled Model | — | 导 HEF |

CLI 等价：`hailo parser/optimize/compiler/profiler/analyze-noise/visualizer/har`。

---

## 4. 阶段一：Parse（ONNX → HAR）

```python
from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8")            # 标准 Hailo-8；Mini PCIe/8R 用 "hailo8r"
hn, npz = runner.translate_onnx_model(
    onnx_path, onnx_model_name,
    start_node_names=[...],     # 可选；先不带，让它自动推断
    end_node_names=[...],       # 可选；切分点用这个
    net_input_shapes={...},     # 仅原模型有动态轴时用
)
runner.save_har(f"{name}_hailo_model.har")
```

要点：
- **导 ONNX 要求静态 shape（opset 11-17，建议 11-14）+ onnxsim**。正常不需要 `net_input_shapes`；若动态轴没固定住、parse 报错才用它。
- **切分点 = `end_node_names`**：我们的网络在 stage3 后对半切（发送端纯卷积，attention 挪 decoder）。发送端 encoder 用 `end_node_names` 指定其输出节点，**单独 parse 成独立 HAR**；接收端 decoder 另成一段。两段各自 parse → optimize → compile，对应两块板子。
- 调试利器：`hailo visualizer {har} --no-browser` 导 SVG 看 parse 后的图，定位坏算子。
- `from_tf('原始层名')`：因 Hailo 的 HN 层名 ≠ ONNX 原名，写 `.alls` 指定层时可用原名定位（支持 ONNX）。

---

## 5. 阶段二：Optimize（量化）+ 三段式验证

### 5.1 三段式 emulator 验证法（核心方法论）
把"掉点"拆成三段定位，每段只剩一个变量：

```python
# 1) NATIVE：验 parse（应与 PyTorch/ONNX 完全一致）
with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
    native_res = runner.infer(ctx, calib_normalized)

# 2) FP_OPTIMIZED：验模型改造（加了归一化/resize 才需要；与 native 略有差异）
# 3) QUANTIZED：验量化掉点（核心数字）
with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
    q_res = runner.infer(ctx, image_dataset)
```
逻辑：NATIVE 不对 → parse 坏（算子/切分点）；FP 不对 → 改造坏；QUANTIZED 才掉 → 纯量化问题，去调 `.alls`。
> quantized emulator **不是 bit-exact**，但快且近似得好。

### 5.2 量化 API + calibration
```python
runner = ClientRunner(har="..._hailo_model.har")
alls = "...量化命令...\n"
runner.load_model_script(alls)
runner.optimize(calib_dataset)               # 多输入用 dict: {"net/input_layer1": calib}
runner.save_har("..._quantized_model.har")
```
- **calib ≥1024 张**（level 2/4 要 1024，level 3 要 256）。代表你的真实分布（Cityscapes 抽 Cityscapes，别用 ImageNet）。
- **NHWC**（通道在最后），与 PyTorch 的 NCHW 不同，准备 calib array 要转轴。
- 预处理须与训练/部署一致。
- `runner.get_hn_dict()["layers"]` 查 Hailo 给的输入层名。

### 5.3 optimization_level 的真实耗时（影响 β 扫描策略）
ResNet-50 + compression_level=4 @ A4000：
```
level 0: 59s   level 1: 206s   level 2: 256s(默认)
level 3: 2828s(47min)   level 4: 11002s(~3h)
```
> 我们的网络更大、输入 512×1024，单模型 level 4 可能数小时。
> **策略：迭代/调试用 level 2；只对出论文的几个 β 跑 level 4。** 别全程 level 4。

### 5.4 官方 11 步「Debugging Accuracy」（掉点不达标时按序来）
1. calib ≥1024 + 有 GPU
2. `SDK_FP_OPTIMIZED` 验预处理（常见坑：归一化、calib dtype）
3. 确保训练用了 BatchNorm（压窄激活范围，对量化关键）
4. `analyze_noise` 定位坏层（看 SNR）
5. 用了 compression 先调低
6. 提 `optimization_level=4`（显存爆配 `batch_size=2`）
7. 输出层 16-bit：`quantization_param(output_layer1, precision_mode=a16_w16)`
8. 敏感层 16-bit（注意：激活非 linear/relu/leaky 时精度受激活精度限制 → 指向 attention）
9. activation clipping（percentile 裁离群）
10. Finetune（更多数据、更长训练）
11. Finetune 换 loss 类型

---

## 6. 量化调优旋钮（`.alls` 命令参考）

> 全塞同一个 `.alls` 字符串，靠 `optimize()` 触发。命令分三类（见 §6.4）。

### 6.1 精度（逐层 bit 宽）
```
quantization_param(<layer>, precision_mode=a16_w16)   # 提精度
quantization_param(<layer>, precision_mode=a8_w4)     # 省内存
```
- 三档：`a8_w8`(默认) / `a8_w4` / `a16_w16`。a16_w16 支持：输出节点、特定支持层、或整网（**仅 Hailo-8 家族**）。
- ⚠️ **softmax chain 限制**：16-bit 支持层里 Reduce Max/Sum、Resize、EW Add 标 `*`，注脚说**这些若属于 softmax chain 则不支持 16-bit**。即 attention 的 softmax（多列时被拆成 reduce/EW 链）不能简单靠 16-bit 救 → 用 §6.3 的 transformer 命令。

### 6.2 力度与压缩
```
model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=2)
```
- `optimization_level` 0-4：越高越准越慢（背后是 AdaRound/bias correction/equalization 等 post-quant 算法）。**对部署速度无影响，只费 host 时间 → 该开高就开高（除迭代期）。**
- `compression_level` 0-5：4-bit 权重占比，**省权重内存+提 FPS，对功耗影响可忽略**；需配高 optimization_level 找回精度。
- 小模型 compression 会被自动 revert，强制用：`model_optimization_config(compression_params, auto_4bit_weights_ratio=0.8)`。
- ⚠️ **4-bit 压的是权重内存，不是 latent 的 bpp** —— **别画进 bpp-mIoU 曲线**，最多作"部署端权重可再压"的附注。

### 6.3 ★ Transformer 专用命令（RTFormer attention 的主武器）
attention 量化不要只想着 16-bit，开这套组合拳：
```
pre_quantization_optimization(layer_norm_decomposition, mode=nn_core)   # LayerNorm 分解
pre_quantization_optimization(smart_softmax_stats, policy=enabled)      # softmax 高效统计（默认已开）
pre_quantization_optimization(matmul_equalization, layers=<mm>, policy=enabled)   # Q/K/V 范围均衡
pre_quantization_optimization(matmul_correction, layers=<mm>, correction_type=zp_comp_weights)  # 零点矫正
pre_quantization_optimization(matmul_decomposition, layers=[<mm>], policy=enabled, precision_mode=a16_w8)  # 16bit matmul 用 8bit 实现
pre_quantization_optimization(quarot, policy=enabled)                   # QuaRot 旋转法抹离群
pre_quantization_optimization(defuse, layers=<mm>, num_splits=2, defuse_type=MHA)  # MHA 太大时拆分
```

### 6.4 clipping（裁离群）
```
pre_quantization_optimization(weights_clipping, layers={conv*}, mode=mmse)
pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])
```
mode：`manual`(手给范围) / `percentile`(百分位) / `mmse`(最小均方自动) / `mmse_if4b`(4-bit 时)。

### 6.5 post-quantization（量化后修复，通常自动）
```
post_quantization_optimization(adaround, policy=enabled)           # 学最优舍入方向；level>=3 自动开
post_quantization_optimization(bias_correction, ...)              # 矫正输出偏移
post_quantization_optimization(finetune, policy=enabled, def_loss_type=cosine, dataset_size=4096, epochs=8)
```
- 这些由 `optimization_level` 自动调度，**正常不用手动**。只在 AdaRound 显存崩时手动调小其 batch。
- **Finetune = 蒸馏式量化微调**（loss 可 l2/l2rel/cosine/ce + `loss_layer_names` 指定对齐层）—— **与本项目训练主线（feature 蒸馏）同源**，可作论文呼应点 + 终极救场。重，最后手段。

### 6.6 命令分类（理解 optimize 干了两件事）
`optimize()` = 应用 model script 改造 **+** 跑量化。
- **改造类**（Full-precision 阶段）：`normalization` / `resize` / `input_conversion`(YUV→RGB) / `nms_postprocess`。→ Demo 3 **可先不加**。
- **量化类**（Numerical 阶段）：`model_optimization_flavor` / `quantization_param` / `pre_/post_quantization_optimization`。→ **Demo 3 主战场**。
- **编译类**（Compilation 阶段）：见 §8。

---

## 7. RTFormer 算子可行性裁决（Hailo-8 支持情况）

> 来源：DFC v3.33.1 Supported Layers。结论：**整体乐观**，最担心的 attention 因 MHA 一等支持而风险大降。

| RTFormer 部件 | 映射到 Hailo 算子 | 支持状态 | 风险/备注 |
|---|---|---|---|
| stem + stage1~3（纯卷积） | Conv/DWConv/GroupConv/BN/Add/Pool/Concat | ✅ 全支持 | 零风险；RepVGG/BN 折叠可落地 |
| cross-resolution attention（softmax，数据相关 K/V） | **Multi Head Attention**（一等支持，官方给 PyTorch 示例，K/Q/V 可数据驱动） | ✅ 支持 | **最大惊喜**，原担心的 softmax 由 MHA 内部处理 |
| GFA external memory matmul（冻结记忆） | 重写为 **1×1 Conv / Dense**（weight matmul） | ✅ 可降解 | data×data matmul **仅 MHA 内支持**，自由 matmul 不支持 |
| **GFA grouped double-normalization** | Reduce Mean/Sum/SumSquare + Elementwise Mul/Div 拼 | ⚠️ **唯一未明确祝福** | **Demo 3 #1 风险点，必须亲手 parse 验** |
| LayerNorm | Reduce Mean + Reduce SumSquare + EW Div（或 `layer_norm_decomposition`） | ✅ 可拼 | 对应 3.30 "improved LayerNorm quantization" |
| decoder 上采样 / head bilinear | Resize（NN + Bilinear） | ✅ 支持 | 注意 align_corners/half_pixel 组合限制 |
| 深度/法线头 depthwise 3×3 | Depthwise Conv | ✅ 支持 | |
| 法线头 L2 normalize | Reduce SumSquare + EW Div | ✅ 可拼 | |
| 分割头 argmax | Argmax | ✅ 支持 | **最多 64 类**（19/40/59 都 OK），须为最后一层 |
| 双分支融合 | Concat | ✅ 支持 | **仅 features 维、≤4 输入、4D** |

**剩余必验真风险（Demo 3 集火点）**：
1. GFA double-normalization 能否用 Reduce+EW 拼出并过 parse —— **优先单独拎一个 GFA block parse**。
2. GFA 有无 data×data matmul（若有只能走 MHA 或改写）。
3. Transpose 代价：Height↔Width transpose 仅当量化张量 <1.5MB 且"非最优"；Width↔Column 无此限。
4. Reshape 仅支持固定模式（照 MHA 示例的 `flatten(2)+permute` 最稳）。

**叙事微调**：原以为 RTFormer 优势在"线性/external 注意力可精确折叠"；现发现 Hailo 对**标准 softmax MHA 也友好**，即使 GFA 折叠路线受阻，退回"标准 MHA + 蒸馏桥接"也有官方兜底。RTFormer 上板风险可控。

### 7.1 ★★ DFC 3.33 的 `Floor` / `Cast` 是 **no-op**：取整必须放 host

> 实测（发送端 bottleneck 踩坑，根因已定位）。**这一条对任何"图内取整"的模型都成立**（超分编解码、
> 量化感知、向量量化等），先看这里再写图。

**现象**：`Floor` 节点在 DFC 里能 parse、能 optimize、能 compile，但**不产生任何取整效果**；
`Cast` 到 int32 同样无效。

**证据（差分探针，`scripts/probe_control_ops.py`）**——同一个输入张量过 5 个单输出图，
结果与 ORT 对照：

| 算子 | Hailo 结果 | 判定 |
|---|---|---|
| `Mul(x, 1)` | 与 ORT 一致 | ✅ 探针本身是可靠的 |
| `Abs(x)` | 与 ORT 一致 | ✅ |
| `Mul(x, 2)` | 与 ORT 一致 | ✅ |
| `Floor(x)` | **原样返回 x** | ❌ no-op |
| `Cast(x→int32)` | **原样返回 x** | ❌ no-op |

探针自检通过（`Mul`/`Abs` 与 ORT 完全吻合），所以"取整失效"是**编译器行为**，不是探针 bug。

**更硬的证据：编译器根本没保留这个算子**（`scripts/check_floor_layer.py`）。
对含 `Floor` 的 bottleneck 图，optimized HAR 里 **23 层，一个 rounding 层都没有**，
只有 GDN 用的 2 个 `ew_mult` + 2 个 `feature_multiplier`：

```
layer types:  11 conv | 4 output_layer | 2 input_layer
              2 feature_multiplier | 2 ew_mult | 1 concat | 1 resize
rounding-related layers: (none)
```

也就是说 `Floor` 在 **parse 阶段就被吃掉了**，根本没有进 HEF —— 所以**真板子也不会取整**，
不是"emulator 骗人、板子上是好的"。

**代价（实测，64 张真实验证图，CompressAI 真实 rANS 码流）**：

| 路线 | bpp | z bits/img | y bits/img |
|---|---|---|---|
| PyTorch FP32 | 0.1796 | 1 145 | 45 944 |
| Hailo int8（图内 Floor） | **0.4219**（+135%） | **59 988**（52×） | 50 604 |
| Hailo int8（取整放 host） | **0.1808**（+0.63%） | 1 118 | 46 265 |

超-latent 的 `z_hat` 没取整 → 传输符号不是整数 → factorized prior 看到离群点 →
z 流每个符号多花几十倍比特。**y 流几乎不变**，因为 y 本来就由 means/scales 量化，
不依赖取整。所以"bpp 涨 2.3 倍"这个现象要能从 z/y 分项一眼定位。

**正确做法：把取整拆到 host，图里不留 `Floor`**。

```
HEF #1 "enc" : (x_h, x_l) -> (y, z_pre)     # align_h + g_a + h_a（重活全在片上）
HOST         : z_hat = floor(z_pre + 0.5) + median    # 标量，numpy 一行
HEF #2 "hs"  : z_hat -> (means, scales)                # h_s
```

host 只多传一个 `[1,128,8,8]`（32 KiB）的小张量，换来 bpp 从 0.42 回到 0.18。
这与"取整/熵编码放主机侧"的部署约定完全一致（熵编码本来就不在 Hailo 上做）。

**注意 median 语义**：CompressAI 传输的是 `round(z - median)`，`z_hat = 符号 + median`。
所以校验"符号是不是整数"要查 **`z_hat - median`**，不是 `z_hat` 本身——
`z_hat` 带 median 偏移，本来就不是整数（`max|x - round(x)| ≈ 0.49` 是正常的）。

**替代方案已排除**：`Cast` 到 int32 同样是 no-op；`trunc(d + 1024.5) - 1024`
这种"用 Cast 凑出 floor"的写法也因 Cast 无效而不成立（`probe_cast_rounding.py` 实测）。
**DFC 3.33 没有可用的片上取整原语，host 取整是唯一解。**

### 7.2 ★★ 发送端最终定为「全 Hailo 方案」（结论，已坐实）

> 决策记录（2026-10-25）。发送端 = **backbone + bottleneck_enc + bottleneck_hs 三个 HEF 全在 Hailo 上**，
> 取整 `floor(z_pre+0.5)+median` 放 host 侧标量算子。**不再走 hybrid（Hailo backbone + ORT fp16
> bottleneck）路线。** 依据：全量校准 + **全量 2000 张验证集**的实测数字（下表）。
> （2026-10-25 修正：此前此处用的是 500 张分层子集，mIoU 口径偏低、不可与训练日志的 0.3499 混用，
> 详见本节末尾的「口径归因」。）

**最终链路**（三个 HEF + 一个 host 标量算子）：

```
HEF #1 "backbone" : image(0..255)      -> (x_h, x_l)      # on-chip normalization
HEF #2 "enc"      : (x_h, x_l)         -> (y, z_pre)      # align_h + g_a + h_a
HOST              : z_hat = floor(z_pre + 0.5) + median   # 标量，整数化保证
HEF #3 "hs"       : z_hat              -> (means, scales) # h_s
```

**全量校准（1792 张，`calibset_size=1792` 显式设置，DFC 日志确认「Using dataset with 1792 entries」）**
—— 三个 HEF 全部 compile 成功，**均为单 context**：

| HEF | context | Control | Compute | Memory | 大小 |
|---|---|---|---|---|---|
| `sender_backbone.hef` | 1 | 71.1% | 75.4% | 54.3% | 8.4 MiB |
| `sender_bottleneck_enc.hef` | 1 | 14.1% | 12.3% | 10.8% | 1.8 MiB |
| `sender_bottleneck_hs.hef` | 1 | 48.4% | 33.6% | 22.4% | 1.7 MiB |

> backbone 的 cluster_0 compute 已到 100%（上限 75% 的 override 下映射后回落），是全链路最紧的一环；
> enc/hs 余量充足。**单 context = 不需要 `--runtime-data` 就能在 host 上拿性能数字（见 §9）。**

**全量验证集（2000 张，`val_manifest.json` 的 `full`）实测 —— 这才是与训练日志同口径的数字**：

> **口径说明（2026-10-25 修正）**：训练日志里 student 的 `mIoU=0.3499` 是
> `tools/train_student.py` → `evaluate_student` 在 **ADE20K 全量 2000 张 validation** 上算的
> （`cfg.data.val` → `ade20k_local`，resize 512×512，数据集累计混淆矩阵，150 类，ignore=255，
> bpp = **估计** bpp）。用**原代码路径**复现得到 `bpp=0.1738 mIoU=0.3499 pixel_acc=0.7627`，
> 与训练日志逐位一致。**它和部署评估的 0.2865 不是同一口径**（差的是评估集，不是模型，也不是 bug）。

| 指标（全量 2000 张） | PyTorch FP32 | Hailo int8（全 Hailo） | 差异 |
|---|---|---|---|
| bpp（真实 rANS） | 0.17467 | 0.17583 | **+0.66%** |
| z bits/img | 1 180 | 1 156 | −2% |
| y bits/img | 44 610 | 44 937 | +0.7% |
| cos `x_h` | 1.0 | min 0.99588 / mean 0.99778 | — |
| cos `x_l` | 1.0 | min 0.99404 / mean 0.99786 | — |
| cos `y` | 1.0 | min 0.99519 / mean 0.99799 | — |
| cos `z_hat` | 1.0 | min 0.98371 / mean 0.99168 | — |
| cos `means` | 1.0 | min 0.99184 / mean 0.99637 | — |
| cos `scales` | 1.0 | min 0.99784 / mean 0.99924 | — |
| **mIoU（接收端全链路回放）** | **0.350361** | **0.350852** | **+0.0005** |
| 像素准确率 | 0.762714 | 0.762696 | −0.00002 |

- **同口径下 FP32 = 0.3504，与训练日志的 0.3499 一致**（差 0.0005 来自 batch size 8 vs 1 的浮点累加顺序）。
- **Hailo 与 FP32 的差异在噪声内**：2000 张配对 t 检验，per-image ΔmIoU 均值 **−0.000543**，
  **t = −0.70，p = 0.48（不显著）**。结论：**全 Hailo 量化误差对分割精度的影响可以忽略**。
- **`z_hat` 符号 100% 整数**（`max|x-round(x)| = 2.4e-07`），2000 张无一例外 —— host 取整这条路走通了。

**⚠️ 500 张子集（`val_manifest.json` 的 `subset`，场景分层）的 mIoU 偏低是「评估集口径」问题，不是精度问题**：

| 口径 | 图片数 | PyTorch FP32 mIoU | pixel_acc | 出现类别数 |
|---|---|---|---|---|
| 训练评估（全量 validation） | 2000 | **0.3499** | 0.7627 | 150 |
| 部署评估（场景分层 subset） | 500 | **0.2865** | 0.7008 | 147 |

**实测归因**（脚本 `deploy/diagnose_miou_gap.py`，同一条解码路径同时跑两套口径）：

- **解码路径逐位相同**：同一批图上，训练路径 `student(x)` 与部署路径
  `bb → be → round(y−means)+means → bn._synthesize → student._receiver`
  的 `y / z_hat / means / scales / x_hat / logits` **max|diff| 全部 = 0.0**。
  → **不是量化/取整/标签对齐的问题**。
- **预处理逐位相同**：dataset 读图 vs 文件读图，image / label / logits **max|diff| = 0.0**
  （`deploy/check_preproc_equivalence.py`）。→ **不是预处理不一致的问题**。
- **标签映射正确**：ADE20K source id 1..150 → 0..149、0 → 255，两条路径 0 个像素不一致。
- **唯一变量 = 评估集**：2000 张上 mIoU = 0.3498；同 2000 张里随机抽 500 张，
  mIoU = **0.3407 ± 0.0118**（p5 = 0.3217，p95 = 0.3599）—— 随机子集仍然在 0.34 附近，
  **而这个分层 subset 落在 200 次随机抽样的第 0 位（百分位 0.0%，即最差的那一档）**。
- **机制**：mIoU 对**每个出现的类别等权平均**。subset 里稀有类几乎塌掉：
  class 105 在 subset 只有 **1 张图**（全量 4 张），IoU 从 0.513 → **0.000**；
  class 37 只 4 张（全量 37 张），IoU 0.544 → 0.015。
  subset 中「≤2 张图」的 21 个类，平均 |ΔIoU| = **0.129**；
  「≥20 张图」的 38 个类只有 **0.077**。
  按 gt 像素量加权的 IoU（全量 0.632 vs subset 0.558）差距也明显小于等权 mIoU 的差距。

> **结论**：0.35 → 0.29 完全是 **500 张分层子集这个样本本身偏难 + 稀有类塌陷** 造成的
> 评估集口径差异，**与 Hailo 量化无关，也与任何代码 bug 无关**。
> 报数字时必须写明评估集：**全量 2000 张 = 0.3504**，500 张子集 = 0.2865，两者不可混用。

**可复用脚本**（都在 `/root/Semantic-sensing/deploy/`）：

| 脚本 | 作用 |
|---|---|
| `hailo_split_pipeline.py` | **全 Hailo 链路的可复用实现**：`SplitSenderHailo`（DFC emulator）/ `SplitSenderBoard`（HailoRT 真板）两个后端，接口一致；`host_round_z()` 是那个 host 标量算子 |
| `dump_reference_500.py` | dump PyTorch FP32 参考张量 + 走完整接收端算出 FP32 基线 mIoU（`--split subset|full`） |
| `eval_all_hailo_500.py` | 统一评估：特征余弦 / 真实 rANS bpp / mIoU 代理 / 离群图归因 |
| `analyze_all_hailo_500.py` | 深入分析：64 vs 500 一致性、符号翻转归因、mIoU 差值显著性 |
| `diagnose_miou_gap.py` | **口径归因**：同一批图上同时跑「训练路径」与「部署路径」，逐个中间张量比 max\|diff\|，并按 `--scope full\|subset500` 切换评估集 |
| `repro_train_point.py` | **用原代码路径**（`evaluate_student` + `make_loader`）复现训练日志的 `[POINT]` 行 |
| `bootstrap_miou_subset.py` | 500 张子集 vs 全量的 mIoU 分布：随机子集抽样 + 稀有类塌陷归因 |
| `check_preproc_equivalence.py` | dataset 读图 vs 文件读图的 image/label/logits 逐像素一致性 |
| `compare_bpp_conventions.py` | 「估计 bpp」（训练日志）与「真实 rANS bpp」（部署）的口径对照 |

```bash
# 复现（全量 2000 张，与训练日志同口径 —— 推荐口径）
cd /root/deploy_artifacts/hailo/sender && nohup \
  /root/miniconda3/envs/hailo-dfc/bin/python \
  /root/Semantic-sensing/deploy/hailo_split_pipeline.py \
    --split full --n 2000 --chunk 50 --out-dir /dev/shm/hailo2000/hailo > split2000.log 2>&1 &
# 参考与评估（torch venv）
/root/Semantic-sensing/venv-hailo/bin/python /root/Semantic-sensing/deploy/dump_reference_500.py \
  --split full --n 2000 --device cuda --out-dir /dev/shm/hailo2000/ref
/root/Semantic-sensing/venv-hailo/bin/python /root/Semantic-sensing/deploy/eval_all_hailo_500.py \
  --ref-dir /dev/shm/hailo2000/ref --hailo-dir /dev/shm/hailo2000/hailo \
  --out-dir /root/deploy_artifacts/data/eval/all_hailo2000 --n 2000 --device cuda

# 复现训练日志的 [POINT]（用原代码路径，2000 张）
/root/Semantic-sensing/venv-hailo/bin/python /root/Semantic-sensing/deploy/repro_train_point.py \
  --device cuda --out-dir /root/deploy_artifacts/data/eval/gap_diag
# 归因：口径差异定位
/root/Semantic-sensing/venv-hailo/bin/python /root/Semantic-sensing/deploy/diagnose_miou_gap.py \
  --scope subset500 --path both --device cuda
/root/Semantic-sensing/venv-hailo/bin/python /root/Semantic-sensing/deploy/bootstrap_miou_subset.py \
  --n 2000 --subset-size 500 --draws 200 --device cuda
```

**⚠️ 遗留 bug（本轮修掉）**：`dump_reference_500.py` 与 `eval_all_hailo_500.py` 之前把模型留在 CPU
却把输入张量 `.to(device)`，因此 `--device cuda` 会直接报
`Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)`；
且 rANS 熵编码器只能在 CPU 跑。现在：模型按阶段显式 `.to(device)`，rate 阶段强制 CPU、
mIoU 阶段再移回 device。**这两个脚本之前只在 `--device cpu` 下跑过，所以 bug 没暴露。**

**遗留**：`sender_bottleneck.hef`（单图版本，图内含 `Floor` → bpp 翻 2.3 倍的那个有 bug 的 HEF）
**保留但不使用**，勿误用。

---

## 8. 阶段三：Compile（→ HEF）

```python
hef = runner.compile()        # 默认目标 Hailo-8，无需额外指定
```
> 文档明说：**除 Performance Mode 外，编译命令多为高级/边缘情况，DFC 已自动最大化性能。Demo 3 一句 `compile()` 即可。**

要点：
- **Context Switch**（`context_switch_param(mode=allowed/enabled/disabled)`）：模型太大一次铺不下时自动切多 context，推理时来回切换 = "compile≠板上跑得好"的根源。**编完看 # of Contexts，这是 RTFormer 对 26 TOPS 是否偏大的早期信号。** `disabled` 会让必须多 context 的模型编译失败。
- `allocator_param(automatic_ddr=...)`：PCIe DDR 缓冲，我们 PCIe 默认 True 即可（Ethernet 平台才需 False）。
- **Performance Mode**：拉满利用率冲 FPS，编译时间剧增 → Phase 3 才用。
- **`.auto.alls`（可复现）**：compiled HAR 里自动生成，记录复现该 HEF 的精确指令。
  ```bash
  hailo har extract <COMPILED_HAR> --auto-model-script-path auto.alls
  hailo compiler <QUANTIZED_HAR> --model-script auto.alls   # 快速重编同一 HEF
  ```
  出论文 HEF/性能数字时存好它，保证可复现。
- `get_hailo_runtime_model()`（preview）：把 HEF 包回 ONNX，用 ONNXRuntime 跑前后处理、核心段交 Hailo。要求可切 pre/main/post 三段（opset 8/11-17）。我们的切分架构可能用得上，Demo 3 不必。
- 忽略：place/shortcut/portal/merge/buffers/output_mux 等手动分配命令。

---

## 9. Profiler（不上板时的仪表盘）

```bash
hailo profiler network.har          # Native=概览 / Quantized=优化+噪声 / Compiled=+编译数据
```
关键字段：**# of Contexts**（1=单 context，2+=大模型）、Throughput/FPS、Latency、Power(±20%)、SNR Chart。

⚠️ **硬限制（划定 Demo 3 边界）**：
- **单 context**：host 上就能估 FPS/latency/power。
- **多 context**：**性能数据不可得**，除非 `--runtime-data`（须板上 `hailortcli run2 -m raw measure-fw-actions` 生成）。

→ 即：**RTFormer 若多 context，不上板拿不到性能数字**；host 上止步于"能编 + context 数 + 掉点"。编完先看 # of Contexts，它决定 Demo 3 能走多远。

**SNR 诊断**（`analyze_noise` / `hailo analyze-noise`）：
- SNR chart（顶部）= 量化到当前层对最终输出的影响；SNR on-layer = 该层局部（末层低属正常）。
- 低 SNR 层 → 用 §6.1/§6.3/§6.4 针对性救。把"猜哪层掉点"变"看图定位"。

---

## 10. Demo 3 完整链路（落地顺序）

```
1. PyTorch → 静态 ONNX(opset 11-14) → onnxsim
2. sanity：先拿最小子图（stem+s1~3 纯卷积段）走通 parse→optimize→compile，隔离环境风险
3. 单独 parse 一个 GFA block + 一个 attention block，暴露 double-norm/matmul/softmax 算子坑（#1 风险）
4. 整网 parse：translate_onnx_model（先不带可选参数）→ visualizer 看图 → 必要时 end_node_names 切分
5. NATIVE emulator 验 parse 数值对齐
6. optimize(calib≥1024, NHWC) + .alls 量化命令 → QUANTIZED emulator 验掉点
7. 当前 semantic-sensing 项目只用完整 976 张 held-out（`1024..1999`）的 fixed-19
   large-object macro IoU 与同 checkpoint FP32 reference 成对判断；小样本只作诊断
8. compile() → 看 # of Contexts → 存 .auto.alls
```

**达标即停的 checkpoint（Demo 3 验收线）**：
1. ✅ RTFormer（或其关键 block，尤其 GFA double-norm）能过 DFC parse
2. ✅ NATIVE emulator 数值与 PyTorch 对齐
3. ✅ 当前 ConvBN checkpoint 的完整 976 张 fixed-19 large-object macro IoU 已落盘，
   与 FP32 reference `0.3529407298349502` 成对比较；不使用 150 类 mIoU 选型
4. ✅ compile 成功出 HEF，记录 # of Contexts
5. （可选）单 context 时记录 Profiler 的 FPS/latency 估计

板子（HailoRT 实测延迟/功耗）= Phase 3 可选，Demo 3 不做。

---

## 11. 与上层项目的接口约定（防跑偏）

- 本项目方法论 = **Learned Source Coding for Machines**，**不是 JSCC**：Hailo 部署不涉及信道，纯信源编码部件的部署。
- bottleneck = 压缩点 + 算力切分点，在 **stage3 后（conv→attention 边界）**；发送端纯卷积、attention 在接收端 decoder → **两段各自 parse/optimize/compile，对应两块对称 Hailo-8**。
- 训练主线 = feature 蒸馏；Hailo 的 `finetune`（蒸馏式）与之同源，可呼应。
- **创新点在 bottleneck，不在部署**；本部署文档是工程支撑，不是论文创新叙事。
- 4-bit compression_level 与论文 bpp 是两个维度，勿混。
