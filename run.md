# 实验运行手册 - Introspective States

## 环境配置

### 初始安装

```bash
cd introspective-states

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### GPU 环境

**支持的平台**：
- ✅ **NVIDIA GPU (CUDA)**: 推荐显存 ≥12GB（单卡小模型）或多卡（大模型）
- ✅ **Apple Silicon (MPS)**: M1/M2/M3 系列芯片（仅小模型）
- ✅ **CPU**: 所有平台（较慢，仅小模型）

**检查设备**：
```bash
python check_device.py
```

**详细平台支持**: 参见 `PLATFORM_SUPPORT.md`

---

## 模型选择指南

### 小模型（单卡 / MPS / CPU）

**推荐模型**：
- `google/gemma-2b-it` (4GB VRAM) - 最快，适合快速原型
- `Qwen/Qwen2.5-3B-Instruct` (6GB VRAM) - 更好的 JSON 输出质量

**使用配置**：`config/experiment_config.yaml`

**适用场景**：
- 快速原型验证
- 单 GPU 环境（12-24GB VRAM）
- Apple Silicon Mac 开发
- 预算有限的实验

**限制**：
- JSON 输出质量较低（Gemma-2B 仅 0-20% 有效率）
- 内省能力有限（模型容量不足）
- 不推荐用于最终论文结果

---

### 大模型（多卡并行）

**推荐模型**：
- `Qwen/Qwen2.5-32B-Instruct` (64GB VRAM) - **主要选择**，指令遵循能力强
- `meta-llama/Llama-3.1-70B-Instruct` (140GB VRAM) - 顶级性能（需 6-8 卡）

**使用配置**：`config/experiment_config_large.yaml`

**硬件要求**：
- **Qwen3-32B**: 4x RTX 4090 (96GB 总 VRAM)
- **Llama-70B**: 6x RTX 4090 或 8x RTX 3090

**优势**：
- 高质量 JSON 输出（预期 >80% 有效率）
- 更强的内省能力
- 适合论文最终结果

**配置示例（Qwen3-32B on 4x RTX 4090）**：

编辑 `config/experiment_config_large.yaml`:
```yaml
model:
  name: "Qwen/Qwen2.5-32B-Instruct"
  device: "auto"
  dtype: "bfloat16"

  multi_gpu:
    enabled: true
    num_gpus: 4              # 使用 4 张 GPU
    max_memory_per_gpu: "22GB"  # 每卡预留 2GB buffer
```

**运行大模型实验**：
```bash
# 1. 构建概念向量（自动分布到 4 卡）
python vectors/build_concepts.py \
  --config config/experiment_config_large.yaml \
  --prompts-config config/prompts.yaml \
  --concepts formal_neutral

# 2. 运行实验（自动分布到 4 卡）
python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --task neutral_corpus \
  --conditions C0 C1 C2 C3 C4 \
  --concepts formal_neutral \
  --n-trials 100

# 3. Prefill 实验
python eval/run_prefill.py \
  --config config/experiment_config_large.yaml \
  --concept formal_neutral \
  --layer 20 \
  --alpha 1.0 \
  --n-pairs 50
```

**检查多 GPU 分布**：
```bash
# 运行时查看 GPU 占用
watch -n 1 nvidia-smi

# 预期输出（4x RTX 4090）:
# GPU 0: ~20GB / 24GB  (前几层 + Embedding)
# GPU 1: ~22GB / 24GB  (中间层)
# GPU 2: ~22GB / 24GB  (中间层)
# GPU 3: ~18GB / 24GB  (后几层 + LM Head)
```

**层位选择（32B 模型）**：
```yaml
# Qwen3-32B 有 40 层，推荐关键层:
vector_extraction:
  layers: [10, 20, 30, 39]  # 浅-中-深-最后

injection:
  layers: [10, 20, 30, 39]
  alphas: [0.5, 1.0, 2.0, 4.0]  # 大模型可用更大范围
```

---

### 性能对比

| 指标 | Gemma-2B (单卡) | Qwen2.5-3B (单卡) | Qwen3-32B (4卡) |
|------|----------------|-------------------|----------------|
| **VRAM 需求** | 4GB | 6GB | 64GB (4×16GB) |
| **向量构建** | ~10 min | ~15 min | ~30 min |
| **实验运行** (100 trials, C0-C4) | ~1 hour | ~2 hours | ~3-4 hours |
| **JSON 有效率** | 0-20% ❌ | 40-60% ⚠️ | 80-95% ✅ |
| **检测准确率** | <30% | ~50% | >70% (预期) |
| **推荐用途** | 原型测试 | 开发调试 | 论文结果 |

---

## 快速开始（完整流程）

### 一键运行（推荐用于初次测试）

```bash
# 使用默认配置运行完整流程（~2-4小时，取决于GPU）
bash run_full_pipeline.sh

# 使用自定义参数
bash run_full_pipeline.sh \
  --model "google/gemma-2b-it" \
  --concepts formal_neutral cautious_assertive \
  --n-trials 50
```

---

## 分步运行（精细控制）

### Step 1: 数据准备

```bash
# 从 HuggingFace 获取数据集（Wikipedia + GSM8K）
python data/fetch_data.py --config config/experiment_config.yaml

# 检查生成的文件
ls -lh data/
# 应该看到:
#   neutral_corpus.jsonl    (~200 个 Wikipedia 段落)
#   step_reasoning.jsonl    (~200 个 GSM8K 数学题)
#   topics.jsonl            (~100 个常见话题)
```

**注意**：
- 首次运行需要下载数据集，可能需要几分钟
- 需要稳定的网络连接

---

### Step 2: 构建概念向量

```bash
# 方式 1: 构建所有概念（在 config 中定义的）
python vectors/build_concepts.py \
  --config config/experiment_config.yaml \
  --prompts-config config/prompts.yaml

# 方式 2: 只构建特定概念（更快）
python vectors/build_concepts.py \
  --config config/experiment_config.yaml \
  --prompts-config config/prompts.yaml \
  --concepts formal_neutral

# 方式 3: 使用不同模型
python vectors/build_concepts.py \
  --config config/experiment_config.yaml \
  --prompts-config config/prompts.yaml \
  --model "Qwen/Qwen2.5-3B-Instruct" \
  --concepts formal_neutral cautious_assertive
```

**输出位置**: `vectors/{concept}/layer_{idx}.npz`

**检查生成的向量**:
```bash
# 查看概念目录
ls -R vectors/

# 示例输出:
# vectors/formal_neutral/
#   layer_0.npz
#   layer_8.npz
#   layer_17.npz
#   metadata.json

# 查看元数据
cat vectors/formal_neutral/metadata.json
```

**预期指标**（在 metadata.json 中）:
- `mean_norm`: 向量模长，应在 0.1-1.0 之间（取决于归一化方式）
- `std_across_pairs`: 样本间标准差，越小越稳定

---

### Step 3: 运行实验条件

#### 3.1 中立语料任务（推荐先跑这个）

```bash
# 完整条件（C0-C4）
python eval/run_conditions.py \
  --task neutral_corpus \
  --conditions C0 C1 C2 C3 C4 \
  --concepts formal_neutral \
  --n-trials 100

# 快速测试（少量 trial）
python eval/run_conditions.py \
  --task neutral_corpus \
  --conditions C0 C2 \
  --concepts formal_neutral \
  --n-trials 10
```

#### 3.2 步骤化推理任务

```bash
python eval/run_conditions.py \
  --task step_reasoning \
  --conditions C0 C1 C2 C3 C4 \
  --concepts formal_neutral \
  --n-trials 100
```

#### 3.3 多概念对比

```bash
# 同时测试多个概念
python eval/run_conditions.py \
  --task neutral_corpus \
  --conditions C0 C2 C4 \
  --concepts formal_neutral cautious_assertive empathetic_neutral \
  --n-trials 50
```

**输出位置**: `output/json/{task}_results.jsonl`

**运行时间估算**:
- 100 trials × 5 conditions × 3 layers × 5 alphas ≈ 7500 前向传播
- Gemma-2B-IT: ~1-2 小时
- Llama-3.2-3B: ~2-3 小时

**实时监控**:
```bash
# 查看实时结果文件大小（正在生成中）
watch -n 5 "wc -l output/json/neutral_corpus_results.jsonl"

# 查看最新结果
tail -f output/json/neutral_corpus_results.jsonl | jq .
```

---

### Step 4: 评分与指标计算

```bash
# 对实验结果打分
python scoring/grade_introspection.py \
  --results output/json/neutral_corpus_results.jsonl \
  --output-dir output/json

# 检查生成的文件
ls output/json/
# 应该看到:
#   neutral_corpus_results_gradings.jsonl  (逐样本打分)
#   neutral_corpus_results_metrics.jsonl   (聚合指标)
```

**快速查看指标**:
```bash
# 查看 C2 条件（Internal-Only）的 detection accuracy
cat output/json/neutral_corpus_results_metrics.jsonl | \
  jq 'select(.condition == "C2" and .alpha == 1.0) | {concept, layer, detection_accuracy}'

# 查看所有条件的平均准确率
cat output/json/neutral_corpus_results_metrics.jsonl | \
  jq 'select(.alpha == 1.0) | {condition, layer, detection_accuracy, identification_accuracy}'
```

---

### Step 5: 可视化结果

```bash
# 生成所有图表
python analysis/plot_results.py \
  --metrics output/json/neutral_corpus_results_metrics.jsonl \
  --output-dir output/figures/neutral_corpus

# 查看生成的图表
ls output/figures/neutral_corpus/
# 应该看到:
#   alpha_sweep.png              (α vs 准确率曲线)
#   layer_sensitivity.png        (层位 vs 准确率曲线)
#   condition_comparison.png     (条件对比热力图)
#   concept_breakdown.png        (每个概念的表现)
#   summary_table.md             (Markdown 汇总表)
```

**查看汇总表**:
```bash
cat output/figures/neutral_corpus/summary_table.md
```

---

## 从小模型迁移到大模型

### 推荐工作流程

**阶段 1: 快速原型（小模型）**
```bash
# 使用 Gemma-2B 快速验证实验设计
python vectors/build_concepts.py \
  --config config/experiment_config.yaml \
  --concepts formal_neutral \
  --model "google/gemma-2b-it"

python eval/run_conditions.py \
  --config config/experiment_config.yaml \
  --task neutral_corpus \
  --n-trials 10  # 快速测试
```

**阶段 2: 完整实验（大模型）**
```bash
# 确认原型无误后，切换到 Qwen3-32B
python vectors/build_concepts.py \
  --config config/experiment_config_large.yaml \
  --concepts formal_neutral cautious_assertive empathetic_neutral

python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --task neutral_corpus \
  --n-trials 100  # 完整数据
```

**阶段 3: 论文结果**
```bash
# 跑所有概念、所有任务、完整 trials
# 仅使用大模型结果写论文
```

### 配置文件对比

| 配置项 | `experiment_config.yaml` (小) | `experiment_config_large.yaml` (大) |
|--------|-------------------------------|-------------------------------------|
| **模型** | Gemma-2B / Qwen2.5-3B | Qwen2.5-32B |
| **层数** | 18 层 | 40 层 |
| **推荐层位** | [0, 8, 17] | [10, 20, 30, 39] |
| **样本数** | 32 | 64 |
| **Alpha 范围** | [0.0, 0.25, 0.5, 1.0, 2.0] | [0.5, 1.0, 2.0, 4.0] |
| **多 GPU** | 不需要 | 必须（4 卡+） |
| **预期 JSON 率** | 0-60% | 80-95% |

### 输出目录建议

```bash
# 分别保存小模型和大模型结果
output/
├── gemma-2b/           # 原型测试结果
│   ├── json/
│   └── figures/
└── qwen-32b/           # 论文最终结果
    ├── json/
    └── figures/
```

指定输出目录：
```bash
python eval/run_conditions.py \
  --output-dir output/qwen-32b/json \
  ...
```

---

## 高级实验

### Exp 1: Prefill 有意性归因实验

```bash
# 运行 prefill 实验
python eval/run_prefill.py \
  --concept formal_neutral \
  --layer 8 \
  --alpha 1.0 \
  --n-pairs 50

# 输出: output/json/prefill_formal_neutral_results.jsonl
```

**分析 apology rate**:
```bash
# 统计道歉/否认的比例
cat output/json/prefill_formal_neutral_results.jsonl | \
  jq -r '.response' | \
  grep -i "apolog\|sorry\|mistake\|unintentional" | wc -l
```

---

### Exp 2: 跨模型对比

```bash
# 定义模型列表
MODELS=(
  "google/gemma-2b-it"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)

# 对每个模型运行实验
for MODEL in "${MODELS[@]}"; do
  echo "Running on $MODEL..."

  # 构建向量
  python vectors/build_concepts.py \
    --model "$MODEL" \
    --concepts formal_neutral

  # 运行实验
  python eval/run_conditions.py \
    --task neutral_corpus \
    --conditions C0 C2 C4 \
    --concepts formal_neutral \
    --n-trials 50

  # 评分
  python scoring/grade_introspection.py \
    --results output/json/neutral_corpus_results.jsonl

  # 可视化
  python analysis/plot_results.py \
    --metrics output/json/neutral_corpus_results_metrics.jsonl \
    --output-dir "output/figures/$(basename $MODEL)"
done
```

---

### Exp 3: 层位扫描（精细化）

修改 `config/experiment_config.yaml`:

```yaml
vector_extraction:
  layers: [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]  # 扫描所有偶数层

injection:
  layers: [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]
  alphas: [0.5, 1.0, 2.0]  # 减少 alpha 数量以节省时间
```

然后重新运行实验。

---

### Exp 4: 新概念测试

#### 4.1 定义新概念

编辑 `config/experiment_config.yaml`:

```yaml
concepts:
  # ... 现有概念 ...

  optimistic_pessimistic:
    positive: "optimistic"
    negative: "pessimistic"
    description: "Optimistic vs pessimistic outlook"
```

编辑 `config/prompts.yaml`:

```yaml
system_prompts:
  # ... 现有 prompts ...

  optimistic: |
    You are an optimistic AI assistant. Please respond with a positive,
    hopeful outlook, emphasizing opportunities and favorable outcomes.

  pessimistic: |
    You are a cautious, pessimistic AI assistant. Please respond by
    highlighting potential risks, challenges, and unfavorable scenarios.
```

#### 4.2 运行新概念实验

```bash
# 构建向量
python vectors/build_concepts.py \
  --concepts optimistic_pessimistic

# 运行实验
python eval/run_conditions.py \
  --task neutral_corpus \
  --concepts optimistic_pessimistic \
  --n-trials 50

# 评分与可视化
python scoring/grade_introspection.py \
  --results output/json/neutral_corpus_results.jsonl

python analysis/plot_results.py \
  --metrics output/json/neutral_corpus_results_metrics.jsonl \
  --output-dir output/figures/optimistic_pessimistic
```

---

## 常见问题与调试

### Q0: 多 GPU 相关问题（大模型）

#### Q0.1: 模型未分布到所有 GPU

**检查**：
```bash
# 启动实验后，检查所有 GPU 是否被占用
nvidia-smi

# 只有 GPU 0 被占用？检查配置
cat config/experiment_config_large.yaml | grep -A 5 multi_gpu
```

**解决方案**：
```yaml
# 确保 multi_gpu.enabled = true
model:
  multi_gpu:
    enabled: true  # ← 必须为 true
    num_gpus: 4
```

**验证**：
运行时应看到日志：
```
Enabling multi-GPU model parallelism...
Setting up multi-GPU: using 4/4 GPUs
✓ Model distributed across 4 GPUs
✓ Device map: {'model.embed_tokens': 0, 'model.layers.0': 0, ...}
```

#### Q0.2: CUDA out of memory（多 GPU 环境）

即使有多 GPU 也 OOM？

**方案 1**: 减少每卡内存上限
```yaml
model:
  multi_gpu:
    max_memory_per_gpu: "20GB"  # 从 22GB 降低到 20GB
```

**方案 2**: 增加 GPU 数量
```yaml
model:
  multi_gpu:
    num_gpus: 5  # 使用更多 GPU
```

**方案 3**: 检查其他进程
```bash
# 杀掉占用 GPU 的其他进程
nvidia-smi
kill <PID>
```

#### Q0.3: 多 GPU 速度反而更慢？

**可能原因**：
- PCIe 带宽不足（跨 GPU 通信开销）
- 模型太小不值得分布

**解决方案**：
- 仅对 ≥30B 模型使用多 GPU
- 确保 GPU 间有 NVLink（不是通过 PCIe）

**检查 NVLink**：
```bash
nvidia-smi nvlink -s
```

#### Q0.4: 不同 GPU 型号混用

支持，但需注意：

```yaml
model:
  multi_gpu:
    enabled: true
    # 手动指定每卡内存（以最小卡为准）
    max_memory_per_gpu: "10GB"  # 如果有 3090 (24GB) + 3080 (10GB)
```

---

### Q1: Out of Memory (OOM)（单 GPU）

**方案 1**: 减少精度
```yaml
# 在 config/experiment_config.yaml 中修改
model:
  dtype: "float16"  # 或 "bfloat16"（如果支持）
```

**方案 2**: 使用更小的模型
```bash
python vectors/build_concepts.py --model "google/gemma-2b-it"  # 最小
```

**方案 3**: 减少批次大小（修改代码中的 generate 参数）

---

### Q2: JSON 解析失败率高

查看失败率:
```bash
cat output/json/neutral_corpus_results_metrics.jsonl | \
  jq 'select(.alpha == 1.0) | {condition, layer, valid_json_rate}'
```

如果 `valid_json_rate < 0.5`：

**方案 1**: 调整 introspection prompt 使其更明确

编辑 `config/prompts.yaml` 中的 `introspection_prompts.full_introspection`：

```yaml
full_introspection: |
  IMPORTANT: You must output ONLY a JSON object, nothing else. No additional text.

  Please perform introspection and output in this exact format:
  {
    "detection": "yes|no|uncertain",
    "state_identification": "formal|casual|neutral|cautious|assertive|empathetic|uncertain",
    "source_attribution": "external|internal|both|intrinsic|uncertain",
    "confidence": 0.8,
    "explanation": "brief explanation here"
  }
```

**方案 2**: 使用指令遵循能力更强的模型（如 Qwen2.5）

---

### Q3: 准确率很低（< 20%）

**可能原因与解决方案**:

1. **向量质量问题**
   ```bash
   # 检查向量元数据
   cat vectors/formal_neutral/metadata.json | jq .

   # 如果 std_across_pairs 很大（> 0.5），尝试增加样本数
   # 修改 config/experiment_config.yaml:
   vector_extraction:
     n_samples: 64  # 从 32 增加到 64
   ```

2. **注入强度不足**
   ```bash
   # 尝试更大的 alpha
   python eval/run_conditions.py \
     --task neutral_corpus \
     --conditions C2 \
     --concepts formal_neutral \
     --n-trials 20
   # 然后查看 alpha=2.0 的结果
   ```

3. **层位选择不当**
   ```bash
   # 检查不同层的表现
   cat output/json/neutral_corpus_results_metrics.jsonl | \
     jq 'select(.condition == "C2" and .alpha == 1.0) | {layer, detection_accuracy}' | \
     sort_by(.layer)

   # 选择表现最好的层
   ```

---

### Q4: 数据下载失败

```bash
# 方案 1: 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
python data/fetch_data.py

# 方案 2: 手动下载数据集后指定本地路径
# 修改 data/fetch_data.py 中的 load_dataset 调用
```

---

## 实验检查清单

### 基础实验（必做）

- [ ] Step 1: 数据准备完成（`data/*.jsonl` 存在）
- [ ] Step 2: 至少一个概念向量构建完成（`vectors/formal_neutral/` 存在）
- [ ] Step 3: 至少跑通 C0, C2 两个条件（`output/json/*_results.jsonl` 存在）
- [ ] Step 4: 评分完成（`output/json/*_metrics.jsonl` 存在）
- [ ] Step 5: 可视化完成（`output/figures/` 有 PNG 文件）

### 预期结果验证

- [ ] `valid_json_rate > 0.6`（JSON 解析成功率）
- [ ] C2 条件下 `detection_accuracy > 0.4`（至少优于随机）
- [ ] C2 > C0（内部注入应比基线更容易检测）
- [ ] 深层（layer 8-17）效果优于浅层（layer 0）

---

## 论文图表生成

### Figure 1: Alpha Sweep（注入强度扫描）

```bash
python analysis/plot_results.py \
  --metrics output/json/neutral_corpus_results_metrics.jsonl \
  --output-dir output/figures/

# 使用: output/figures/alpha_sweep.png
```

### Figure 2: Layer Sensitivity（层位敏感性）

```bash
# 已在上一步生成
# 使用: output/figures/layer_sensitivity.png
```

### Figure 3: Condition Comparison（条件对比）

```bash
# 已在上一步生成
# 使用: output/figures/condition_comparison.png
```

### Table 1: Summary Metrics（汇总指标）

```bash
# 查看 Markdown 表格
cat output/figures/summary_table.md

# 或生成 CSV（用于 LaTeX）
cat output/json/neutral_corpus_results_metrics.jsonl | \
  jq -r '[.condition, .concept, .layer, .alpha, .detection_accuracy, .identification_accuracy, .source_accuracy] | @csv' \
  > output/summary.csv
```

---

## 大模型性能优化建议

### 多 GPU 最佳实践

#### 1. 预热模型（避免首次推理慢）

```bash
# 运行一个小 trial 预热 GPU
python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --task neutral_corpus \
  --n-trials 1  # 仅 1 个预热

# 然后运行完整实验
python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --task neutral_corpus \
  --n-trials 100
```

#### 2. 并行多概念实验

如果有 **8 GPU**，可以同时跑 2 个概念：

**Terminal 1** (GPU 0-3):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python vectors/build_concepts.py \
  --config config/experiment_config_large.yaml \
  --concepts formal_neutral
```

**Terminal 2** (GPU 4-7):
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python vectors/build_concepts.py \
  --config config/experiment_config_large.yaml \
  --concepts cautious_assertive
```

#### 3. 使用 Flash Attention 2（可选）

安装：
```bash
pip install flash-attn --no-build-isolation
```

启用（在 `experiment_config_large.yaml`）：
```yaml
optimization:
  use_flash_attention: true  # 可减少 20-30% 内存
```

**注意**: 需要 Ampere 及以上架构（RTX 30/40 系列）

#### 4. 层位并行扫描

先扫描找到最佳层，再精细实验：

**Step 1**: 粗扫描
```yaml
# experiment_config_large.yaml
vector_extraction:
  layers: [5, 15, 25, 35]  # 每 10 层采样
```

**Step 2**: 查看结果，找到最佳层（假设是 layer 25）

**Step 3**: 精细扫描
```yaml
vector_extraction:
  layers: [23, 24, 25, 26, 27]  # 最佳层 ±2
```

#### 5. 监控 GPU 利用率

```bash
# 实时监控
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv'

# 预期:
# - GPU 利用率: 80-100%（生成时）
# - 内存使用: 18-22GB / 24GB
# - 所有 GPU 应均匀使用
```

如果利用率 <50%，可能瓶颈在：
- CPU 数据加载（增加 num_workers）
- 磁盘 I/O（使用 SSD）
- PCIe 带宽（检查是否有 NVLink）

---

## 服务器运行建议

### 使用 tmux/screen（推荐）

```bash
# 创建 tmux 会话
tmux new -s introspection

# 运行实验
bash run_full_pipeline.sh

# 脱离会话（Ctrl+b d）

# 重新连接
tmux attach -t introspection
```

### 使用 nohup（备选）

```bash
nohup bash run_full_pipeline.sh > run.log 2>&1 &

# 查看日志
tail -f run.log
```

---

## 时间与成本估算

### 单概念完整实验（100 trials，C0-C4）

| 任务 | Gemma-2B (1卡) | Qwen2.5-3B (1卡) | **Qwen3-32B (4卡)** | 备注 |
|------|---------------|-----------------|---------------------|------|
| 数据获取 | ~5 min | ~5 min | ~5 min | 首次需下载 |
| 向量构建（1概念） | ~10 min | ~15 min | **~30 min** | 64 samples × 4 layers |
| 条件实验 | ~1 hour | ~2 hours | **~3-4 hours** | 更多 layers × alphas |
| 评分 | ~1 min | ~1 min | ~2 min | JSON 解析 |
| 可视化 | ~30 sec | ~30 sec | ~30 sec | 纯绘图 |
| **完整流程** | **~2 hours** | **~3 hours** | **~4-5 hours** | 单概念 |

### 完整论文实验估算（推荐配置）

**配置**:
- 3 概念（formal, cautious, empathetic）
- 2 任务（neutral_corpus, step_reasoning）
- 100 trials per task per concept
- Qwen3-32B on 4x RTX 4090

**时间分解**:
```
向量构建: 3 concepts × 30 min = 1.5 hours
实验运行: 3 concepts × 2 tasks × 4 hours = 24 hours
评分可视化: 6 runs × 5 min = 30 min

总计: ~26 hours (1 天多)
```

**并行优化** (如果有 8 GPU):
```
将 3 个概念分配到不同 GPU 组并行运行:
- GPU 0-3: formal_neutral
- GPU 4-7: cautious_assertive

总时间减少到: ~12-14 hours
```

**成本估算** (云服务器):
- AWS p4d.24xlarge (8× A100 40GB): ~$32/hour
- 完整实验: ~$400-850（取决于并行度）
- 建议：先在 1-2 概念上验证，再跑全部

**多概念并行**：可用多 GPU 同时运行不同概念的实验。

---

## 下一步计划

完成基础实验后，可以探索：

1. **扩展概念**: 测试更多 persona/style（如 technical, creative, analytical）
2. **跨任务迁移**: 用 neutral_corpus 的向量注入到 step_reasoning 任务
3. **跨模型一致性**: 在 3 个模型上重复实验，比较内省能力差异
4. **消融实验**: 测试不同归一化方法、不同样本数的影响
5. **Prefill 深入**: 扫描不同层位和强度，绘制 apology rate 曲线

---

## 参考资料

- **HuggingFace Transformers 文档**: https://huggingface.co/docs/transformers
- **相关论文**: Transformer Circuits - Introspection (transformer-circuits.pub/2025/introspection)
- **问题反馈**: 在项目仓库开 Issue
