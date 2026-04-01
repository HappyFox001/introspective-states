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

- **推荐显存**: ≥12GB (用于 Gemma-2B-IT)
- **更大模型**: Llama-3.2-3B / Qwen2.5-3B 需要 ≥16GB

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

### Q1: Out of Memory (OOM)

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

| 任务 | Gemma-2B-IT | Llama-3.2-3B | 备注 |
|------|-------------|--------------|------|
| 数据获取 | ~5 min | ~5 min | 首次需下载 |
| 向量构建（1概念） | ~10 min | ~15 min | 32 samples × 3 layers |
| 条件实验（100 trials） | ~1 hour | ~2 hours | C0-C4, 3 layers, 5 alphas |
| 评分 | ~1 min | ~1 min | 纯计算 |
| 可视化 | ~30 sec | ~30 sec | 纯绘图 |
| **完整流程** | **~2 hours** | **~3 hours** | 单概念，100 trials |

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
