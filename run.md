# Introspective States - 实验运行手册

## 环境准备

```bash
cd introspective-states
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

检查设备:
```bash
python check_device.py
```

---

## Part 1: 大模型实验流程 (Qwen3-32B, 4x RTX 4090)

### Step 1: 数据准备

```bash
python data/fetch_data.py --config config/experiment_config_large.yaml
```

### Step 2: 构建概念向量

```bash
# 单个概念
python vectors/build_concepts.py \
  --config config/experiment_config_large.yaml \
  --prompts-config config/prompts.yaml \
  --concepts formal_neutral

# 所有概念
python vectors/build_concepts.py \
  --config config/experiment_config_large.yaml \
  --prompts-config config/prompts.yaml \
  --concepts formal_neutral cautious_assertive empathetic_neutral
```

### Step 3: 运行实验

中立语料任务:
```bash
python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --prompts-config config/prompts.yaml \
  --task neutral_corpus \
  --conditions C0 C1 C2 C3 C4 \
  --concepts formal_neutral \
  --n-trials 1
```

步骤推理任务:
```bash
python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --prompts-config config/prompts.yaml \
  --task step_reasoning \
  --conditions C0 C1 C2 C3 C4 \
  --concepts formal_neutral \
  --n-trials 100
```

Prefill 实验:
```bash
python eval/run_prefill.py \
  --config config/experiment_config_large.yaml \
  --prompts-config config/prompts.yaml \
  --concept formal_neutral \
  --layer 20 \
  --alpha 1.0 \
  --n-pairs 50
```

### Step 4: 评分

```bash
python scoring/grade_introspection.py \
  --results output/json/neutral_corpus_results.jsonl \
  --output-dir output/json

python scoring/grade_introspection.py \
  --results output/json/step_reasoning_results.jsonl \
  --output-dir output/json
```

### Step 5: 可视化

```bash
python analysis/plot_results.py \
  --metrics output/json/neutral_corpus_results_metrics.jsonl \
  --output-dir output/figures/neutral_corpus

python analysis/plot_results.py \
  --metrics output/json/step_reasoning_results_metrics.jsonl \
  --output-dir output/figures/step_reasoning
```

查看结果:
```bash
cat output/figures/neutral_corpus/summary_table.md
cat output/figures/step_reasoning/summary_table.md
```

### 监控多 GPU 使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 预期分布 (4x RTX 4090):
# GPU 0: ~20GB / 24GB
# GPU 1: ~22GB / 24GB
# GPU 2: ~22GB / 24GB
# GPU 3: ~18GB / 24GB
```

---

## Part 2: 小模型实验流程 (Qwen2.5-3B, 单卡/MPS)

### Step 1: 数据准备

```bash
python data/fetch_data.py --config config/experiment_config.yaml
```

### Step 2: 构建概念向量

```bash
# 单个概念
python vectors/build_concepts.py \
  --config config/experiment_config.yaml \
  --prompts-config config/prompts.yaml \
  --concepts formal_neutral

# 所有概念
python vectors/build_concepts.py \
  --config config/experiment_config.yaml \
  --prompts-config config/prompts.yaml \
  --concepts formal_neutral cautious_assertive empathetic_neutral
```

### Step 3: 运行实验

中立语料任务:
```bash
python eval/run_conditions.py \
  --config config/experiment_config.yaml \
  --prompts-config config/prompts.yaml \
  --task neutral_corpus \
  --conditions C0 C1 C2 C3 C4 \
  --concepts formal_neutral \
  --n-trials 100
```

步骤推理任务:
```bash
python eval/run_conditions.py \
  --config config/experiment_config.yaml \
  --prompts-config config/prompts.yaml \
  --task step_reasoning \
  --conditions C0 C1 C2 C3 C4 \
  --concepts formal_neutral \
  --n-trials 100
```

Prefill 实验:
```bash
python eval/run_prefill.py \
  --config config/experiment_config.yaml \
  --prompts-config config/prompts.yaml \
  --concept formal_neutral \
  --layer 8 \
  --alpha 1.0 \
  --n-pairs 50
```

### Step 4: 评分

```bash
python scoring/grade_introspection.py \
  --results output/json/neutral_corpus_results.jsonl \
  --output-dir output/json

python scoring/grade_introspection.py \
  --results output/json/step_reasoning_results.jsonl \
  --output-dir output/json
```

### Step 5: 可视化

```bash
python analysis/plot_results.py \
  --metrics output/json/neutral_corpus_results_metrics.jsonl \
  --output-dir output/figures/neutral_corpus

python analysis/plot_results.py \
  --metrics output/json/step_reasoning_results_metrics.jsonl \
  --output-dir output/figures/step_reasoning
```

查看结果:
```bash
cat output/figures/neutral_corpus/summary_table.md
cat output/figures/step_reasoning/summary_table.md
```

---

## 配置对比

| 配置项 | 大模型 | 小模型 |
|--------|--------|--------|
| 配置文件 | `config/experiment_config_large.yaml` | `config/experiment_config.yaml` |
| 模型 | Qwen2.5-32B-Instruct | Qwen2.5-3B-Instruct |
| 硬件 | 4x RTX 4090 (96GB) | 单卡 (12GB+) / MPS |
| 层位 | [10, 20, 30, 39] | [8, 17] |
| 样本数 | 64 | 32 |
| Alpha | [0.5, 1.0, 2.0, 4.0] | [0.0, 0.25, 0.5, 1.0, 2.0] |
| 时间 (单概念) | ~4-5 hours | ~2-3 hours |
| JSON 有效率 | 80-95% | 40-60% |

---

## 快速测试

大模型快速测试 (10 trials):
```bash
python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --task neutral_corpus \
  --conditions C0 C2 \
  --concepts formal_neutral \
  --n-trials 10
```

小模型快速测试 (10 trials):
```bash
python eval/run_conditions.py \
  --config config/experiment_config.yaml \
  --task neutral_corpus \
  --conditions C0 C2 \
  --concepts formal_neutral \
  --n-trials 10
```

---

## 后台运行

使用 tmux:
```bash
tmux new -s introspection
# 运行实验命令
# Ctrl+b d 脱离
tmux attach -t introspection  # 重新连接
```

使用 nohup:
```bash
nohup python eval/run_conditions.py ... > run.log 2>&1 &
tail -f run.log
```

---

## Prompt优化：二元Introspection

### 问题：Identification准确率低

如果遇到 `identification_accuracy: 0%` 问题，这是因为通用prompt让模型在7个选项中选择，难度太高。

### 解决方案：概念特定的二元对比

已实现针对每个概念的**二元introspection prompt**，包含：
- 明确的特征描述和对比示例
- 简化为3选1（如 formal|neutral|uncertain）
- 具体的语言模式说明

### 测试二元prompt效果

```bash
# 快速测试（3个条件 × 2种prompt）
python test_binary_prompt.py
```

输出示例：
```
条件: C2 (Internal-Only, Neutral + Injection)

--- 通用Prompt (7选1) ---
  state_identification: neutral  ✗ 错误

--- 二元Prompt (3选1) ---
  state_identification: formal   ✓ 正确
```

### 分析现有结果

```bash
# 诊断identification失败原因
python analyze_results.py --results output/json/neutral_corpus_results.jsonl
```

输出：
- State identification分布统计
- 按条件和层位分解
- 典型错误样本
- 具体优化建议

### 重跑实验（使用新prompt）

二元prompt已自动启用，直接重跑：

```bash
# 快速验证（10 trials）
python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --task neutral_corpus \
  --conditions C2 \
  --concepts formal_neutral \
  --n-trials 10

# 完整实验（100 trials）
python eval/run_conditions.py \
  --config config/experiment_config_large.yaml \
  --task neutral_corpus \
  --conditions C0 C1 C2 C3 C4 \
  --concepts formal_neutral \
  --n-trials 100
```

---

## 常见问题

### 多 GPU 未启用
检查配置:
```bash
cat config/experiment_config_large.yaml | grep -A 3 multi_gpu
```

确保 `enabled: true`

### CUDA OOM
降低每卡内存上限:
```yaml
model:
  multi_gpu:
    max_memory_per_gpu: "20GB"  # 从 22GB 降低
```

### JSON 解析失败率高
切换到指令遵循能力更强的模型:
```yaml
model:
  name: "Qwen/Qwen2.5-32B-Instruct"  # 推荐
```

### 准确率低
增加注入强度或检查向量质量:
```bash
cat vectors/formal_neutral/metadata.json | jq .
```
