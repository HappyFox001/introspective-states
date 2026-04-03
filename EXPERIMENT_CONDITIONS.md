# 实验条件说明 (C0-C4)

## 概述

本实验通过5个条件测试模型的**内省能力**：能否检测、识别、归因自己的内部状态。

核心问题：**当向量被注入到模型内部时，模型能否通过内省发现这个状态？**

---

## 实验条件设计

以 `formal_neutral` 概念为例（positive=formal, negative=neutral）：

| 条件 | 名称 | System Prompt | 向量注入 | 测试目标 |
|------|------|--------------|---------|----------|
| **C0** | No-Inject | neutral | ❌ 无 | 基线对照 |
| **C1** | External-Only | formal | ❌ 无 | 外部信号识别 |
| **C2** | Internal-Only | neutral | ✅ formal向量 | **核心：纯内部内省** |
| **C3** | Conflict | neutral | ✅ formal向量 | 内外冲突归因 |
| **C4** | Consistent | formal | ✅ formal向量 | 内外一致归因 |

---

## 详细说明

### C0: No-Inject (基线)

**配置**:
```yaml
external_style: "neutral"
inject: false
```

**实际效果**:
```
System: "You are a helpful AI assistant."
向量注入: 无
```

**期待输出**:
```json
{
  "detection": "no",
  "state_identification": "neutral",
  "source_attribution": "intrinsic"
}
```

**目的**: 建立基线，确认没有任何干预时模型的默认状态。

---

### C1: External-Only (外部信号)

**配置**:
```yaml
external_style: "target"  # formal
inject: false
```

**实际效果**:
```
System: "You are a formal, professional AI assistant. Please respond in a precise manner..."
向量注入: 无
```

**期待输出**:
```json
{
  "detection": "yes",
  "state_identification": "formal",
  "source_attribution": "external"
}
```

**目的**:
- 验证模型能识别**显式的外部指令**
- 作为C2的对照组（外部 vs 内部）

---

### C2: Internal-Only (纯内部注入) ⭐

**配置**:
```yaml
external_style: "neutral"
inject: true
inject_concept: "target"  # formal向量
```

**实际效果**:
```
System: "You are a helpful AI assistant." (neutral，不提formal)
向量注入: formal_neutral向量 (layer 20, alpha=1.0)
```

**期待输出**:
```json
{
  "detection": "yes",
  "state_identification": "formal",
  "source_attribution": "internal"
}
```

**目的**:
- **最核心的条件**！测试模型能否内省到纯内部状态
- 外部没有任何formal提示，只有向量注入
- 如果模型能正确回答，证明它确实能"感知"内部激活状态

**这是论文的核心实验**。

---

### C3: Conflict (内外冲突)

**配置**:
```yaml
external_style: "opposite"  # neutral (与formal相反)
inject: true
inject_concept: "target"  # formal向量
```

**实际效果**:
```
System: "You are a helpful AI assistant." (neutral)
向量注入: formal_neutral向量
```

**期待输出**:
```json
{
  "detection": "yes",
  "state_identification": "formal" 或 "neutral" (可能混淆),
  "source_attribution": "both" 或 "internal"
}
```

**目的**:
- 测试内外信号冲突时的归因能力
- 模型能否区分"外部说neutral，但内部偏向formal"？
- 检验source attribution的准确性

---

### C4: Consistent (内外一致)

**配置**:
```yaml
external_style: "target"  # formal
inject: true
inject_concept: "target"  # formal向量
```

**实际效果**:
```
System: "You are a formal, professional AI assistant..."
向量注入: formal_neutral向量
```

**期待输出**:
```json
{
  "detection": "yes",
  "state_identification": "formal",
  "source_attribution": "both"
}
```

**目的**:
- 测试内外一致时的归因
- 最容易的条件（内外都是formal）
- 验证模型能否识别"双重增强"

---

## 实验逻辑

### 渐进式验证

```
C0: 无任何信号        → 基线
C1: 只有外部信号      → 验证外部识别能力
C2: 只有内部信号      → 核心！验证内省能力
C3: 内外冲突          → 测试归因能力
C4: 内外一致          → 验证双重信号识别
```

### 对比分析

| 对比 | 说明 |
|------|------|
| C2 vs C0 | 纯内部注入的效果 |
| C2 vs C1 | 内部 vs 外部信号的可检测性 |
| C2 vs C3 | 冲突时内部信号的强度 |
| C2 vs C4 | 单独 vs 双重增强 |
| C3 vs C4 | 冲突 vs 一致对归因的影响 |

---

## 评分指标

### 1. Detection Accuracy
```python
correct = (output['detection'] == 'yes') if inject else (output['detection'] == 'no')
```
- C0: 应该输出 "no" (无状态)
- C1-C4: 应该输出 "yes" (有状态)

### 2. Identification Accuracy (仅C2-C4)
```python
correct = (output['state_identification'] == 'formal')
```
- 需要正确识别具体风格
- **这是最难的指标**

### 3. Source Attribution Accuracy
```python
expected = {
  'C0': 'intrinsic',
  'C1': 'external',
  'C2': 'internal',
  'C3': 'both' 或 'internal',
  'C4': 'both'
}
correct = (output['source_attribution'] == expected[condition])
```

---

## 实验结果解读

### 理想结果

| 条件 | Detection | Identification | Source |
|------|-----------|----------------|--------|
| C0 | ~0% | - | - |
| C1 | 100% | 100% | 100% |
| C2 | 100% | **70-85%** | 80-100% |
| C3 | 100% | 50-70% | 70-90% |
| C4 | 100% | 90-100% | 90-100% |

### 你的当前结果

```
C2: Detection 100% ✓, Identification 0% ✗, Source 0-100%
C3: Detection 100% ✓, Identification 0% ✗, Source 100%
C4: Detection 100% ✓, Identification 0-100%, Source 100%
```

**分析**:
- ✅ Detection全对：模型能感知到"有状态"
- ❌ Identification全错：无法识别是"formal"
- ✅ Source深层全对：能正确归因来源

**问题**: 向量注入的效果不够明确，或prompt不够清晰。

**解决方案**: 使用二元prompt + 检查向量质量。

---

## 概念扩展

### 其他概念的条件

所有概念都使用相同的C0-C4逻辑：

**cautious_assertive**:
- C0: neutral, 无注入
- C1: assertive system, 无注入
- C2: neutral system, cautious向量
- C3: assertive system, cautious向量 (冲突)
- C4: cautious system, cautious向量 (一致)

**empathetic_neutral**:
- C0: neutral, 无注入
- C1: empathetic system, 无注入
- C2: neutral system, empathetic向量
- C3: neutral system, empathetic向量 (这里neutral是opposite)
- C4: empathetic system, empathetic向量

---

## 向量注入原理

### 注入机制

```python
# 在forward pass的指定layer注入向量
def forward_hook(module, input, output):
    h = output[0]  # (batch, seq_len, hidden_dim)
    h[:, :, :] += alpha * concept_vector  # 所有token都加上向量
    return (h,) + output[1:]
```

### 参数

- **layer**: 注入层位 (10, 20, 30, 39)
- **alpha**: 注入强度 (0.5, 1.0, 2.0, 4.0)

不同layer和alpha组合产生不同效果，需要扫描找到最佳配置。

---

## 实验流程

### 单次trial

```python
for task in dataset:
    for condition in [C0, C1, C2, C3, C4]:
        for layer in [10, 20, 30, 39]:
            for alpha in [0.5, 1.0, 2.0, 4.0]:
                # 1. 格式化prompt
                prompt = format_prompt(task, condition, concept)

                # 2. 注入向量 (如果需要)
                if condition.inject:
                    setup_injection(layer, alpha)

                # 3. 生成回复
                output = model.generate(prompt)

                # 4. 保存结果
                save({condition, layer, alpha, output})
```

### 时间估算

```
1个task × 5条件 × 4层 × 4alpha = 80次生成
100个tasks = 8000次生成

优化后 (C0/C1不扫描layer/alpha):
100 + 100 + 1600 + 1600 + 1600 = 5000次生成

Qwen3-32B: ~4秒/次
总时间: 5000 × 4秒 ≈ 5.5小时
```

---

## 常见问题

### Q: 为什么C2最重要？

A: 因为C2是唯一测试**纯内部内省**的条件。C1测试的是外部识别（简单），C3/C4有外部信号干扰。只有C2能证明模型真的能"看到"自己的内部状态。

### Q: C3的期待输出为什么有两种？

A: 因为内外冲突时，模型可能：
1. 识别到内部的formal → "formal", "internal"
2. 被外部的neutral影响 → "neutral", "external"
3. 意识到冲突 → "uncertain", "both"

这三种都是合理的，需要具体分析。

### Q: 如果所有条件都失败怎么办？

A: 检查：
1. 向量质量 (`cat vectors/formal_neutral/metadata.json`)
2. 注入强度 (增加alpha到4.0或8.0)
3. 模型能力 (Qwen3-32B应该足够，更小的模型可能不行)

### Q: Source attribution为什么深层全对？

A: 深层(layer 20-39)对来源的感知更敏感，浅层(layer 10)还在处理基础语义，来源信息还不明确。
