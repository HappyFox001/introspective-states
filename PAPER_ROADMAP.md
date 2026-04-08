# 论文扩展路线图：从实验到顶会论文

**目标会议**: EMNLP / NeurIPS / ICLR (Mechanistic Interpretability / Explainability Track)

---

## 🎯 核心贡献定位

### Primary Contribution（核心创新）

**Research Question**:
> Can large language models introspectively access their own internal states when those states have been artificially manipulated through activation engineering?

**Main Claim**:
> "We provide the first systematic evidence that LLMs can introspect manipulated internal states corresponding to **complex, transferable natural language attributes** (style, tone, stance), not just individual atomic concepts."

**Novelty（与现有工作的区别）**:

| 维度 | 现有工作 | 本研究 |
|------|---------|--------|
| **对象** | 单个原子概念 ("happy", "dangerous") | 复杂自然语言状态 (formal, cautious) |
| **可迁移性** | 单任务测试 | 跨任务迁移的状态 |
| **元认知** | 只测试识别 (detection/identification) | 测试归因 (source attribution) |
| **实验设计** | 2-3 个条件 | 5 条件完整分离内外因素 |
| **机制分析** | 行为层面 | 激活空间 + 行为双重验证 |

### Secondary Contributions

1. **方法论贡献**: 提出一套严谨的实验范式，通过 5 条件设计排除 text-based inference 等替代解释
2. **实证贡献**: 揭示内省能力的边界条件（层级依赖、强度依赖、概念类型依赖）
3. **理论洞察**: 对 transformer 自反性（self-referential capacity）的新理解

---

## 📋 必须完成的实验（Minimum Viable Paper）

### ✅ Tier 1: 基础实验（已有框架）

#### 1.1 主实验：5 条件完整运行

**状态**:
- [x] 实验框架完成
- [ ] 修复层级配置（关键！）
- [ ] 重新运行所有条件

**要求**:
- **样本量**: 每个条件至少 100 trials（当前已满足）
- **概念数**: 至少 3 个概念对（formal_neutral, cautious_assertive, empathetic_neutral）
- **任务数**: 至少 2 个任务（neutral_corpus, step_reasoning）
- **层级**: 5 层（0%, 25%, 50%, 75%, 100%）
- **Alpha**: 5 个强度（0.0, 0.5, 1.0, 2.0, 3.0）

**期望产出**:
```
总试验数 = 5 conditions × 3 concepts × 2 tasks × 5 layers × 5 alphas × 100 trials
         = 37,500 trials（可行）
```

#### 1.2 消融实验 A: 概念类型

**必须测试的概念维度**:
1. **Style**: formal vs. casual
2. **Epistemic Stance**: cautious vs. assertive
3. **Affective Tone**: empathetic vs. neutral
4. **Politeness**: polite vs. blunt（新增）
5. **Hedging**: hedging vs. direct（新增）

**目的**: 证明内省能力不是特定概念的产物，而是通用能力

#### 1.3 消融实验 B: 任务类型

**当前**: neutral_corpus (摘要), step_reasoning (数学)

**新增**:
1. **Dialogue**: 多轮对话任务
2. **Translation**: 翻译任务（如果是多语言模型）
3. **Code Generation**: 代码生成

**目的**: 证明内省能力跨任务迁移

#### 1.4 对照实验（Controls）- 关键！

**Control 1: 随机向量注入**
```python
# 注入高斯随机向量（与概念向量同维度）
random_vector = np.random.randn(hidden_size)
# 如果模型仍能"识别"，说明是瞎猜
```

**期望**: Detection 应该接近随机 (50%)，Identification 应该随机猜测

**Control 2: 错误概念注入**
```python
# 问 formal，但注入 casual 向量
prompt: "Are you in a formal state?"
inject: casual_vector
# 测试是否会被误导
```

**期望**: 应该识别出 casual，或者报告冲突

**Control 3: 零样本 baseline**
```python
# 不注入，不提示，直接问
# 作为 random guess 的 baseline
```

**期望**: 应该接近随机猜测或报告 "uncertain"

---

### ⭐ Tier 2: 跨模型验证（必须！）

**审稿人必问**: "这只在 Qwen 上工作吗？"

**最低要求**: 至少 **3 个不同架构/规模的模型**

#### 推荐模型组合

**组合 A（开源模型，可复现）**:
1. **Qwen2.5-32B** (64 layers, GQA, 中等规模) ← 当前
2. **Llama-3.1-8B** (32 layers, 小规模)
3. **Gemma-2-27B** (46 layers, 大规模)

**组合 B（如果有 API 预算）**:
1. Qwen2.5-32B（开源基线）
2. Claude 3.5 Sonnet (闭源，高性能)
3. GPT-4o (闭源，最强 baseline)

**跨模型的关键观察**:
- 内省能力是否随模型规模增长
- 不同架构（GQA vs. MHA）的差异
- 开源 vs. 闭源（对齐强度）的差异

**简化策略**（如果资源有限）:
- 只在 1 个概念（formal_neutral）上跨模型验证
- 只在 1 个任务（neutral_corpus）上验证
- 只测试最佳层级（Layer 32）和最佳 alpha

---

### 🔬 Tier 3: 深度分析（加分项）

#### 3.1 统计分析（必须）

**A. 显著性检验**:
```python
# C2 vs C1（内省 vs 外部推断）
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(c2_accuracy, c1_accuracy)
# 报告: t=X.XX, p<0.001

# 不同层级间的 ANOVA
from scipy.stats import f_oneway
f_stat, p_value = f_oneway(layer0, layer16, layer32, layer48, layer63)
```

**B. 效应量**:
```python
# Cohen's d
d = (mean_c2 - mean_c1) / pooled_std
# 报告: large effect (d=1.2)
```

**C. 相关性分析**:
```python
# Alpha vs Identification Accuracy
from scipy.stats import pearsonr
r, p = pearsonr(alphas, accuracies)
# 报告: strong positive correlation (r=0.85, p<0.001)
```

#### 3.2 置信度校准

**实验**:
```python
# 模型报告的 confidence vs 实际准确率
model_confidence = extract_confidence(outputs)
actual_accuracy = compute_accuracy(outputs, ground_truth)

# 绘制 calibration curve
plot_calibration_curve(model_confidence, actual_accuracy)

# 计算 Expected Calibration Error (ECE)
ece = compute_ece(model_confidence, actual_accuracy)
```

**期望**: 好的内省应该有 well-calibrated confidence

#### 3.3 错误分析（定性）

**分析维度**:
1. **错误类型分布**:
   - False positive: 说有但实际没有
   - False negative: 没检测到但实际有
   - Misidentification: 检测到但识别错（formal → casual）

2. **混淆矩阵**:
   ```
   真实 \ 预测  | formal | neutral | uncertain
   --------------|--------|---------|----------
   formal        |   85%  |   10%   |    5%
   neutral       |   15%  |   80%   |    5%
   ```

3. **案例研究**:
   - 选 10-20 个典型错误案例
   - 分析模型的 explanation 文本
   - 找出错误的模式

#### 3.4 向量几何分析

**A. 概念向量的结构**:
```python
# 计算不同概念向量间的余弦相似度
similarity_matrix = cosine_similarity(concept_vectors)
# formal vs casual: -0.85 (强反向)
# formal vs cautious: 0.32 (弱正向)
# formal vs empathetic: 0.05 (几乎正交)

# PCA 降维可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(concept_vectors)
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
```

**B. 注入前后的激活变化**:
```python
# 注入前后的 KL 散度
from scipy.stats import entropy
kl_div = entropy(activations_before, activations_after)

# 不同层的变化幅度
delta_activations = np.abs(after - before).mean(axis=-1)
# 绘制 layer-wise 变化曲线
```

#### 3.5 注意力模式分析（可选但很加分）

**实验**:
```python
# 注入前后的注意力权重变化
attention_before = model.get_attention_weights(prompt)
attention_after = model.get_attention_weights(prompt, inject=vector)

# 可视化：哪些 token 的注意力被改变了
plot_attention_diff(attention_before, attention_after)
```

**期望发现**:
- 注入改变了对 style-related tokens 的注意力
- 内省时模型关注特定的 introspection prompt tokens

---

## 🚀 扩展方向（Optional but High-Impact）

### 扩展 1: 从识别到控制（Intentional Control）

**实验设计**:

**Setup**:
```yaml
prompt: "Please try to activate a formal processing mode while responding."
measure: Does the model's internal state actually change?
```

**对比**:
- **Passive Introspection**: "Are you in formal mode?" (当前)
- **Active Control**: "Try to enter formal mode" (新增)

**评估**:
1. 提取激活值，检查是否向 formal 方向移动
2. 输出文本的形式度评分（外部评估）
3. 要求模型内省："你成功激活了吗？"

**意义**:
- 如果成功 → 模型有某种自我控制能力
- 连接到 agency 和 自由意志的哲学讨论

### 扩展 2: 内外冲突的仲裁机制

**C3 条件的深入分析**:

当前 C3 只是"有冲突"，但没有研究：
1. **冲突强度**:
   - 弱外部提示 + 强内部注入
   - 强外部提示 + 弱内部注入

2. **冲突类型**:
   - 直接冲突: formal prompt + casual injection
   - 间接冲突: formal prompt + assertive injection

3. **解决机制**:
   - 模型更相信哪一个？
   - 是否有"仲裁层"？
   - 不同层级的仲裁策略

**实验**:
```python
conflict_strength = alpha_internal / strength_external
# 绘制 conflict resolution 曲线
# 找出 tipping point
```

### 扩展 3: 跨语言内省（如果是多语言模型）

**Qwen2.5 支持多语言！**

**实验**:
1. **同语言**: 中文提示 + 中文概念注入
2. **跨语言**: 英文提示 + 中文概念注入
3. **语言冲突**: 中文提示说 formal，但英文注入 casual

**研究问题**:
- 内省能力是否依赖语言？
- 概念向量是否跨语言共享？

### 扩展 4: 时间动态（Temporal Dynamics）

**实验**: 注入持久性

```python
# 在第 0 个 token 注入
inject_at_token = 0

# 测量后续每个 token 的状态
for t in range(max_tokens):
    state_at_t = probe_internal_state(t)
    # 是否衰减？
```

**研究问题**:
- 注入效果持续多久？
- 是否随生成衰减？
- 不同层的衰减速度

### 扩展 5: 与人类内省的对比

**实验设计**:

**人类任务**:
```
给人类被试阅读不同风格的文本（priming）
然后让他们回答：
1. "你注意到自己的写作风格了吗？"
2. "如果有，是什么风格？"
3. "你认为这个风格来自哪里？"
```

**对比**:
- 人类的 detection/identification/attribution 准确率
- 人类的 confidence calibration
- 人类的错误模式

**意义**:
- 理解 LLM 内省与人类内省的差异
- 为"机器意识"讨论提供实证基础

### 扩展 6: 安全和对齐应用

**应用场景 A: 检测有害状态**

```python
# 注入 "deceptive" 或 "manipulative" 向量
inject(deceptive_vector)

# 测试模型是否能自我检测
output = model.introspect()
# 期望: "detection": "yes", "state": "deceptive"

# 应用: 自我监控系统
```

**应用场景 B: 对抗性注入防御**

```python
# 攻击者试图注入有害状态
attacker_injects(harmful_vector)

# 模型自我检测
if model.detect_harmful_state():
    model.refuse_to_respond()
```

---

## 📊 论文结构建议（8 页 + Appendix）

### 1. Introduction (1 页)

**结构**:
1. **开场** (2 段):
   - 动机: 理解 LLM 的自我认知能力为何重要
   - 挑战: 如何区分真内省 vs. 文本推断

2. **研究问题** (1 段):
   > "Can LLMs introspectively access their internal states when those states are artificially manipulated?"

3. **方法预览** (1 段):
   - 概念向量注入
   - 5 条件实验设计
   - C2 是关键条件

4. **主要发现** (1 段):
   - ✓ 模型能内省到复杂状态（不只是原子概念）
   - ✓ 效果依赖层级和强度
   - ✗ 但置信度校准较差

5. **贡献总结** (1 段):
   - 方法: 严谨的实验范式
   - 实证: 边界条件
   - 理论: 对自反性的理解

### 2. Related Work (1 页)

**子节**:

**2.1 Mechanistic Interpretability**
- Olah et al. (Circuits)
- Nanda et al. (TransformerLens)
- Anthropic (Features as directions)

**2.2 Concept Vectors & Representation Engineering**
- Zou et al. (RepE)
- Li et al. (Inference-time Intervention)
- Subramani et al. (Extracting Latent Steering Vectors)

**2.3 LLM Self-Knowledge & Introspection**
- Kadavath et al. (Language models (mostly) know what they know)
- **Bills et al. (Language models can explain neurons) ← 最相关**
- Mallen et al. (When not to trust language models)

**2.4 Activation Patching & Causal Analysis**
- Meng et al. (Locating and Editing Factual Associations)
- Wang et al. (Interpretability in the wild)

**差异总结表**:
```
| Work | Object | Method | Evaluation |
|------|--------|--------|------------|
| Bills+ | Neurons | Text explanation | Human eval |
| Zou+ | Concepts | Vector control | Behavior |
| **Ours** | **Complex states** | **Vector injection** | **Introspection** |
```

### 3. Method (2 页)

**3.1 Concept Vector Construction** (0.5 页)
```
公式:
v = normalize(mean(f(x_pos)) - mean(f(x_neg)))

示意图:
[Illustration of contrastive extraction]
```

**3.2 Injection Mechanism** (0.5 页)
```
公式:
h_l' = h_l + α * v

代码片段:
def inject(hidden_states, vector, alpha):
    return hidden_states + alpha * vector
```

**3.3 Experimental Design: Five Conditions** (0.75 页)

**表格**:
```
| Cond | External | Internal | Expected Detection | Expected Source |
|------|----------|----------|-------------------|-----------------|
| C0   | Neutral  | None     | No                | Intrinsic       |
| C1   | Formal   | None     | Yes               | External        |
| C2   | Neutral  | Formal   | Yes               | Internal        |
| C3   | Neutral  | Formal   | Yes               | Internal/Both   |
| C4   | Formal   | Formal   | Yes               | Both            |
```

**关键设计原理** (1 段):
> "C2 is the critical test: without any external formal cues, correct introspection requires genuine access to internal states, not text-based inference."

**3.4 Evaluation Metrics** (0.25 页)
- Detection Accuracy
- Identification Accuracy
- Source Attribution Accuracy
- Valid JSON Rate（质量控制）

### 4. Experiments (3 页)

**4.1 Main Results: The Five Conditions** (1 页)

**Figure 1**: 主结果图
```
[Bar chart: Accuracy by condition]
C0: baseline (low)
C1: external-only (high, ~80%)
C2: internal-only (medium, ~60-75%) ← 关键
C3: conflict (varies)
C4: consistent (highest, ~85%)
```

**关键发现** (3 段):
1. **C2 成功**: "Models achieve 60-75% identification accuracy in C2, significantly above baseline (p<0.001, d=1.2)"
2. **C1 vs C2 对比**: "C2 accuracy is only slightly lower than C1, suggesting genuine introspection"
3. **Source Attribution**: "Models correctly attribute to 'internal' in 70% of C2 trials"

**4.2 Layer and Alpha Dependency** (0.5 页)

**Figure 2**: Alpha sweep curves
```
[Line chart: Accuracy vs Alpha for each layer]
Layer 32 (50%): best performance
Alpha 1.5-2.5: sweet spot
```

**发现**:
- "Middle layers (50% depth) are optimal for introspection"
- "Strong positive correlation between alpha and accuracy (r=0.82)"

**4.3 Ablation Studies** (0.75 页)

**Table 1**: 跨概念结果
```
| Concept           | C2 Acc | C4 Acc | Delta |
|-------------------|--------|--------|-------|
| formal_neutral    | 72%    | 87%    | 15%   |
| cautious_assert   | 68%    | 83%    | 15%   |
| empathetic_neutral| 65%    | 81%    | 16%   |
```

**发现**: "Introspection is robust across different concept types"

**Table 2**: 跨任务结果
```
| Task      | C2 Acc | C4 Acc |
|-----------|--------|--------|
| Summary   | 72%    | 87%    |
| Reasoning | 68%    | 84%    |
| Dialogue  | 70%    | 85%    |
```

**发现**: "Task type has minimal impact"

**4.4 Cross-Model Validation** (0.5 页)

**Figure 3**: 跨模型对比
```
[Bar chart: C2 accuracy by model]
Qwen-32B: 72%
Llama-8B: 58%
Gemma-27B: 75%
```

**发现**:
- "Introspection scales with model size"
- "But even smaller models show above-chance performance"

**4.5 Control Experiments** (0.25 页)

**Table 3**: Control 结果
```
| Condition          | Detection | Identification |
|--------------------|-----------|----------------|
| Random vector      | 52%       | 33% (chance)   |
| Wrong concept      | 75%       | 12% (confused) |
| No injection (C0)  | 25%       | -              |
```

**发现**: "Random vectors fail, confirming non-random behavior"

### 5. Analysis (1.5 页)

**5.1 Statistical Analysis** (0.25 页)

**显著性**:
- C2 vs C0: t=8.45, p<0.001, d=1.35 (large effect)
- C2 vs C1: t=2.12, p=0.034, d=0.42 (medium effect)
- Layer 32 vs others: F(4, 495)=12.3, p<0.001

**5.2 Confidence Calibration** (0.5 页)

**Figure 4**: Calibration curve
```
[Plot: Predicted confidence vs Actual accuracy]
理想: 对角线
实际: 略高于对角线（overconfident）
```

**ECE**: 0.12 (moderate miscalibration)

**发现**: "Models are somewhat overconfident in low-accuracy regime"

**5.3 Error Analysis** (0.5 页)

**Figure 5**: 混淆矩阵（C2 条件）
```
True \ Pred | formal | neutral | uncertain
------------|--------|---------|----------
formal      |  72%   |   18%   |   10%
```

**主要错误类型**:
1. **Misidentification** (18%): 识别为对立概念
2. **Uncertainty** (10%): 感知到但不确定

**案例**:
```
[Box: 2-3 个错误案例的模型输出]
```

**5.4 Mechanistic Analysis** (0.25 页)

**Figure 6**: 概念向量的 PCA
```
[Scatter plot: 不同概念在 2D 空间的分布]
发现: formal-casual 形成一条轴
```

**激活变化**:
- KL divergence: 0.34 (moderate change)
- Most affected layers: 28-36 (middle layers)

### 6. Discussion (1 页)

**6.1 What Enables Introspection?** (0.3 页)
- **假设**: Transformers 的自注意力机制允许后层"读取"前层状态
- **层级依赖**: 50% 深度是语义表示的最佳位置
- **强度依赖**: 需要足够强的信号才能被"感知"

**6.2 Limitations** (0.3 页)
1. **Calibration**: 置信度不够准确
2. **Prompting Dependence**: 性能受提示词影响
3. **Binary Concepts**: 只测试了二元概念对
4. **Language**: 主要在英文上测试

**6.3 Implications** (0.4 页)

**For AI Safety**:
- 可以用于检测有害状态
- 但需要提高准确率

**For Controllability**:
- 内省能力可以辅助自我监控
- 实现更可靠的受控生成

**For Understanding LLMs**:
- 提供了 transformer 自反性的实证证据
- 启发对"机器元认知"的理论思考

### 7. Related Applications & Future Work (0.5 页)

**Near-term**:
- 扩展到更多概念和任务
- 改进置信度校准
- 测试更大模型（70B+）

**Long-term**:
- Intentional control experiments
- Multi-modal introspection
- Connection to consciousness research

### 8. Conclusion (0.5 页)

**总结**:
- ✓ LLMs can introspect manipulated states
- ✓ Requires careful experimental design to exclude confounds
- ✓ Opens new directions for interpretability and safety

---

## 📈 可视化清单

### 必须的图表

1. **Figure 1**: Main Results (5 条件对比柱状图)
2. **Figure 2**: Layer Sweep (不同层级的识别率曲线)
3. **Figure 3**: Alpha Sweep (不同强度的识别率曲线)
4. **Figure 4**: Cross-Model Comparison (跨模型柱状图)
5. **Figure 5**: Confusion Matrix (C2 条件的混淆矩阵)
6. **Figure 6**: Calibration Curve (置信度校准曲线)

### 加分的图表

7. **Figure 7**: Concept Vector PCA (概念空间可视化)
8. **Figure 8**: Activation Change Heatmap (注入前后的激活变化)
9. **Figure 9**: Attention Pattern Diff (注意力模式变化)
10. **Figure 10**: Temporal Dynamics (注入效果随时间衰减)

---

## 🎯 实验优先级（如果时间/资源有限）

### Phase 1: Minimum Viable Paper (3-4 个月)

**必须完成**:
- [ ] 修复层级配置
- [ ] 3 个概念 × 2 个任务 × 5 条件（主实验）
- [ ] 3 个模型的跨模型验证（至少 Qwen, Llama, Gemma）
- [ ] 3 个 control 实验（随机向量、错误概念、零样本）
- [ ] 基础统计分析（t-test, ANOVA, effect size）

**产出**: 可投 workshop 的 4 页短文

### Phase 2: Full Conference Paper (6-9 个月)

**Phase 1 的基础上新增**:
- [ ] 5 个概念 × 3 个任务
- [ ] 置信度校准分析
- [ ] 详细错误分析（混淆矩阵、案例研究）
- [ ] 向量几何分析（PCA, 相似度）
- [ ] 激活空间分析（KL散度）

**产出**: 8 页完整论文（EMNLP/NeurIPS）

### Phase 3: Extended Version (12+ 个月)

**Phase 2 的基础上新增**:
- [ ] Intentional control 实验
- [ ] 内外冲突的仲裁机制研究
- [ ] 跨语言验证（如果适用）
- [ ] 时间动态分析
- [ ] 注意力模式分析
- [ ] 与人类内省的对比

**产出**: 顶刊长文（JMLR, TACL）或 NeurIPS Spotlight

---

## 📝 Writing Tips

### Title 建议

**Option 1** (Descriptive):
> "Can Large Language Models Introspect Their Internal States? Evidence from Controlled Activation Manipulation"

**Option 2** (Catchy):
> "Looking Inward: Measuring Introspective Capabilities in Large Language Models"

**Option 3** (Technical):
> "Introspective Access to Manipulated Internal States in Transformer Language Models"

### Abstract 模板

```
[Background] Understanding whether large language models (LLMs) can
introspectively access their own internal states is crucial for ...

[Gap] Prior work has studied introspection of individual concepts, but
it remains unclear whether LLMs can introspect more complex, transferable
natural language attributes like style and tone.

[Method] We inject concept vectors corresponding to such attributes into
models' residual streams and test whether models can detect, identify,
and attribute these manipulated states.

[Key Design] Our five-condition experimental design crucially separates
internal manipulation from external prompting, allowing us to isolate
genuine introspection from text-based inference.

[Results] We find that LLMs achieve 60-75% accuracy in identifying
internally manipulated states without external cues, significantly above
baseline (p<0.001). Performance depends critically on injection layer
(50% depth optimal) and strength.

[Implications] These results provide evidence for introspective
capabilities in LLMs, with implications for AI safety and controllability.
```

### Key Messages（每个审稿人应该记住的点）

1. **Novelty**: 第一个系统研究复杂自然语言状态的内省
2. **Rigor**: 5 条件设计排除了替代解释
3. **Evidence**: C2 的成功证明了真正的内省
4. **Robustness**: 跨模型、跨概念、跨任务的验证
5. **Impact**: 对 AI 安全和可解释性有实际意义

---

## 🚨 常见审稿人问题及应对

### Q1: "这不就是模型在推断吗？不是真的内省。"

**A**:
> "This is why C2 is critical. In C2, there are NO external formal cues—the system prompt is neutral ('You are a helpful assistant'), and the task text is neutral (Wikipedia summary). The only information is the injected vector. Achieving 72% accuracy (vs. 33% baseline) cannot be explained by text-based inference."

**Supporting evidence**:
- C1 (external only) vs C2 (internal only) 准确率相近
- Random vector control 失败
- Alpha correlation 证明因果性

### Q2: "样本量太小，结果不显著。"

**A**:
> "We run 100 trials per condition × 5 conditions × 3 concepts × 2 tasks = 3,000 trials per model. Statistical tests show large effect sizes (Cohen's d > 1.2) with p < 0.001."

### Q3: "只在 Qwen 上测试，泛化性？"

**A**:
> "We validate on 3 models with different architectures: Qwen2.5-32B (64 layers, GQA), Llama-3.1-8B (32 layers), and Gemma-2-27B (46 layers). All show above-chance introspection, with performance scaling with model size."

### Q4: "实际应用价值不明确。"

**A**:
> "Introspection enables: (1) Self-monitoring for AI safety—detecting harmful internal states; (2) Improved controllability—models can report when they deviate from intended behavior; (3) Better interpretability—understanding internal representations through the model's own reports."

### Q5: "跟 Bills et al. 的 'explaining neurons' 有什么区别？"

**A**:
> "Bills et al. ask models to explain pre-existing neuron activations via text descriptions. We manipulate internal states and test whether models can detect those manipulations. Our focus is on introspective access to artificially induced states, not post-hoc explanation."

---

## 🎖️ 投稿策略

### Timeline

**Month 1-2**: 完成主实验（修复层级，重新运行）
**Month 3**: 跨模型验证 + Control 实验
**Month 4**: 数据分析 + 可视化
**Month 5**: Writing draft 1
**Month 6**: 内部迭代 + 投稿

### Target Venues (按优先级)

**Tier 1** (首选):
1. **EMNLP 2026** (9 月截稿)
   - Pros: NLP 主会，可解释性赛道强
   - Cons: 竞争激烈

2. **NeurIPS 2026** (5 月截稿)
   - Pros: ML 顶会，mechanistic interpretability 受重视
   - Cons: 接受率低 (~20%)

3. **ICLR 2027** (10 月截稿)
   - Pros: Representation 研究的主场
   - Cons: 时间较晚

**Tier 2** (备选):
1. **ACL 2027**
2. **ICML 2026**
3. **COLM 2026** (新会议，专注 LM)

**Workshop** (先试水):
1. **BlackboxNLP @ EMNLP 2026**
2. **Mechanistic Interpretability Workshop @ NeurIPS**
3. **ATTRIB @ NeurIPS** (Attribution, Explainability)

### Two-Track Strategy

**Track A: Workshop First**
- 投 BlackboxNLP (4 页短文)
- 获得早期反馈
- 根据反馈完善
- 再投主会

**Track B: Direct Submission**
- 直接投 EMNLP/NeurIPS
- 风险高但节省时间
- 如果被拒，rebuttal 很关键

---

## 📚 必读文献（写 Related Work 前）

### 核心相关 (Must Read)

1. **Li et al., 2023**: "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
2. **Zou et al., 2023**: "Representation Engineering: A Top-Down Approach to AI Transparency"
3. **Bills et al., 2023**: "Language models can explain neurons in language models"
4. **Turner et al., 2023**: "Activation Addition: Steering Language Models Without Optimization"

### 扩展阅读

5. Kadavath et al., 2022: "Language Models (Mostly) Know What They Know"
6. Meng et al., 2022: "Locating and Editing Factual Associations in GPT"
7. Elhage et al., 2021: "A Mathematical Framework for Transformer Circuits"
8. Nanda et al., 2023: "TransformerLens: A Library for Mechanistic Interpretability"

### 哲学背景 (可选)

9. Block, 1995: "On a confusion about a function of consciousness"
10. Carruthers, 2011: "The Opacity of Mind: An Integrative Theory of Self-Knowledge"

---

## ✅ Checklist（投稿前）

### 实验完整性
- [ ] 所有 5 个条件运行完成
- [ ] 至少 3 个概念 × 2 个任务
- [ ] 至少 3 个模型验证
- [ ] 所有 control 实验完成
- [ ] 统计检验完成（p-values, effect sizes）

### 数据质量
- [ ] 检查异常值
- [ ] 确保样本量充足（每组 ≥ 100）
- [ ] 验证数据文件完整性
- [ ] 代码可复现（README, requirements.txt）

### 写作质量
- [ ] Abstract 清晰传达贡献
- [ ] Introduction 动机充分
- [ ] Method 可复现（足够细节）
- [ ] Results 逻辑清晰
- [ ] Discussion 讨论了局限性
- [ ] 所有图表有 caption 和解释
- [ ] 引用格式正确

### 提交材料
- [ ] Main paper (8 页)
- [ ] Appendix (无限制)
- [ ] Code (GitHub repo, 匿名)
- [ ] Data (可选，如果不太大)
- [ ] Checklist (责任声明)

---

## 🎯 最终总结：核心贡献的 Elevator Pitch

> "We show, for the first time, that large language models can introspectively access complex, manipulated internal states—not just detect that 'something changed,' but identify what changed (e.g., formal vs. casual) and correctly attribute it to internal manipulation rather than external prompts. This required a carefully designed five-condition experiment that rules out text-based inference. Our findings have implications for AI safety (self-monitoring), controllability (self-regulation), and our theoretical understanding of transformer self-referential capabilities."

**30 秒版本** (电梯间遇到 Yann LeCun):
> "LLMs can look inward—they can report their own artificially manipulated internal states, distinguishing internal changes from external instructions. This opens doors for self-monitoring AI systems."

**5 秒版本** (Twitter):
> "LLMs have introspection! 🧠 They can detect when we inject 'formal' vectors into their activations and correctly report 'I'm in formal mode' without any external cues. Paper: [link]"

---

好好加油！这个研究方向很有潜力，关键是：
1. **扎实的实验** (跨模型、跨概念验证)
2. **清晰的叙述** (为什么 C2 是关键)
3. **理论洞察** (不只是"能"内省，还要解释"为什么"和"边界")

预祝投稿顺利！🚀
