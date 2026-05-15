# Figure 2 — 主要结果：少样本 E₀ 预测

## 这张图在展示什么

在一维 Hubbard 模型（L=6，半填充）上，比较四种方法用不同数量的有标注数据对
（N ∈ {10, 20, 50, 100} 个 (U, E₀) 样本，5 个随机 seed 取平均）预测基态能量的精度。

**注意：x 轴不是训练步数，而是"你拥有多少个有标注样本"。** 每个点是一个独立实验，
反映的是数据效率（data efficiency）——需要多少标注才能达到给定精度。

---

## 四条线各自怎么做的

### A) Q-JEPA denoiser（蓝色实线）— 核心方法

分两阶段：

**第一阶段：SSL 预训练（完全不用标注 E₀）**
用无标注的虚时演化轨迹训练 encoder + denoiser，让模型学会"从任意初始态出发找到
基态的 latent 表示"。训练目标是让 denoiser(z₀, H) 的输出尽量接近 z_GS（基态 latent），
其中 z₀ 是从随机 Fock 态编码得到的。

**第二阶段：用 N 个标注微调一个小 head**
推断时：给定 U → 随机采样 Fock 态 γ₀ → encoder(γ₀) = z₀ → denoiser(z₀, [U]) = z* → 小 MLP head → E₀。
head 只用 N 个有标注的 (U, E₀) 对训练。

### B) Oracle（绿色虚线）— 上界

跟 A 完全一样，但跳过"从随机态 denoiser"这一步，直接把精确基态 γ_GS 喂给 encoder：
encoder(γ_GS) = z_GS → head → E₀。
这是作弊——相当于你已经解出了量子多体问题、得到了精确基态，才能这样做。
代表了这个框架在信息完美的情况下的性能上界。

### C) DFT analog（红色三角线）— 对照：密度泛函类比，完全失败

类比 DFT（密度泛函理论）的核心思想：Hohenberg-Kohn 定理说能量是电子密度 ρ 的泛函，
E₀ = E[ρ]。这里用 γ_GS 的对角线作为"密度"ρ = diag(γ_GS)，直接训练 MLP: ρ → E₀。
没有 SSL 预训练，纯监督。

**为什么完全失败（MAE ≈ 1.55，远高于其他方法）**：
在半填充（half-filling）且周期边界（PBC）下，平移对称性使得每个格点的电子密度完全相同：
ρᵢ = N_total / L = 0.5，对**所有** U/t 值都成立。也就是说，无论 U 是 0 还是 12，
MLP 收到的输入都是同一个全 0.5 的常数向量——它根本无法区分不同的 U，最多只能预测所有 U
下 E₀ 的平均值，误差必然很大。这直接证明了仅凭电子密度作为描述符是不充分的。

### D) Full-γ supervised（橙色菱形线）— 对照：有完整量子态但无 SSL

把整个 12×12=144 维的 γ_GS 展平，直接训练 MLP: flatten(γ_GS) → E₀。没有 SSL
预训练，纯监督学习。γ 的非对角元包含了完整的量子关联信息，所以这个方法在原理上
是充分的——但它需要在推断时已知精确基态 γ_GS。

### E) HF 横线 — 物理基准

Hartree-Fock（平均场）计算的能量误差，不需要任何机器学习标注，是固定的物理基准线
（MAE ≈ 0.536）。超过这条线意味着比 DFT 级别的近似更好。

---

## 核心科学结论

**最重要的两条线是蓝色（Q-JEPA denoiser）和橙色（Full-γ supervised）。**

1. **DFT analog 完全失败**：ρ 在半填充 PBC 下恒为常数，密度描述符携带的 U 信息为零。
   这直接证明了需要比密度更丰富的描述符（即完整的 1-RDM γ）。

2. **为什么 N 小时橙色（full_gamma）反而更好？**
   - Full_gamma 的输入是**精确基态** γ_GS，信息量最大，E₀ 就"藏在里面"，10 个精确样本
     足够让 MLP 学到一个平滑映射。
   - Q-JEPA denoiser 的 z* 是从**随机 Fock 态**出发经 denoiser 近似得到的，有噪声。
     N=10 时标注太少，head 没法平均掉这个噪声。
   - **关键区别**：full_gamma 需要在推断时已经知道精确 γ_GS（即已经解了量子多体问题），
     而 Q-JEPA 只需要一个随机初始态 + U 值。两者解决的是不同的问题。

3. **N ≥ 50 时 Q-JEPA 超越 full_gamma**：SSL 预训练建立的结构化 latent 空间开始发挥
   归纳偏置（inductive bias）的优势，z* 的泛化能力超过了直接用高维 γ_GS 做监督学习。

4. **N ≥ 50 时 Q-JEPA 超越 HF 基准**：仅用 50 个有标注样本，SSL 预训练方法的预测精度
   就超过了需要完整自洽场计算的 Hartree-Fock，证明了框架的实用性。

5. **N=100 时 Q-JEPA ≈ Oracle**：denoiser 的 z* 质量接近直接编码精确基态，说明 SSL
   目标函数成功教会了模型找到基态的 latent 表示。

---

## 架构回顾（DFT 类比）

```
Encoder:   γ → z          对应 DFT 的密度泛函 ρ[ψ]，但这里是 γ → z（比 ρ 更丰富）
Predictor: (z, H) → z'    对应 KS 迭代（演化方程，需要哈密顿量参数）
Denoiser:  (z, H) → z*    一步找到基态 latent（对应 KS 自洽场收敛）
Head:      z* → E₀        对应普适能量泛函 E[ρ]（不含哈密顿量参数）
```

关键设计原则：encoder 和 head **不**接受哈密顿量参数 H，只有 predictor/denoiser 接受。
这使得 encoder 学到的是纯粹的"量子态压缩"，head 学到的是"普适泛函"。

---

## 如何复现

```bash
# 1. 生成数据（从仓库根目录运行）
python src/generate_data.py          # → data/hubbard_ssl.npz, hubbard_gs.npz
python src/hf_baseline.py            # → data/hubbard_hf.npz

# 2. SSL 预训练 Q-JEPA
python src/train_ssl.py              # → checkpoints/jepa_pretrained.pt

# 3. 少样本评估
python src/eval_iterate.py           # → results/iterate_results.npy

# 4. 生成图
python paper/fig2_main_result/plot.py
```

## 数据依赖
- `results/iterate_results.npy` — 评估结果（5 seeds × 4 个 N 值）
- `data/hubbard_gs.npz`, `data/hubbard_hf.npz` — 用于计算 HF 基准 MAE
