# DDIM 模型推理与训练算法详解

## 概述

本工程实现的是 **DDIM (Denoising Diffusion Implicit Models)** 的一个变体，用于条件生成任务（音频驱动的运动生成）。模型直接预测去噪后的数据 `x_start`，而不是预测噪声 `ε`。

---

## 一、噪声调度（Noise Schedule）

### 1.1 Cosine Schedule（默认）

```python
# 时间步归一化
timesteps = arange(0, n_timestep+1) / n_timestep + cosine_s  # cosine_s = 8e-3

# Alpha bar (累积乘积)
alpha_bars = cos²(timesteps / (1 + cosine_s) * π/2)
alpha_bars = alpha_bars / alpha_bars[0]  # 归一化

# Beta 计算
betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
betas = clip(betas, 0, 0.999)
```

**公式：**
```math
\begin{align}
\bar{\alpha}_t &= \frac{\cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)}{\cos^2\left(\frac{s}{1+s} \cdot \frac{\pi}{2}\right)} \\
\beta_t &= 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \quad \beta_t \in [0, 0.999]
\end{align}
```

### 1.2 累积参数

```math
\begin{align}
\alpha_t &= 1 - \beta_t \\
\bar{\alpha}_t &= \prod_{s=1}^{t} \alpha_s \\
\bar{\alpha}_0 &= 1
\end{align}
```

---

## 二、前向扩散过程（Forward Diffusion）

### 2.1 加噪公式

```math
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1-\bar{\alpha}_t) \cdot I)
```

**实现：**
```python
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
```

其中 `ε ~ N(0, I)` 是标准高斯噪声。

### 2.2 代码实现

```python
def q_sample(self, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_t = (
        extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )
    return x_t
```

---

## 三、训练算法（Training）

### 3.1 训练流程

**步骤：**

1. **采样时间步**
   ```python
   t ~ Uniform(0, T-1)  # T = 1000
   ```

2. **前向加噪**
   ```python
   ε ~ N(0, I)
   x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
   ```

3. **模型预测**
   ```python
   x̂_0 = model(x_t, cond_frame, cond, t)
   ```
   
   注意：本模型设置 `predict_epsilon=False`，所以模型直接预测 `x_start`（去噪后的数据），而不是噪声 `ε`。

4. **计算损失**
   ```python
   target = x_0  # 因为 predict_epsilon=False
   loss = PVA_loss(x̂_0, x_0)
   ```

### 3.2 损失函数组成

#### 3.2.1 PVA 损失（Position, Velocity, Acceleration）

对每个部分（scale, pitch, yaw, roll, t, exp）分别计算：

```math
\begin{align}
\text{Position Loss:} \quad \mathcal{L}_P &= \text{MSE}(\hat{x}_0, x_0) \cdot w_{\text{part}} \cdot w_{\text{dim}} \\
\text{Velocity Loss:} \quad \mathcal{L}_V &= \text{MSE}(\hat{v}, v) \cdot w_{\text{part}} \cdot w_{\text{dim}} \\
\text{Acceleration Loss:} \quad \mathcal{L}_A &= \text{MSE}(\hat{a}, a) \cdot w_{\text{part}} \cdot w_{\text{dim}}
\end{align}
```

其中：
- `v = x[1:] - x[:-1]`（速度：相邻帧的差分）
- `a = v[1:] - v[:-1]`（加速度：速度的差分）
- `w_part`：部分权重（如 `part_w_dict`）
- `w_dim`：维度权重（`dim_ws`，可选）

#### 3.2.2 Last Frame 损失（可选）

```math
\mathcal{L}_{\text{last}} = \text{MSE}(\hat{x}_0[:, 0], \text{cond\_frame}) \cdot w_{\text{part}} \cdot w_{\text{dim}}
```

#### 3.2.3 正则化损失（可选）

```math
\mathcal{L}_{\text{reg}} = |\hat{x}_0|_{\text{mean}} \cdot r_w
```

其中 `r_w = 1e-4`（对非 scale 部分）。

#### 3.2.4 总损失

```math
\mathcal{L}_{\text{total}} = \sum_{\text{parts}} (\mathcal{L}_P + \mathcal{L}_V + \mathcal{L}_A) + \mathcal{L}_{\text{last}} + \mathcal{L}_{\text{reg}}
```

### 3.3 训练公式总结

```math
\begin{align}
t &\sim \text{Uniform}(0, T-1) \\
\varepsilon &\sim \mathcal{N}(0, I) \\
x_t &= \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \varepsilon \\
\hat{x}_0 &= f_\theta(x_t, \text{cond\_frame}, \text{cond}, t) \\
\mathcal{L} &= \text{PVA\_Loss}(\hat{x}_0, x_0)
\end{align}
```

---

## 四、推理算法（DDIM Sampling）

### 4.1 DDIM 采样流程

**步骤：**

1. **初始化**
   ```python
   x_T ~ N(0, I)  # 从纯噪声开始
   ```

2. **时间步序列生成**
   ```python
   # 生成采样时间步对
   times = linspace(-1, T-1, sampling_steps+1)  # 默认50步
   times = reversed(times.int())  # [999, 979, 959, ..., 19, -1]
   time_pairs = zip(times[:-1], times[1:])  # [(999,979), (979,959), ..., (19,-1)]
   ```

3. **逐步去噪**（对每个 `(t, t_next)` 对）

   a. **模型预测**
   ```python
   x̂_0, ε̂ = model_predictions(x_t, cond_frame, cond, t)
   ```
   
   其中：
   ```python
   x̂_0 = model(x_t, cond_frame, cond, t)  # 直接预测 x_start
   ε̂ = predict_noise_from_start(x_t, t, x̂_0)  # 从 x̂_0 计算噪声
   ```

   b. **DDIM 更新公式**
   ```python
   if t_next < 0:  # 最后一步
       x = x̂_0
   else:
       α = ᾱ_t
       α_next = ᾱ_{t_next}
       
       # DDIM 系数
       σ = η * sqrt((1 - α/α_next) * (1 - α_next) / (1 - α))
       c = sqrt(1 - α_next - σ²)
       
       # 更新
       x = x̂_0 * sqrt(α_next) + c * ε̂ + σ * z
   ```
   
   其中 `z ~ N(0, I)` 是新的随机噪声，`η = 1`（确定性采样时设为0）。

### 4.2 DDIM 更新公式推导

DDIM 的核心思想是使用确定性或低方差的采样过程。

**标准 DDPM 采样：**
```math
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t z
```

**DDIM 采样（重新参数化）：**
```math
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \varepsilon_\theta(x_t, t) + \sigma_t \cdot z
```

其中：
- `η = 1`：DDIM（确定性，`σ_t = 0`）
- `η = 0`：完全确定性采样
- `0 < η < 1`：介于两者之间

**本实现中的公式：**
```math
\begin{align}
\sigma_t &= \eta \cdot \sqrt{\frac{(1-\bar{\alpha}_t/\bar{\alpha}_{t-1})(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}} \\
c_t &= \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \\
x_{t-1} &= \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + c_t \cdot \hat{\varepsilon} + \sigma_t \cdot z
\end{align}
```

### 4.3 从 x_start 计算噪声

由于模型直接预测 `x_start`，需要从 `x_start` 反推噪声：

```math
\hat{\varepsilon} = \frac{\sqrt{\bar{\alpha}_t}^{-1} \cdot x_t - \hat{x}_0}{\sqrt{1/\bar{\alpha}_t - 1}}
```

**代码实现：**
```python
def predict_noise_from_start(self, x_t, t, x0):
    return (
        (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / 
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    )
```

### 4.4 推理算法总结

```math
\begin{align}
\text{初始化:} \quad &x_T \sim \mathcal{N}(0, I) \\
\text{For } t = T-1, T-2, \ldots, 0: \\
&\hat{x}_0 = f_\theta(x_t, \text{cond\_frame}, \text{cond}, t) \\
&\hat{\varepsilon} = \frac{\sqrt{\bar{\alpha}_t}^{-1} x_t - \hat{x}_0}{\sqrt{1/\bar{\alpha}_t - 1}} \\
&\text{If } t = 0: \\
&\quad x_0 = \hat{x}_0 \\
&\text{Else:} \\
&\quad \sigma_t = \eta \cdot \sqrt{\frac{(1-\bar{\alpha}_t/\bar{\alpha}_{t-1})(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}} \\
&\quad c_t = \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \\
&\quad z \sim \mathcal{N}(0, I) \\
&\quad x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + c_t \cdot \hat{\varepsilon} + \sigma_t \cdot z
\end{align}
```

---

## 五、关键特性

### 5.1 模型输出语义

- **`predict_epsilon=False`**：模型直接预测 `x_start`（去噪后的数据）
- 推理时从 `x_start` 计算 `pred_noise` 用于 DDIM 更新

### 5.2 条件生成

- **条件输入**：
  - `cond_frame`：条件帧 `[B, feat_dim]`
  - `cond`：音频条件 `[B, seq_len, audio_dim]`
- **Classifier-Free Guidance**：
  ```python
  output = uncond + guidance_weight * (cond - uncond)
  ```
  其中 `guidance_weight = 2`，`cond_drop_prob = 0.2`

### 5.3 数据归一化

- 训练数据在输入前进行归一化：`x_norm = (x - μ) / σ`
- 所有计算在归一化空间进行
- 输出也在归一化空间

---

## 六、与标准 DDIM 的差异

1. **模型输出**：直接预测 `x_start` 而非噪声 `ε`
2. **损失函数**：使用 PVA 损失而非简单的 MSE
3. **条件生成**：使用 Classifier-Free Guidance
4. **噪声调度**：使用 Cosine Schedule

---

## 七、算法流程图

### 训练流程
```
x_0 (真实数据)
  ↓
采样 t ~ Uniform(0, T-1)
  ↓
加噪: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
  ↓
模型预测: x̂_0 = model(x_t, cond_frame, cond, t)
  ↓
计算损失: L = PVA_loss(x̂_0, x_0)
  ↓
反向传播更新参数
```

### 推理流程
```
初始化: x_T ~ N(0, I)
  ↓
For t = T-1, T-2, ..., 0:
  ↓
  预测: x̂_0 = model(x_t, cond_frame, cond, t)
  ↓
  计算噪声: ε̂ = predict_noise_from_start(x_t, t, x̂_0)
  ↓
  DDIM更新: x_{t-1} = sqrt(ᾱ_{t-1}) * x̂_0 + c_t * ε̂ + σ_t * z
  ↓
x_0 (生成结果)
```

---

## 八、关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_timestep` | 1000 | 训练时间步总数 |
| `sampling_timesteps` | 50 | 推理采样步数 |
| `schedule` | "cosine" | 噪声调度类型 |
| `predict_epsilon` | False | 模型预测 x_start |
| `guidance_weight` | 2 | Classifier-Free Guidance 权重 |
| `cond_drop_prob` | 0.2 | 条件dropout概率 |
| `clip_denoised` | True | 裁剪到 [-1, 1] |
| `eta` | 1.0 | DDIM采样参数（η=1为DDIM，η=0为确定性） |

---

## 九、数学公式总结

### 前向过程
```math
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)
```

### 训练目标
```math
\mathcal{L} = \mathbb{E}_{t,x_0,\varepsilon} \left[ \text{PVA\_Loss}(f_\theta(x_t, c, t), x_0) \right]
```

### DDIM 采样
```math
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \hat{\varepsilon} + \sigma_t z
```

其中 `σ_t` 由 `η` 参数控制，`η=1` 时为标准 DDIM。
