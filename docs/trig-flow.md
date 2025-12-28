# TrigFlow [arxiv.org/html/2410.11081v2](https://arxiv.org/html/2410.11081v2)

## Basic Setup

TrigFlow is a theoretical framework that unifies EDM (Elucidating the Design Space of Diffusion-Based Generative Models), Flow Matching, and Velocity Prediction through a simplified parameterization. The key insight is to use trigonometric functions to parameterize the diffusion process, which significantly simplifies the formulation of diffusion models, the associated probability flow ODE, and consistency models. This unification eliminates the need for complex noise schedules and provides a more elegant mathematical foundation for training generative models.

## Diffusion Process

In TrigFlow, the diffusion process is parameterized using trigonometric functions. Given a data sample $x$ from the target distribution and noise $z \sim \mathcal{N}(0, I)$ from a standard normal distribution, the noisy sample at time $t$ is defined as:

```math
x_t = \cos(t) \cdot x + \sin(t) \cdot z
```

where $t \in [0, \pi/2]$. This parameterization ensures that at $t = 0$, we have $x_0 = x$ (pure data), and at $t = \pi/2$, we have $x_{\pi/2} = z$ (pure noise).

## PF-ODE

The probability flow ODE (PF-ODE) describes the evolution of the diffusion process. The velocity is defined as the time derivative of the position: $v_t = \frac{dx_t}{dt}$

Taking the derivative of $x_t = \cos(t) \cdot x + \sin(t) \cdot z$ with respect to time $t$, we obtain:

```math
v_t = \frac{dx_t}{dt} = -\sin(t) \cdot x + \cos(t) \cdot z
```

## Deriving the Consistency Property

Starting from the diffusion process and velocity definitions:

```math
\begin{align*}
x_t &= \cos(t) \cdot x_0 + \sin(t) \cdot z \\
v_t &= -\sin(t) \cdot x_0 + \cos(t) \cdot z
\end{align*}
```

Solving this linear system for $x_0$ and $z$. We can write it in matrix form:

```math
\begin{bmatrix} x_t \\ v_t \end{bmatrix} = \begin{bmatrix} \cos(t) & \sin(t) \\ -\sin(t) & \cos(t) \end{bmatrix} \begin{bmatrix} x_0 \\ z \end{bmatrix}
```

The coefficient matrix is a rotation matrix with determinant $\cos^2(t) + \sin^2(t) = 1$, so its inverse is its transpose:

```math
\begin{bmatrix} x_0 \\ z \end{bmatrix} = \begin{bmatrix} \cos(t) & -\sin(t) \\ \sin(t) & \cos(t) \end{bmatrix} \begin{bmatrix} x_t \\ v_t \end{bmatrix}
```

Expanding this, we obtain:

```math
\begin{align*}
x_0 &= \cos(t) \cdot x_t - \sin(t) \cdot v_t \\
z &= \sin(t) \cdot x_t + \cos(t) \cdot v_t
\end{align*}
```

Now, consider a state at a different time $r$, where $r \in [0, \pi/2]$ and $r$ can be any time point (not necessarily related to $t$). By the same diffusion process definition:

```math
x_r = \cos(r) \cdot x_0 + \sin(r) \cdot z
```

Substituting the expressions for $x_0$ and $z$:

```math
\begin{align*}
x_r &= \cos(r) \cdot [\cos(t) \cdot x_t - \sin(t) \cdot v_t] + \sin(r) \cdot [\sin(t) \cdot x_t + \cos(t) \cdot v_t] \\
&= [\cos(r)\cos(t) + \sin(r)\sin(t)] \cdot x_t + [-\cos(r)\sin(t) + \sin(r)\cos(t)] \cdot v_t \\
&= \cos(t-r) \cdot x_t - \sin(t-r) \cdot v_t
\end{align*}
```

This leads to the **consistency property**:

```math
x_r = \cos(t-r) \cdot x_t - \sin(t-r) \cdot v_t = \cos(r) \cdot x_0 + \sin(r) \cdot z
```

The consistency property states that if we start from the same noise $z$ and follow the diffusion trajectory toward the data $x_0$, then **any point on this trajectory should lead to the same destination** $x_0$. In other words, whether we are at time $t$ (state $x_t$) or at time $r$ (state $x_r$), both points are on the same path from $z$ to $x_0$, and we can jump directly from one point to another without following the entire ODE trajectory.

## From Consistency Property to Training Objective

Now we derive the training objective for consistency models. The key idea is to define a **consistency function** $g_\theta(x_t, t)$ that maps any point on the trajectory to the data $x_0$.

From the consistency property, we know that $x_0 = \cos(t) \cdot x_t - \sin(t) \cdot v_t$. We define the consistency function as:

```math
g_\theta(x_t, t) = \cos(t) \cdot x_t - \sin(t) \cdot F_\theta(x_t, t)
```

where $F_\theta(x_t, t)$ is the neural network that predicts the velocity $v_t$. If the network is perfect, $F_\theta(x_t, t) = v_t$, then $g_\theta(x_t, t) = x_0$.

### Consistency Constraint

The consistency property requires that for any two points $(x_t, t)$ and $(x_r, r)$ on the same trajectory, they should map to the same $x_0$:

```math
g_\theta(x_t, t) = g_\theta(x_r, r) = x_0
```

This means the consistency function should be constant along the trajectory. Therefore, its total derivative along the trajectory should be zero:

```math
\frac{dg_\theta}{dt} = 0
```

### Compute the Total Derivative

Using the chain rule, the total derivative of $g_\theta$ with respect to time is:

```math
\frac{dg_\theta}{dt} = \frac{\partial g_\theta}{\partial x_t} \cdot \frac{dx_t}{dt} + \frac{\partial g_\theta}{\partial t}
```

Expanding $g_\theta = \cos(t) \cdot x_t - \sin(t) \cdot F_\theta(x_t, t)$, we compute the partial derivatives:

```math
\begin{align*}
\frac{\partial g_\theta}{\partial x_t} &= \cos(t) - \sin(t) \cdot \frac{\partial F_\theta}{\partial x_t} \\
\frac{\partial g_\theta}{\partial t} &= -\sin(t) \cdot x_t - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{\partial F_\theta}{\partial t}
\end{align*}
```

Applying the chain rule:

```math
\begin{align*}
\frac{dg_\theta}{dt} &= \frac{\partial g_\theta}{\partial x_t} \cdot \frac{dx_t}{dt} + \frac{\partial g_\theta}{\partial t} \\
&= \left[\cos(t) - \sin(t) \cdot \frac{\partial F_\theta}{\partial x_t}\right] \cdot \frac{dx_t}{dt} + \left[-\sin(t) \cdot x_t - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{\partial F_\theta}{\partial t}\right] \\
&= \cos(t) \cdot \frac{dx_t}{dt} - \sin(t) \cdot \frac{\partial F_\theta}{\partial x_t} \cdot \frac{dx_t}{dt} - \sin(t) \cdot x_t - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{\partial F_\theta}{\partial t} \\
&= -\sin(t) \cdot x_t + \cos(t) \cdot \frac{dx_t}{dt} - \cos(t) \cdot F_\theta - \sin(t) \cdot \left[\frac{\partial F_\theta}{\partial t} + \frac{\partial F_\theta}{\partial x_t} \cdot \frac{dx_t}{dt}\right] \\
\frac{dg_\theta}{dt} &= -\sin(t) \cdot x_t + \cos(t) \cdot \frac{dx_t}{dt} - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{dF_\theta}{dt} = 0
\end{align*}
```

where $\frac{dF_\theta}{dt} = \frac{\partial F_\theta}{\partial t} + \frac{\partial F_\theta}{\partial x_t} \cdot \frac{dx_t}{dt}$ is the total derivative of $F_\theta$.

We know that $\frac{dx_t}{dt} = v_t = -\sin(t) \cdot x_0 + \cos(t) \cdot z$ and $x_t = \cos(t) \cdot x_0 + \sin(t) \cdot z$. Substituting these into the expression:

```math
\begin{align*}
\frac{dg_\theta}{dt} &= -\sin(t) \cdot x_t + \cos(t) \cdot \frac{dx_t}{dt} - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{dF_\theta}{dt} \\
&= -\sin(t) \cdot [\cos(t) \cdot x_0 + \sin(t) \cdot z] + \cos(t) \cdot [-\sin(t) \cdot x_0 + \cos(t) \cdot z] - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{dF_\theta}{dt} \\
&= -\sin(t)\cos(t) \cdot x_0 - \sin^2(t) \cdot z - \cos(t)\sin(t) \cdot x_0 + \cos^2(t) \cdot z - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{dF_\theta}{dt} \\
&= [-\sin(t)\cos(t) - \cos(t)\sin(t)] \cdot x_0 + [-\sin^2(t) + \cos^2(t)] \cdot z - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{dF_\theta}{dt} \\
&= -2\sin(t)\cos(t) \cdot x_0 + [\cos^2(t) - \sin^2(t)] \cdot z - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{dF_\theta}{dt}
\end{align*}
```

Using trigonometric identities and rearranging terms:

```math
\frac{dg_\theta}{dt} = -\sin(2t) \cdot x_0 + \cos(2t) \cdot z - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{dF_\theta}{dt}
```

Now, we express this in terms of $x_t$ and $v_t$. From the earlier derivation, we have:

```math
\begin{align*}
x_0 &= \cos(t) \cdot x_t - \sin(t) \cdot v_t \\
z &= \sin(t) \cdot x_t + \cos(t) \cdot v_t
\end{align*}
```

Substituting into $-\sin(2t) \cdot x_0 + \cos(2t) \cdot z$:

```math
\begin{align*}
-\sin(2t) \cdot x_0 + \cos(2t) \cdot z &= -\sin(2t) \cdot [\cos(t) \cdot x_t - \sin(t) \cdot v_t] + \cos(2t) \cdot [\sin(t) \cdot x_t + \cos(t) \cdot v_t] \\
&= -\sin(2t)\cos(t) \cdot x_t + \sin(2t)\sin(t) \cdot v_t + \cos(2t)\sin(t) \cdot x_t + \cos(2t)\cos(t) \cdot v_t \\
&= [-\sin(2t)\cos(t) + \cos(2t)\sin(t)] \cdot x_t + [\sin(2t)\sin(t) + \cos(2t)\cos(t)] \cdot v_t
\end{align*}
```

Using trigonometric identities $\sin(2t) = 2\sin(t)\cos(t)$ and $\cos(2t) = \cos^2(t) - \sin^2(t)$:

```math
\begin{align*}
&= [-2\sin(t)\cos^2(t) + \sin(t)(\cos^2(t) - \sin^2(t))] \cdot x_t + [2\sin^2(t)\cos(t) + \cos(t)(\cos^2(t) - \sin^2(t))] \cdot v_t \\
&= [-2\sin(t)\cos^2(t) + \sin(t)\cos^2(t) - \sin^3(t)] \cdot x_t + [2\sin^2(t)\cos(t) + \cos^3(t) - \cos(t)\sin^2(t)] \cdot v_t \\
&= [-\sin(t)\cos^2(t) - \sin^3(t)] \cdot x_t + [\sin^2(t)\cos(t) + \cos^3(t)] \cdot v_t \\
&= -\sin(t)[\cos^2(t) + \sin^2(t)] \cdot x_t + \cos(t)[\sin^2(t) + \cos^2(t)] \cdot v_t \\
&= -\sin(t) \cdot x_t + \cos(t) \cdot v_t
\end{align*}
```

Therefore, the complete expression becomes:

```math
\begin{align*}
\frac{dg_\theta}{dt} &= -\sin(t) \cdot x_t + \cos(t) \cdot v_t - \cos(t) \cdot F_\theta - \sin(t) \cdot \frac{dF_\theta}{dt} \\
&= -\cos(t) \cdot (F_\theta - v_t) - \sin(t) \cdot (x_t + \frac{dF_\theta}{dt})
\end{align*}
```

For the consistency function to remain constant along the trajectory, TrigFlow uses the cosine-scaled consistency condition:

```math
\cos(t) \cdot \frac{dg_\theta}{dt} = 0
```

Substituting the expression for $dg_\theta/dt$:

```math
\cos(t) \cdot \frac{dg_\theta}{dt} = -\cos^2(t) \cdot (F_\theta - v_t) - \cos(t)\sin(t) \cdot (x_t + \frac{dF_\theta}{dt}) = 0
```

The reason for multiplying by $\cos(t)$ will be explained below. For now, we define the **tangent** $g$ using the stop-gradient version $F_\theta^-$:

```math
g := -\cos^2(t) \cdot (F_\theta^- - v_t) - \cos(t)\sin(t) \cdot (x_t + \frac{dF_\theta^-}{dt})
```

where $F_\theta^-$ denotes the stop-gradient version (frozen parameters) used during training for stability.

**Why multiply by $\cos(t)$?** This scaling is a design choice that serves three important purposes:

- **Numerical stability:** When $t \to \pi/2$ (high noise regime), $\cos(t) \to 0$, causing the tangent to naturally decay to zero. Without scaling, the second term $-\sin(t)(x_t + dF_\theta/dt)$ remains $O(1)$, leading to exploding loss in high-noise regions.

- **Velocity error dominance:** The scaled form makes the velocity error term $-\cos^2(t)(F_\theta - v_t)$ the dominant component in low-to-medium noise regions, aligning with TrigFlow's goal of unifying EDM and Flow Matching.

- **Loss scale alignment:** The $\cos(t)$ factor acts as a noise-dependent weighting, similar to the $\sigma(t)$ weighting in EDM formulations, ensuring consistent loss scales across different noise levels.

Note that $\cos(t) \cdot dg_\theta/dt = 0$ is mathematically equivalent to $dg_\theta/dt = 0$ (since $\cos(t) \neq 0$ for $t \in [0, \pi/2)$), but the scaled version provides better training dynamics and numerical stability.

### From Differential Consistency to a Fixed-Point Objective

At this point, we have derived the consistency condition as a differential constraint:

```math
\cos(t) \cdot \frac{dg_\theta}{dt} = 0
```

However, directly minimizing the squared magnitude of this constraint, i.e., $\mathcal{L} = ||\cos(t) \cdot dg_\theta/dt||^2$, would be problematic for several reasons:

1. **Computational complexity:** The constraint involves second-order derivatives (through $dF_\theta/dt$), requiring expensive Hessian computations.
2. **Numerical instability:** Direct optimization of differential constraints often leads to unstable training dynamics, especially when the constraint involves Jacobian-vector products.
3. **Optimization difficulty:** Constrained optimization problems are generally harder to solve than unconstrained regression problems.

Instead, TrigFlow follows the consistency model paradigm and interprets the consistency condition as a **fixed-point constraint**. The key insight is that if the consistency condition holds, then an infinitesimal update along the trajectory should leave the velocity prediction unchanged. This motivates viewing the problem as finding a fixed point of the velocity prediction function.

Specifically, if we consider an infinitesimal step along the trajectory, the consistency condition implies that the velocity prediction should satisfy:

```math
F_\theta(x_t, t) = F_\theta(x_t, t) + \cos(t) \cdot \frac{dg_\theta}{dt}
```

Rearranging and using the definition of the tangent $g = \cos(t) \cdot dg_\theta/dt$, we obtain the **bootstrap target**:

```math
F_\theta^{\text{target}} = F_\theta^- + g
```

where $F_\theta^-$ denotes a stop-gradient (frozen) copy of the network. This formulation transforms the differential constraint into a **self-distillation regression problem**, where the current network $F_\theta$ is trained to match the target $F_\theta^- + g$.

This approach is analogous to:
- **Temporal Difference (TD) learning** in reinforcement learning, where the target is a bootstrapped estimate
- **Consistency Models**, which use a similar fixed-point interpretation
- **Iterative refinement methods**, where the solution is approached through successive approximations

The key advantage is that this transforms a constrained optimization problem into a standard regression problem, which is computationally tractable and numerically stable.

### Training Objective

The training objective is to minimize the squared difference between the current network prediction and the bootstrap target:

```math
\mathcal{L} = ||F_\theta(x_t, t) - F_\theta^-(x_t, t) - g||^2
```

This objective ensures that the network learns to predict velocities such that the consistency function remains constant along trajectories. By training $F_\theta$ to match $F_\theta^- + g$, we iteratively refine the network toward the fixed point that satisfies the consistency condition, enabling fast one-step or few-step generation from any point on the trajectory.

## Training Process

The training process for consistency models involves several key steps that implement the theoretical framework we derived. This section outlines the complete training pipeline and explains each component in detail.

### Training Pipeline Overview

The training process consists of the following steps:

1. **Sample time $t$ and generate noisy sample** $x_t$
2. **Compute velocity target** $dx_t/dt$ (different for CT vs CD)
3. **Compute network output** $F_\theta(x_t, t)$ **and JVP** $dF_\theta/dt$
4. **Compute tangent** $g$ (with progressive warmup)
5. **Normalize tangent** for numerical stability
6. **Compute loss function** with adaptive weighting
7. **Update parameters** via backpropagation

We now examine each step in detail with corresponding code implementations.

### Step 1: Time Sampling and Noisy Sample Generation

The first step involves sampling a time value $t$ from the interval $[0, \pi/2]$ and generating the corresponding noisy sample $x_t$. In practice, we sample from a log-normal distribution over the noise scale $\sigma$ and convert it to time using the trigonometric mapping.

```python
# Sample noise scale from log-normal distribution
sigma = torch.randn(x0.shape[0], device=x0.device).reshape(-1, 1, 1, 1)
sigma = (sigma * P_std + P_mean).exp()  # Log-normal: log(sigma) ~ N(P_mean, P_std^2)

# Convert sigma to time t using arctan
t = torch.arctan(sigma / sigma_data)

# Generate noise z and noisy sample x_t
z = torch.randn_like(x0) * sigma_data
x_t = torch.cos(t) * x0 + torch.sin(t) * z
```

**Why log-normal sampling?** The log-normal distribution provides better coverage across different noise levels. Since diffusion models need to learn from low noise (near data) to high noise (pure noise), uniform sampling in $t$ would under-sample the important low-noise regions. The log-normal distribution with parameters $P_{\text{mean}}$ and $P_{\text{std}}$ ensures we sample more frequently from the range where $\sigma$ is in a reasonable scale relative to the data.

**Why $t = \arctan(\sigma / \sigma_{\text{data}})$?** This mapping transforms the noise scale $\sigma$ to the time parameter $t$ in a way that naturally aligns with the trigonometric parameterization. When $\sigma = 0$, we get $t = 0$ (pure data), and as $\sigma \to \infty$, we get $t \to \pi/2$ (pure noise). The $\sigma_{\text{data}}$ parameter scales the noise relative to the data distribution's standard deviation.

### Step 2: Compute Velocity Target

The velocity target $dx_t/dt$ is computed differently depending on whether we're doing **Consistency Training (CT)** or **Consistency Distillation (CD)**.

**For Consistency Training (CT):** We use the analytical expression derived from the diffusion process:

```python
# Compute dx_t/dt analytically
dxt_dt = torch.cos(t) * z - torch.sin(t) * x0
```

This directly implements $dx_t/dt = \cos(t) \cdot z - \sin(t) \cdot x_0$, which we derived earlier from the definition $x_t = \cos(t) \cdot x_0 + \sin(t) \cdot z$.

**For Consistency Distillation (CD):** We use predictions from a pre-trained diffusion model:

```python
# Use pre-trained model to predict velocity
with torch.no_grad():
    pretrain_pred = model_pretrained(x_t / sigma_data, t)
    dxt_dt = sigma_data * pretrain_pred
```

**Why the distinction?** CT trains the model from scratch using only the data distribution, making it more principled but potentially slower to converge. CD leverages knowledge from a pre-trained diffusion model, often leading to faster training and better sample quality, but requires training a teacher model first.

### Step 3: Compute Network Output and JVP

We compute the network's velocity prediction $F_\theta(x_t, t)$ and its total derivative $dF_\theta/dt$ using Jacobian-Vector Products (JVP). The JVP efficiently computes the directional derivative along the trajectory.

```python
# Define wrapper function for JVP computation
def model_wrapper(scaled_x_t, t):
    pred, logvar = model(scaled_x_t, t.flatten(), return_logvar=True)
    return pred, logvar

# Define tangents for JVP (direction along trajectory)
v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
v_t = torch.cos(t) * torch.sin(t)

# Compute F_theta and dF_theta/dt simultaneously using JVP
F_theta, F_theta_grad, logvar = torch.func.jvp(
    model_wrapper,
    (x_t / sigma_data, t),
    (v_x, v_t),
    has_aux=True
)
logvar = logvar.view(-1, 1, 1, 1)
F_theta_grad = F_theta_grad.detach()
F_theta_minus = F_theta.detach()
```

**Why use JVP?** The JVP computes $dF_\theta/dt$ efficiently in a single forward pass, avoiding the need for multiple backward passes or explicit gradient computation. The JVP computes the derivative along the direction $(v_x, v_t)$, which corresponds to the direction of the trajectory $(dx_t/dt, dt/dt) = (dx_t/dt, 1)$.

**Why normalize $x_t$ by $\sigma_{\text{data}}$?** The network is typically trained with normalized inputs for better numerical stability and faster convergence. Dividing by $\sigma_{\text{data}}$ scales the input to have unit variance, which aligns with common normalization practices in deep learning.

**Why detach $F_\theta^-$ and $F_\theta^{\text{grad}}$?** Using stop-gradient (detached) versions creates a stable training target. The frozen network $F_\theta^-$ provides a reference prediction, and we update the current network $F_\theta$ to match $F_\theta^- + g$ rather than directly matching the tangent, which stabilizes training dynamics.

### Step 4: Compute Tangent with Progressive Warmup

The tangent $g$ is computed using the derived formula, with a progressive warmup factor $r$ that gradually introduces the second term for training stability.

```python
# Progressive warmup: gradually introduce second term
r = min(1.0, step / warmup_steps)  # warmup_steps typically 1000

# Compute tangent g
g = -torch.cos(t)**2 * (sigma_data * F_theta_minus - dxt_dt)
second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
g = g + second_term
```

This implements:

```math
\begin{align*}
g &= -\cos^2(t) \cdot (\sigma_d F_\theta^- - dx_t/dt) - r \cdot \cos(t)\sin(t) \cdot (x_t + \sigma_d \cdot F_\theta^{\text{grad}})
\end{align*}
```

**Why progressive warmup?** The factor $r$ starts at 0 and gradually increases to 1 during the first `warmup_steps` iterations. This allows the model to first focus on learning the primary velocity error term (the first part of $g$) before introducing the more complex trajectory curvature term (the second part). This progressive introduction helps prevent training instabilities in the early stages.

**Why scale by $\sigma_d$?** The $\sigma_{\text{data}}$ scaling factor ensures that the network predictions and gradients are in the correct units, accounting for the normalization of inputs and outputs.

### Step 5: Tangent Normalization

To prevent training instabilities from large tangent values, we normalize the tangent vector:

```python
# Compute normalization
g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())  # Spatial dimension invariance
g = g / (g_norm + c)  # c = 0.1 (small constant to prevent division by zero)
```

**Why normalize?** Without normalization, the tangent magnitude can vary dramatically across different samples and time steps, leading to unstable gradients. Normalization ensures consistent gradient scales.

**Why the spatial dimension correction?** The factor $\sqrt{\text{numel}(g_{\text{norm}}) / \text{numel}(g)}$ accounts for the fact that the norm is computed over spatial dimensions. This ensures the normalization is invariant to the spatial resolution of the data.

**Why add constant $c$?** The small constant $c = 0.1$ prevents division by zero when the tangent norm is very small, while also providing a minimum scale that prevents the normalized tangent from becoming too large.

### Step 6: Compute Loss Function

The loss function measures the difference between the current network prediction and the target (frozen network + tangent):

```python
# Adaptive weighting (optional)
weight = 1  # Or: weight = 1 / sigma for adaptive weighting

# Compute loss with logvar (learnable variance)
loss = (weight / torch.exp(logvar)) * torch.square(F_theta - F_theta_minus - g) + logvar
loss = loss.mean()
```

This implements:

```math
\mathcal{L} = \mathbb{E}\left[\frac{w(t)}{\exp(\text{logvar})} \cdot ||F_\theta(x_t, t) - F_\theta^-(x_t, t) - g||^2 + \text{logvar}\right]
```

**Why the weighting factor?** The weight $w(t)$ can be set to a constant (1) or made adaptive (e.g., $1/\sigma$) to adjust the importance of different noise levels. Adaptive weighting often gives more importance to lower noise levels where the data signal is stronger.

**Why include logvar?** The learnable variance parameter `logvar` allows the model to adaptively adjust the loss scale. The term $\log(\text{var}) = \text{logvar}$ in the loss encourages the model to learn appropriate uncertainty estimates, balancing the prediction error term with the variance regularization.

**Why match $F_\theta^- + g$ rather than just $g$?** The target $F_\theta^- + g$ represents the desired velocity prediction that satisfies the consistency constraint. By training $F_\theta$ to match this target, we ensure that updating the network parameters maintains consistency along trajectories.

### Step 7: Parameter Update

Finally, we perform backpropagation and update the model parameters:

```python
# Zero gradients
optimizer.zero_grad()

# Backward pass
loss.backward()

# Gradient clipping for stability
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

# Update parameters
optimizer.step()
```

**Why gradient clipping?** Gradient clipping prevents exploding gradients that can destabilize training. By limiting the gradient norm to a maximum value (typically 100.0), we ensure that parameter updates remain bounded and training proceeds smoothly.

**Common optimizer choices:** Adam or AdamW are typically used for consistency model training, with learning rates in the range $10^{-4}$ to $10^{-3}$ depending on the dataset and model size.

### Summary of Key Design Choices

1. **Log-normal time sampling:** Ensures good coverage across noise levels
2. **JVP computation:** Efficiently computes $dF_\theta/dt$ in a single forward pass
3. **Stop-gradient targets:** Stabilizes training by using frozen network predictions
4. **Progressive warmup:** Gradually introduces complex training objectives
5. **Tangent normalization:** Prevents training instabilities from large gradient magnitudes
6. **Adaptive loss weighting:** Balances learning across different noise levels
7. **Learnable variance:** Allows the model to adaptively adjust loss scales

This training pipeline ensures that the network learns to predict velocities such that the consistency function $g_\theta(x_t, t)$ remains constant along diffusion trajectories, enabling fast one-step or few-step generation during inference.
