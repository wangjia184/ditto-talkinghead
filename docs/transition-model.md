

## Transition Model [arxiv.org/abs/2509.04394](https://arxiv.org/abs/2509.04394)

TiM introduces a unified framework that generalizes diffusion and flow-matching models through time-dependent interpolation.

### Forward Interpolant

The noisy state $x_t$ at time $t$ is constructed as a linear combination of clean data $x$ and noise $\epsilon \sim \mathcal{N}(0, I)$:
 
```math
\begin{align*}
x_t &= \alpha_t x + \sigma_t \epsilon \\
&\quad \Downarrow \\
\alpha_t x &= x_t - \sigma_t \epsilon &\text{(1)}
\end{align*}
```

This allows solving for the noise component $\epsilon$, which is used in various training objectives.

### Transition Model Extension

TiM extends this by learning $F_\theta(x_t, t, r)$ for **arbitrary-step transitions** from time $t$ to $r$.

where $\alpha_t$ and $\sigma_t$ are **time-dependent coefficients** (not constants) that define the interpolation path.

### Model Prediction

The neural network output $f_\theta(x_t, t)$ can be parameterized to predict a combination of data and noise:

```math
\begin{align*}
f_{\theta}(x_t, t) &= \hat{\alpha}_t x + \hat{\sigma}_t \epsilon \\
&\quad \Downarrow \\
\epsilon &= \frac{ f_{\theta}(x_t, t) - \hat{\alpha}_t x_t}{\hat{\sigma}_t} &\text{(2)}
\end{align*}
```


First we derive $\hat{x}$, the predicted clean data:

```math
\begin{align*}
\alpha_t x &= x_t - \sigma_t \epsilon &\text{Substitute Eq.(2) into Eq.(1)} \\

\alpha_t x &= x_t - \sigma_t \frac{ f_{\theta}(x_t, t) - \hat{\alpha}_t x_t}{\hat{\sigma}_t} \\

\alpha_t x &= x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) + \frac{\sigma_t}{\hat{\sigma}_t} \hat{\alpha}_t x \\

\left( \alpha_t - \frac{\sigma_t}{\hat{\sigma}_t} \hat{\alpha}_t \right) x &= x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) \\

\left( \frac{\hat{\sigma_t}}{\hat{\sigma}_t} \alpha_t - \frac{\sigma_t}{\hat{\sigma}_t} \hat{\alpha}_t \right) x &= x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) \\

 \frac{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha_t}}{\hat{\sigma}_t}   x &= x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) \\

\hat{x} &= \frac{\hat{\sigma}_t}{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha_t}} \left(x_t - \frac{\sigma_t}{\hat{\sigma}_t}f_{\theta}(x_t, t) \right)   \\

\hat{x} &= \frac{\hat{\sigma}_t x_t - \sigma_t  f_{\theta}(x_t, t)}{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha_t}}    \\
\end{align*}

```

Next, we deduct $\hat{\epsilon}$, the predicted noise:

```math
\begin{align*}
\epsilon &= \frac{ f_{\theta}(x_t, t) - \hat{\alpha}_t x}{\hat{\sigma}_t} &\text{Substitute Eq.(1) into Eq.(2)} \\
\epsilon &= \frac{ f_{\theta}(x_t, t) - \hat{\alpha}_t \frac{x_t - \sigma_t \epsilon}{\alpha_t} }{\hat{\sigma}_t} \\
\epsilon &= \frac{ f_{\theta}(x_t, t) }{\hat{\sigma}_t} - \frac{\hat{\alpha}_t}{\hat{\sigma}_t \alpha_t} x_t + \frac{\hat{\alpha}_t \sigma_t}{\hat{\sigma}_t \alpha_t} \epsilon \\
\left( 1 - \frac{\hat{\alpha}_t \sigma_t}{\hat{\sigma}_t \alpha_t} \right) \epsilon &= \frac{ f_{\theta}(x_t, t) }{\hat{\sigma}_t} - \frac{\hat{\alpha}_t}{\hat{\sigma}_t \alpha_t} x_t \\
\left( \hat{\sigma}_t \alpha_t - \hat{\alpha}_t \sigma_t \right) \epsilon &= \alpha_t f_{\theta}(x_t, t) - \hat{\alpha}_t x_t \\
\hat{\epsilon} &= \frac{ \alpha_t f_{\theta}(x_t, t) - \hat{\alpha}_t x_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t }
\end{align*}
```

We now introduce the target timestep `r` and express $x_r$ via the interpolant using the predicted clean data  $\hat{x}$ and predicted noise $\hat{ϵ}$ : 

```math
\begin{align*}
x_r &= \alpha_r \hat{x} + \sigma_r \hat{\epsilon} \\

x_r &= \alpha_r \left( \frac{\hat{\sigma}_t x_t - \sigma_t  f_{\theta}(x_t, t)}{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha_t}}  \right) + \sigma_r \left( \frac{ \alpha_t f_{\theta}(x_t, t) - \hat{\alpha}_t x_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) \\

x_r &= \frac{\hat{\sigma}_t \alpha_r x_t - \sigma_t \alpha_r f_{\theta}(x_t, t)}{\hat{\sigma_t} \alpha_t - \sigma_t \hat{\alpha}_t} + \frac{\sigma_r \alpha_t f_{\theta}(x_t, t) - \sigma_r \hat{\alpha}_t x_t}{\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t} \\

x_r &= \frac{ \left( \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t \right) x_t + \left( \sigma_r \alpha_t - \sigma_t \alpha_r \right) f_{\theta}(x_t, t) }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \\


x_r &= \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } x_t + \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } f_{\theta}(x_t, t) \\


\end{align*}
```

More generally, to learn the `t → r` transition, we condition the predictor on `r` and obtain:


```math
\begin{align*}
x_r &= \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } x_t + \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } f_{\theta}(x_t,t,r)   &\text{(3)}
\end{align*}
```

For simplicity, we define the coefficients:

```math
\begin{align*}

A_{t,r} &:= \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \\

B_{t,r} &:= \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \\

f_{\theta,t,r} &:= f_{\theta}(x_t,t,r)

\end{align*}
```

Then the transition can be written as:

```math
\begin{align*}
x_r &= A_{t,r} x_t + B_{t,r} f_{\theta,t,r} &\text{(4)}
\end{align*}
```

We now differentiate both sides of Eq.(4) with respect to t . Since $x_r$ is independent of `t` , its derivative is zero. Applying the product rule and chain rule yields:


```math
\begin{align*}
\frac{d A_{t,r}}{dt} &= 
\frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt} \\
&&\text{(5)} \\ 
\frac{d B_{t,r}}{dt} &= 
\frac{\partial B_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\frac{\partial B_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
\frac{\partial B_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
\frac{\partial B_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}

\end{align*}
```

Define $C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t$. We compute each partial derivative in Eq. (5) explicitly:

```math
\begin{aligned}
&\begin{cases}
\displaystyle
\frac{\partial A_{t,r}}{\partial \alpha_t} 
&= \frac{\partial}{\partial \alpha_t} \left( \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ 0 \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) \cdot \hat{\sigma}_t }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= -\frac{ \hat{\sigma}_t (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= - \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial A_{t,r}}{\partial \sigma_t} 
&= \frac{\partial}{\partial \sigma_t} \left( \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ 0 \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) \cdot (-\hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \hat{\alpha}_t (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} 
&= \frac{\partial}{\partial \hat{\alpha}_t} \left( \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ (-\sigma_r) \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) \cdot (-\sigma_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \hat{\sigma}_t (\sigma_t \alpha_r - \sigma_r \alpha_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= - \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} 
&= \frac{\partial}{\partial \hat{\sigma}_t} \left( \frac{ \hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ \alpha_r \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) \cdot \alpha_t }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \hat{\alpha}_t (\sigma_r \alpha_t - \sigma_t \alpha_r) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} }
\end{cases} 
\\
\\
&\begin{cases}
\displaystyle
\frac{\partial B_{t,r}}{\partial \alpha_t} 
&= \frac{\partial}{\partial \alpha_t} \left( \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ \sigma_r \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\sigma_r \alpha_t - \sigma_t \alpha_r) \cdot \hat{\sigma}_t }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \sigma_t (\hat{\sigma}_t \alpha_r - \sigma_r \hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \sigma_t \frac{ A_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial B_{t,r}}{\partial \sigma_t} 
&= \frac{\partial}{\partial \sigma_t} \left( \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ (-\alpha_r) \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\sigma_r \alpha_t - \sigma_t \alpha_r) \cdot (-\hat{\alpha}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \alpha_t (\sigma_r \hat{\alpha}_t - \alpha_r \hat{\sigma}_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= - \alpha_t \frac{ A_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial B_{t,r}}{\partial \hat{\alpha}_t} 
&= \frac{\partial}{\partial \hat{\alpha}_t} \left( \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ 0 \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\sigma_r \alpha_t - \sigma_t \alpha_r) \cdot (-\sigma_t) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \frac{ \sigma_t (\sigma_r \alpha_t - \sigma_t \alpha_r) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= \sigma_t \frac{ B_{t,r} }{ C_{t,r} } \\ \\
\displaystyle
\frac{\partial B_{t,r}}{\partial \hat{\sigma}_t} 
&= \frac{\partial}{\partial \hat{\sigma}_t} \left( \frac{ \sigma_r \alpha_t - \sigma_t \alpha_r }{ \hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t } \right) 
&&= \frac{ 0 \cdot (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t) - (\sigma_r \alpha_t - \sigma_t \alpha_r) \cdot \alpha_t }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= -\frac{ \alpha_t (\sigma_r \alpha_t - \sigma_t \alpha_r) }{ (\hat{\sigma}_t \alpha_t - \sigma_t \hat{\alpha}_t)^2 } 
&&= - \alpha_t \frac{ B_{t,r} }{ C_{t,r} }
\end{cases} 
\end{aligned}
```

Substituting Eq.(6) into Eq.(5) gives the total time derivatives:

```math
\begin{equation}
\begin{aligned}
\frac{d A_{t,r}}{dt} &= 
-\hat{\sigma}_t \frac{A_{t,r}}{C_{t,r}} \cdot \frac{d\alpha_t}{dt} &+ 
\hat{\alpha}_t \frac{A_{t,r}}{C_{t,r}} \cdot \frac{d\sigma_t}{dt} &- 
\hat{\sigma}_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\alpha}_t}{dt} &+ 
\hat{\alpha}_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt} \\[0.6em]
\frac{d B_{t,r}}{dt} &= 
\sigma_t \frac{A_{t,r}}{C_{t,r}} \cdot \frac{d\alpha_t}{dt} &- 
\alpha_t \frac{A_{t,r}}{C_{t,r}} \cdot \frac{d\sigma_t}{dt} &+ 
\sigma_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\alpha}_t}{dt} &- 
\alpha_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt}
\end{aligned}
\tag{7}
\end{equation}
```

From the interpolant equation $x_t=\alpha_t x + \sigma_t \epsilon$, we compute its time derivative:

```math
\begin{align*}
x_t &= \alpha_t x + \sigma_t \epsilon \\
&\quad \Downarrow \\
\frac{dx_t}{dt} &= \frac{d \alpha_t}{dt} x +  \frac{d \sigma_t}{dt} \epsilon \tag{8}
\end{align*}
```

We now differentiate Eq.(4) with respect to `t` :

```math
\begin{align*}
A_{t,r} x_t + B_{t,r} f_{\theta,t,r} &= x_r \\

\frac{d}{dt} \left( A_{t,r} x_t + B_{t,r} f_{\theta,t,r} \right) &= \frac{d x_r}{dt} \\

\frac{d}{dt} \left( A_{t,r} x_t \right) + \frac{d}{dt} \left( B_{t,r} f_{\theta,t,r} \right) &= 0  \\
 
\left[ \frac{d A_{t,r}}{dt} x_t + \frac{d x_t}{dt} A_{t,r} \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 \\

\left[ \frac{d A_{t,r}}{dt} \left(\alpha_t x + \sigma_t \epsilon \right) + \frac{d x_t}{dt} A_{t,r} \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right] 
&= 0 &\text{(since $x_t = \alpha_t x + \sigma_t \epsilon$)} \\

\left[ \frac{d A_{t,r}}{dt} \left(\alpha_t x + \sigma_t \epsilon \right) + \left( \frac{d \alpha_t}{dt} x +  \frac{d \sigma_t}{dt} \epsilon \right) A_{t,r} \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right] 
&= 0 &\text{(since Eq.(8))} \\

\left[ \frac{d A_{t,r}}{dt} \alpha_t x + \frac{d A_{t,r}}{dt} \sigma_t \epsilon  +A_{t,r} \frac{d \alpha_t}{dt} x +  A_{t,r} \frac{d \sigma_t}{dt} \epsilon   \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\

\left[ \left( A_{t,r} \frac{d \alpha_t}{dt} + \alpha_t \frac{d A_{t,r}}{dt}    \right) x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\


\left[ \left( A_{t,r} \frac{d \alpha_t}{dt} + \alpha_t 
\left(
\frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    \right) x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  &\text{(since Eq.(5))} \\


\left[ \left( 
  \left( A_{t,r}  + \alpha_t \frac{\partial A_{t,r}}{\partial \alpha_t}  \right) \frac{d \alpha_t}{dt} +
 \alpha_t \frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
 \alpha_t \frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t \frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\


\left[ \left( 
  \left( A_{t,r}  - \alpha_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} }  \right) \frac{d \alpha_t}{dt} +
 \alpha_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t  \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  &\text{(since Eq.(6))} \\

\left[ \left( 
  \left( A_{t,r} \frac{\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t}{C_{t,r}} - \alpha_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} }  \right) \frac{d \alpha_t}{dt} +
 \alpha_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t  \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  &\text{(since $C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t$)} \\

\left[ \left( 
   - \sigma_t  \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} }  \cdot \frac{d \alpha_t}{dt} +
 \alpha_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t  \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\


\left[ \left( 
   - \sigma_t  \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} }  \cdot \frac{d \alpha_t}{dt} +
 \alpha_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \alpha_t  \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt} +
 \hat{\alpha}_t \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}-
 \hat{\alpha}_t \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\

\left[ \left( 
  - \hat{\alpha}_t
  \left(
    \sigma_t   \frac{ A_{t,r} }{ C_{t,r} }  \cdot \frac{d \alpha_t}{dt} 
    - \alpha_t  \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} 
    + \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
    - \alpha_t  \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
  \right)
 -
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} +
 \hat{\alpha}_t \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0   \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
-
 \alpha_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} +
 \hat{\alpha}_t \sigma_t \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{(since Eq.(7))}  \\

\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - \left(
    \alpha_t  \hat{\sigma}_t - \hat{\alpha}_t \sigma_t
  \right)
   \frac{ B_{t,r}}{C_{t,r}} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t \frac{d A_{t,r}}{dt}   \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{(since $C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t$)} \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} + \sigma_t 
\left(
\frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
\frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
\right)     \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{(since Eq.(5))} \\



\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left(  A_{t,r} \frac{d \sigma_t}{dt} +
 \sigma_t \frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \sigma_t} \cdot \frac{d \sigma_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
 \sigma_t \frac{\partial A_{t,r}}{\partial \alpha_t} \cdot \frac{d \alpha_t}{dt} + 
\left(  A_{t,r} + \sigma_t \frac{\partial A_{t,r}}{\partial \sigma_t} \right) \cdot \frac{d \sigma_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \hat{\alpha}_t} \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \frac{\partial A_{t,r}}{\partial \hat{\sigma}_t} \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\
 
\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \sigma_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} + 
\left(  A_{t,r} + \sigma_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \right) \cdot \frac{d \sigma_t}{dt} -
 \sigma_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{(since Eq.(6))} \\
 
\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \sigma_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} + 
\left(  A_{t,r} \frac{\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t}{C_{t,r}} + \sigma_t \hat{\alpha}_t \frac{ A_{t,r} }{ C_{t,r} } \right) \cdot \frac{d \sigma_t}{dt} -
 \sigma_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{(since $C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t$)} \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \sigma_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} + 
 \hat{\sigma}_t\alpha_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} -
 \sigma_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} + 
 \sigma_t \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
- \left( 
 \sigma_t  \hat{\sigma}_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} - 
 \hat{\sigma}_t\alpha_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} +
 \sigma_t  \hat{\sigma}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} - 
 \sigma_t \hat{\alpha}_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}
 - \hat{\sigma}_t \alpha_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt}
+ \hat{\sigma}_t \alpha_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt}
      \right)  \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\
 
\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \hat{\sigma}_t \left( 
 \sigma_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \alpha_t}{dt} - 
 \alpha_t \frac{ A_{t,r} }{ C_{t,r} } \cdot \frac{d \sigma_t}{dt} +
 \sigma_t \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\alpha}_t}{dt} -
 \alpha_t \frac{B_{t,r}}{C_{t,r}} \cdot \frac{d\hat{\sigma}_t}{dt}
 \right)
 -
 \left(  
  \left( \sigma_t \hat{\alpha}_t - \hat{\sigma}_t \alpha_t \right)
  \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}

      \right) \right) \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0  \\



\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \hat{\sigma}_t \frac{d B_{t,r}}{dt}
 -
 \left(  
  \left( \sigma_t \hat{\alpha}_t - \hat{\sigma}_t \alpha_t \right)
  \frac{ B_{t,r} }{ C_{t,r} } \cdot \frac{d \hat{\sigma}_t}{dt}

      \right) \right) \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{(since Eq.(7))}\\


\left[ \left( 
  - \hat{\alpha}_t
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r} \cdot \frac{ d \hat{\alpha}_t} {dt}
\right)    x 
+ \left( 
- \hat{\sigma}_t \frac{d B_{t,r}}{dt}
 -
    B_{t,r}  \cdot \frac{d \hat{\sigma}_t}{dt}
 \right) \epsilon  \right] + 
\left[ \frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} \right]
&= 0 &\text{(since $C_{t,r}:=\hat{\sigma}_t\alpha_t - \hat{\alpha}_t \sigma_t$)} \\

 
  - \hat{\alpha}_t  x 
  \frac{d B_{t,r}}{dt}
  - 
     B_{t,r}  x  \cdot \frac{ d \hat{\alpha}_t} {dt}
- \hat{\sigma}_t \epsilon \frac{d B_{t,r}}{dt}
 -
    B_{t,r} \epsilon \cdot \frac{d \hat{\sigma}_t}{dt}
   + 
\frac{d B_{t,r}}{dt}  f_{\theta,t,r} + \frac{d f_{\theta,t,r}}{dt} B_{t,r} 
&= 0 \\

\hat{\alpha}_t  x 
  \frac{d B_{t,r}}{dt}
+ B_{t,r}  x  \cdot \frac{ d \hat{\alpha}_t} {dt}
+ \hat{\sigma}_t \epsilon \frac{d B_{t,r}}{dt}
+  B_{t,r} \epsilon \cdot \frac{d \hat{\sigma}_t}{dt}
-\frac{d B_{t,r}}{dt}  f_{\theta,t,r} 
- \frac{d f_{\theta,t,r}}{dt} B_{t,r} 
&= 0 \\

\left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right)
 \frac{d B_{t,r}}{dt} 
+ B_{t,r}
\left(
x  \cdot \frac{ d \hat{\alpha}_t} {dt}
+   \epsilon \cdot \frac{d \hat{\sigma}_t}{dt}
- \frac{d f_{\theta,t,r}}{dt}  
\right)
&= 0 \\


\left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right)
 \frac{d B_{t,r}}{dt} 
+ B_{t,r}
\left(
 \frac{ d \hat{\alpha}_t x } {dt}
+    \frac{d \hat{\sigma}_t  \epsilon}{dt}
- \frac{d f_{\theta,t,r}}{dt}  
\right)
&= 0 \\

\left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right)
 \frac{d B_{t,r}}{dt} 
+ B_{t,r}
 \frac{ d\left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right) } {dt}
&= 0 &\text{(9)} \\

\frac{d}{dt}
\left[
  B_{t,r} \cdot \left(\hat{\alpha}_t  x + \hat{\sigma}_t \epsilon -  f_{\theta,t,r}  \right)
\right]
&= 0 &\text{(10)} \\

\end{align*}
```

This is the **TiM Identity Equation**, which states that the quantity $B_{t,r} \cdot (\hat{\alpha}_t x + \hat{\sigma}_t \epsilon - f_{\theta,t,r})$ must be constant along the trajectory. This identity provides the theoretical foundation for training transition models.

## From Identity Equation to Training Objective

The TiM Identity Equation (10) provides a constraint that must be satisfied for any valid transition model. To convert this into a training objective, we need to express it in terms of the model's prediction $f_\theta(x_t, t, r)$ and derive a target value.

### Deriving the Target

From the identity equation, we know that:

```math
B_{t,r} \cdot (\hat{\alpha}_t x + \hat{\sigma}_t \epsilon - f_{\theta,t,r}) = \text{constant}
```

If the transition model is exact, the **transport residual**
```math
\hat{\alpha}_t x + \hat{\sigma}_t \epsilon - f_{\theta,t,r}
```
vanishes, implying the identity equation holds for all $t$.

However, for training, we need a target that can be computed from the current state $(x_t, t)$ and the target time $r$. The key insight is to use the analytical expression for the velocity $v_t$ and the derivative $dF_\theta/dt$ to construct the target.

Rather than yielding an explicit target, the identity equation defines a **transport constraint**. The training objective is obtained by rewriting this constraint into an equivalent squared residual form. Specifically, we use the fact that:

```math
x_r = A_{t,r} x_t + B_{t,r} f_{\theta,t,r}
```

and the velocity relationship:

```math
v_t = \frac{d\alpha_t}{dt} x + \frac{d\sigma_t}{dt} \epsilon
```

Different transports induce different **algebraic transport residuals**, whose normalized forms lead to different training losses. In practice, the loss used in code corresponds to a normalized algebraic residual of the identity equation, which is **algebraically equivalent** to an MSE loss against an implicit target $F_{\text{target}}$. The remainder of this subsection fills in the **missing algebra** for the TrigFlow target, mirroring the style of the TrigFlow derivation in `docs/trig-flow.md` but now for the **two-time transition** setting $(t \to r)$.

### Detailed derivation of the TrigFlow target

This subsection is written to make the appearance of $\tan(t-r)$ **completely non-mysterious** and **exact** (no Taylor expansion). We proceed in two stages:

1. Start from the **TrigFlow transport equation** and **algebraically invert** it to solve for the transition field.
2. (Optional but used in the provided code) convert that exact transport constraint into a **local regression target** that depends on $v_t$ and a directional derivative term.

#### Stage A — exact transport inversion (where $\tan(t-r)$ comes from)

For TrigFlow, the transport between times $t$ and $r$ is defined as a rotation-like map:

```math
\boxed{
x_r
=
\cos(\Delta)\,x_t
-
\sin(\Delta)\,F_{t,r},
\qquad
\Delta := t-r
}
```

In training, $x_t$ and $x_r$ are both computable from the same underlying sample $(x,\epsilon)$ via the interpolant (for TrigFlow, $\alpha_t=\cos t$, $\sigma_t=\sin t$):

```math
x_t = \alpha_t x + \sigma_t \epsilon,
\qquad
x_r = \alpha_r x + \sigma_r \epsilon.
```

Since the only unknown in the transport equation is $F_{t,r}$, the supervision signal is uniquely determined by **solving for $F_{t,r}$**:

```math
\begin{align*}
x_r
&= \cos(\Delta)\,x_t - \sin(\Delta)\,F_{t,r} \\
\sin(\Delta)\,F_{t,r}
&= \cos(\Delta)\,x_t - x_r \\
F_{t,r}
&=
\frac{\cos(\Delta)\,x_t - x_r}{\sin(\Delta)}
\end{align*}
```

```math
\boxed{
F_{t,r}
=
\frac{\cos(\Delta)\,x_t - x_r}{\sin(\Delta)}
}
```

Substitute $x_t=\alpha_t x+\sigma_t\epsilon$ and $x_r=\alpha_r x+\sigma_r\epsilon$ to get the same target expressed in the $(x,\epsilon)$ basis:

```math
\boxed{
F_{\text{target}}
=
\left(
\frac{\cos(\Delta)\alpha_t - \alpha_r}{\sin(\Delta)}
\right)x
+
\left(
\frac{\cos(\Delta)\sigma_t - \sigma_r}{\sin(\Delta)}
\right)\epsilon
}
```

Now we re-group terms to factor out $x_t$ and $x_r$.
Starting from the exact form expressed in the $(x,\epsilon)$ basis (equation above),

```math
\begin{align*}
F_{\text{target}}
&=
\left(
\frac{\cos(\Delta)\alpha_t - \alpha_r}{\sin(\Delta)}
\right)x
+
\left(
\frac{\cos(\Delta)\sigma_t - \sigma_r}{\sin(\Delta)}
\right)\epsilon \\
&=
\frac{\cos(\Delta)\alpha_t}{\sin(\Delta)}x
-
\frac{\alpha_r}{\sin(\Delta)}x
+
\frac{\cos(\Delta)\sigma_t}{\sin(\Delta)}\epsilon
-
\frac{\sigma_r}{\sin(\Delta)}\epsilon \\
&=
\frac{\cos(\Delta)}{\sin(\Delta)}\left(\alpha_t x + \sigma_t \epsilon\right)
-
\frac{1}{\sin(\Delta)}\left(\alpha_r x + \sigma_r \epsilon\right) \\
&=
\frac{\cos(\Delta)}{\sin(\Delta)}x_t
-
\frac{1}{\sin(\Delta)}x_r \\
&=
\frac{\cos(\Delta)\,x_t - x_r}{\sin(\Delta)}
\end{align*}
```

```math
\boxed{
F_{\text{target}}
=
\frac{\cos(\Delta)\,x_t - x_r}{\sin(\Delta)}
}
```

This is the **exact supervision signal** obtained by algebraically inverting the transport equation. 

#### Stage B — eliminating $x_r$ via the velocity identity

The exact supervision signal derived above depends on $x_r$, which is not a function of the network input $(x_t,t)$ and therefore must be eliminated to form a local regression target.
To eliminate $x_r$ without changing the supervision semantics, we use a **crucial trigonometric identity**.

From the interpolant $x_t = \cos t \, x + \sin t \, \epsilon$, differentiate with respect to time:

```math
v_t := \frac{dx_t}{dt} = -\sin t \, x + \cos t \, \epsilon
```

This is the *true velocity field*. Now observe the exact trigonometric identity:

```math
\boxed{
x_r = \cos(\Delta)\,x_t - \sin(\Delta)\,v_t
}
```

This identity is **exact**, not an approximation. It follows from standard angle-difference formulas applied to the interpolant.

Substitute this identity into the exact supervision signal:

```math
\begin{align*}
F_{\text{target}}
&=
\frac{\cos(\Delta)\,x_t - x_r}{\sin(\Delta)} \\
&=
\frac{\cos(\Delta)\,x_t - \left[\cos(\Delta)\,x_t - \sin(\Delta)\,v_t\right]}{\sin(\Delta)} \\
&=
\frac{\sin(\Delta)\,v_t}{\sin(\Delta)} \\
&=
v_t
\end{align*}
```

This shows that under **exact transport consistency**, $F_\theta(x_t,t,r) \equiv v_t$.

However, the code does not regress $F_\theta$ directly to $v_t$. Instead, it enforces the transport equation **in residual form**:

```math
x_r - \cos(\Delta)\,x_t + \sin(\Delta)\,F_\theta = 0
```

Divide by $\cos(\Delta)$:

```math
\frac{x_r}{\cos(\Delta)} - x_t + \tan(\Delta)\,F_\theta = 0
```

Rearrange to obtain the form used in the reference code:

```math
\boxed{
F_{\text{target}} = \frac{x_t - x_r/\cos(\Delta)}{\tan(\Delta)}
}
```


### Training Objective

Semantically, the training objective is to minimize the squared difference between the model prediction and the exact target:

```math
\mathcal{L}
=
\mathbb{E}\left[
\left\|
F_\theta(x_t,t,r)
-
F_{\text{target}}
\right\|^2
\right]
```

where $F_{\text{target}} = \frac{\cos(\Delta)x_t - x_r}{\sin(\Delta)}$ is the exact supervision signal derived in Stage A.

However, TrigFlow does not explicitly construct this target. Instead, it uses an **algebraically equivalent residual form** obtained by rewriting the difference $F_\theta - F_{\text{target}}$:

```math
\begin{align*}
F_\theta - F_{\text{target}}
&=
F_\theta - \frac{\cos(\Delta)x_t - x_r}{\sin(\Delta)} \\
&=
\frac{\sin(\Delta)F_\theta - \cos(\Delta)x_t + x_r}{\sin(\Delta)} \\
&=
\frac{x_r - \cos(\Delta)x_t + \sin(\Delta)F_\theta}{\sin(\Delta)}
\end{align*}
```

Since $\sin^2(\Delta)$ is a function of $(t,r)$ only and acts as a weight, minimizing $\|F_\theta - F_{\text{target}}\|^2$ is equivalent to minimizing the **transport equation residual**:

```math
\left\|
x_r - \cos(\Delta)x_t + \sin(\Delta)F_\theta
\right\|^2
```

Normalize by dividing the residual by $\cos(\Delta)$ (for numerical stability) and rearranging:

```math
\begin{align*}
\frac{x_r - \cos(\Delta)x_t + \sin(\Delta)F_\theta}{\cos(\Delta)}
&=
\frac{x_r}{\cos(\Delta)} - x_t + \tan(\Delta)F_\theta \\
&=
-\left(x_t - \frac{x_r}{\cos(\Delta)} + \tan(\Delta)F_\theta\right)
\end{align*}
```

Since multiplying by $-1$ does not change the squared norm, the final loss used in code is:

```math
\boxed{
\mathcal{L}
=
\mathbb{E}\left[
w(t, r) \cdot
\left\|
x_t
-
\frac{x_r}{\cos(t-r)}
+
\tan(t-r)\,F_\theta(x_t,t,r)
\right\|^2
\right]
}
```

where $w(t, r)$ is a time-dependent weighting function that balances the importance of different time intervals and transition lengths.

**Key insight:** All forms are **algebraically equivalent**. The code minimizes the transport equation residual, which is equivalent to regressing $F_\theta$ toward the exact target $F_{\text{target}}$, but expressed in a form that avoids explicit construction of the target during the forward pass.

**Implementation:** In code, the residual is computed as:

```python
delta = t - r
residual = (
    x_t
    - x_r / torch.cos(delta)
    + torch.tan(delta) * F_theta
)
loss = (residual ** 2).mean()
```

This directly implements the normalized transport consistency residual, where $x_r$ is computed from the interpolant using the same $(x,\epsilon)$ sample.

## Training Process

The training process for TiM involves several key steps that implement the theoretical framework derived above. This section outlines the complete training pipeline and explains each component in detail.

### Training Pipeline Overview

The training process consists of the following steps:

1. **Sample time pairs** $(t, r)$ and generate noisy sample $x_t$
2. **Compute velocity** $v_t$ from the interpolant
3. **Forward pass** to get model prediction $F_\theta(x_t, t, r)$
4. **Compute derivative** $dF_\theta/dt$ using JVP or finite differences
5. **Compute target** $F_{\text{target}}$ using the identity equation
6. **Compute loss** with time-dependent weighting
7. **Update parameters** via backpropagation

We now examine each step in detail with corresponding code implementations.

### Step 1: Time Sampling and Noisy Sample Generation

The first step involves sampling a pair of time values $(t, r)$ where $t \geq r$, and generating the corresponding noisy sample $x_t$. The time $t$ represents the current state, and $r$ represents the target state we want to transition to.

```python
# Sample two time values independently
t_1 = transport.sample_t(batch_size=batch_size, dtype=x.dtype, device=x.device)
t_2 = transport.sample_t(batch_size=batch_size, dtype=x.dtype, device=x.device)

# t is the larger one, r is the smaller one
t = torch.maximum(t_1, t_2)
r = torch.minimum(t_1, t_2)

# Some samples with t=r for diffusion training
n_diffusion = round(diffusion_ratio * len(t))
r[:n_diffusion] = t[:n_diffusion]

# Some samples with r=0 for consistency training
n_consistency = round(consistency_ratio * len(t))
if n_consistency != 0:
    r[-n_consistency:] = transport.T_min

# Reshape for broadcasting
t, r = expand_t_like_x(t, x), expand_t_like_x(r, x)

# Generate noisy sample and velocity
alpha_t, sigma_t, d_alpha_t, d_sigma_t = transport.interpolant(t)
x_t = alpha_t * x + sigma_t * z
v_t = d_alpha_t * x + d_sigma_t * z
```

**Why sample two times?** Sampling two independent times and taking the maximum/minimum ensures we get a diverse set of transition lengths $(t - r)$. This allows the model to learn transitions of varying sizes, from small refinements (when $t \approx r$) to large jumps (when $t \gg r$).

**Why set some $r = t$?** When $r = t$, the transition length is zero, which corresponds to standard diffusion training. This ensures compatibility with existing diffusion model training objectives.

**Why set some $r = 0$?** When $r = 0$, we're transitioning directly to the data distribution, which corresponds to consistency model training. This helps the model learn direct data prediction.

**Why log-normal sampling?** The time sampling uses a log-normal distribution over the noise scale $\sigma$, which provides better coverage across different noise levels. This is similar to EDM and ensures we sample more frequently from important noise ranges.

### Step 2: Compute Velocity

The velocity $v_t$ is computed analytically from the interpolant:

```python
alpha_t, sigma_t, d_alpha_t, d_sigma_t = transport.interpolant(t)
v_t = d_alpha_t * x + d_sigma_t * z
```

This directly implements $v_t = \frac{d\alpha_t}{dt} x + \frac{d\sigma_t}{dt} \epsilon$, which is the time derivative of the interpolant $x_t = \alpha_t x + \sigma_t \epsilon$.

**Why compute velocity analytically?** The velocity can be computed exactly from the known interpolant functions $\alpha_t$ and $\sigma_t$, without needing to approximate derivatives. This provides a clean training signal.

### Step 3: Forward Pass

We perform a forward pass through the model to get the prediction $F_\theta(x_t, t, r)$:

```python
# Prepare time inputs (may involve noise conditioning)
t_input = transport.c_noise(t.flatten())
r_input = transport.c_noise(r.flatten())

# Forward pass
model_output = model(x_t, t_input, r_input, **model_kwargs)
F_pred = model_output  # Model predicts the transition function
```

**Why condition on both $t$ and $r$?** The model needs to know both the current time $t$ and the target time $r$ to predict the appropriate transition. This allows the model to adapt its prediction based on the transition length $(t - r)$.

**What does $c_{\text{noise}}$ do?** The function $c_{\text{noise}}$ transforms the time variable into a noise conditioning format that the model expects. Different transport formulations (EDM, TrigFlow, OT-FM) use different transformations.

### Step 4: Compute Derivative

The derivative $\frac{dF_\theta}{dt}$ is computed using either Jacobian-Vector Products (JVP) or finite differences (DDE - Differential Derivative Estimation):

**Using JVP (Jacobian-Vector Product):**

```python
def model_jvp(x_t, t, r):
    model_kwargs['derivative'] = True
    t_input = transport.c_noise(t.flatten())
    r_input = transport.c_noise(r.flatten())
    return model(x_t, t_input, r_input, **model_kwargs)

# Compute JVP along the trajectory direction
F_pred, dF_dv_dt = torch.func.jvp(
    lambda x_t, t, r: model_jvp(x_t, t, r),
    (x_t, t, r),
    (v_t, torch.ones_like(t), torch.zeros_like(r))
)
```

**Using Finite Differences (DDE):**

```python
epsilon = differential_epsilon  # Typically 0.005

def xfunc(t):
    alpha_t, sigma_t, _, _ = transport.interpolant(t)
    x_t = alpha_t * x + sigma_t * z
    return model_forward(model, x_t, t, r, model_kwargs, rng_state)

# Central difference approximation
fc1_dt = 1 / (2 * epsilon)
dF_dv_dt = xfunc(t + epsilon) * fc1_dt - xfunc(t - epsilon) * fc1_dt
```

**Why use JVP?** JVP computes the derivative efficiently in a single forward pass, avoiding the need for multiple model evaluations. It computes the directional derivative along the trajectory direction $(v_t, 1, 0)$.

**Why use finite differences?** Finite differences (DDE) can be more stable in some cases and doesn't require automatic differentiation support. However, it requires two additional forward passes and may be less accurate.

**Why only compute when $t \neq r$?** When $t = r$, the transition length is zero, so the derivative term $(t - r) \cdot dF_\theta/dt$ vanishes. We skip the expensive derivative computation for these cases.

### Step 5: Compute Target

The target $F_{\text{target}}$ is computed using the identity equation:

```python
# For OT-FM (Flow Matching) transport
F_target = v_t - (t - r) * dF_dv_dt

# For TrigFlow transport
F_target = v_t - torch.tan(t - r) * (x_t + dF_dv_dt)

# For EDM transport (more complex, involves Bt_dv_dBt coefficient)
# ... (see transports.py for full implementation)
```

**Why subtract $(t - r) \cdot dF_\theta/dt$?** This term comes from the identity equation and accounts for how the model prediction changes along the trajectory. It ensures that the model learns to predict transitions that satisfy the identity constraint.

**Why different formulas for different transports?** Different transport formulations (linear, trigonometric, EDM) have different interpolant functions, leading to different forms of the identity equation. The target formula is derived specifically for each transport type.

### Step 6: Compute Loss Function

The loss function measures the difference between the model prediction and the target:

```python
# Basic denoising loss
denoising_loss = mean_flat((F_pred - F_target) ** 2)
denoising_loss = torch.nan_to_num(denoising_loss, nan=0, posinf=1e5, neginf=-1e5)

# Optional: Add directional loss
if use_dir_loss:
    directional_loss = mean_flat(1 - F.cosine_similarity(F_pred, F_target, dim=1))
    directional_loss = torch.nan_to_num(directional_loss, nan=0, posinf=1e5, neginf=-1e5)
    denoising_loss += directional_loss

# Time-dependent weighting
weight = time_weighting(t, r, n_diffusion) * adaptive_weighting(denoising_loss)
weighted_loss = weight * denoising_loss
weighted_loss = weighted_loss.mean()
```

**Time Weighting:** The time weighting function $w(t, r)$ adjusts the importance of different time intervals:

```python
def time_weighting(t, r, n_diffusion):
    delta_t = (t - r).flatten()  # Transition length
    
    if weight_time_type == 'constant':
        weight = torch.ones_like(delta_t)
    elif weight_time_type == 'reciprocal':
        weight = 1 / (delta_t + sigma_d)
    elif weight_time_type == 'sqrt':
        weight = 1 / (delta_t + sigma_d).sqrt()
    elif weight_time_type == 'square':
        weight = 1 / (delta_t + sigma_d)**2
    elif weight_time_type == 'Soft-Min-SNR':
        weight = 1 / (delta_t ** 2 + sigma_d ** 2)
    
    # Diffusion samples (t=r) always have weight 1
    weight[:n_diffusion] = 1.0
    return weight
```

**Why time weighting?** Different transition lengths $(t - r)$ may have different importance for learning. For example, very short transitions might be more critical for high-quality generation, while very long transitions might be less reliable. Time weighting allows us to emphasize the most important transitions.

**Why adaptive weighting?** Adaptive weighting uses the inverse of the loss magnitude to automatically balance easy and hard samples:

```python
def adaptive_weighting(loss, eps=10e-6):
    weight = 1 / (loss.detach() + eps)
    return weight
```

This gives more weight to samples where the model is making progress (lower loss) and less weight to outliers or very difficult samples.

**Why directional loss?** The directional loss encourages the model to predict in the correct direction, not just minimize the magnitude of the error. This can be especially helpful when the prediction and target have similar magnitudes but point in different directions.

### Step 7: Parameter Update

Finally, we perform backpropagation and update the model parameters:

```python
# Zero gradients
optimizer.zero_grad()

# Backward pass
loss.backward()

# Gradient clipping for stability
if max_grad_norm > 0:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# Update parameters
optimizer.step()
lr_scheduler.step()
```

**Why gradient clipping?** Gradient clipping prevents exploding gradients that can destabilize training, especially when computing derivatives through the model.

### Summary of Key Design Choices

1. **Dual time sampling:** Samples both $t$ and $r$ to learn transitions of varying lengths
2. **Mixed training:** Combines diffusion ($t = r$), consistency ($r = 0$), and transition ($t > r$) objectives
3. **Efficient derivative computation:** Uses JVP for efficient gradient computation or finite differences for stability
4. **Time-dependent weighting:** Balances the importance of different transition lengths
5. **Adaptive weighting:** Automatically adjusts sample importance based on loss magnitude
6. **Directional loss:** Encourages correct prediction direction, not just magnitude

This training pipeline ensures that the network learns to predict transitions that satisfy the TiM Identity Equation, enabling flexible generation with arbitrary numbers of steps, from single-step generation to fine-grained multi-step refinement.

