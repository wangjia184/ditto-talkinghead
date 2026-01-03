# Ditto 训练损失函数分析

## 损失函数组成

根据代码分析，Ditto 模型的损失函数由以下部分组成：

### 基础架构
- **Diffusion Loss**: MSE (L2) 损失，用于扩散模型的去噪训练
- **PVA Loss**: 位置(Position)、速度(Velocity)、加速度(Acceleration)的分段加权损失
  - 对运动特征的不同部分（scale、pitch、yaw、roll、translation、expression）分别计算

### 可选损失项
1. **Last Frame Loss** (`--use_last_frame_loss`)
   - 用于约束第一帧（初始条件帧）的预测
   - 确保生成的序列从给定的条件帧开始

2. **Regularization Loss** (`--use_reg_loss`)
   - L1 正则化，对运动特征值的大小进行约束
   - 防止生成过大的不切实际的运动

## 当前 train.sh 的配置

根据 train.sh 的参数设置：

```bash
--use_last_frame_loss          # ✅ 启用
--use_reg_loss                 # ❌ 未启用（默认 False）
--use_emo                       # ✅ 启用
--use_eye_open                 # ✅ 启用
--use_eye_ball                 # ✅ 启用
--use_sc                        # ✅ 启用（source canonical）
--use_last_frame               # ✅ 启用
```

## 损失项详细说明

### 1. 分段 PVA 损失（总是启用）

对运动特征的 6 个部分分别计算损失：

| 部分 | 维度范围 | 说明 | 损失类型 |
|------|---------|------|---------|
| **scale** (缩放) | [0, 1) | 面部大小缩放 | P + V + A |
| **pitch** (俯仰) | [1, 67) | 头部上下倾斜 | P + V + A |
| **yaw** (偏航) | [67, 133) | 头部左右转动 | P + V + A |
| **roll** (滚转) | [133, 199) | 头部倾斜角 | P + V + A |
| **t** (平移) | [199, 202) | 面部 3D 位移 | P + V + A |
| **exp** (表情) | [202, 265) | 面部表情参数 | P + V + A + L + R* |

**损失类型说明：**
- **P (Position Loss)**: `MSE(pred[s:e], gt[s:e])`
  - 直接位置差异

- **V (Velocity Loss)**: `MSE(∆pred[s:e], ∆gt[s:e])`
  - 相邻帧的速度差异
  - 约束运动的流畅性

- **A (Acceleration Loss)**: `MSE(∆²pred[s:e], ∆²gt[s:e])`
  - 二阶导数（加速度）
  - 约束运动的平顺性

- **L (Last Frame Loss)**: 仅对 `exp` 部分
  - `MSE(pred[:, 0, s:e], cond_frame[s:e])`
  - 确保首帧与条件帧一致

- **R (Regularization Loss)**: 仅对 `exp` 部分（当启用）
  - `λ × mean(|pred[s:e]|)`
  - 系数：scale 部分 λ=0，其他部分 λ=1e-4

### 2. 加权机制

每个部分的损失通过 `part_w_dict` 中的权重 `w` 进行加权：

```python
part_w_dict = {
    'scale': [0, 1, 1],      # 权重 = 1
    'pitch': [1, 67, 1],     # 权重 = 1
    'yaw': [67, 133, 1],     # 权重 = 1
    'roll': [133, 199, 1],   # 权重 = 1
    't': [199, 202, 1],      # 权重 = 1
    'exp': [202, 265, 1]     # 权重 = 1
}
```

所有部分的默认权重相同（都是 1）。

### 3. 维度加权（可选）

如果提供 `dim_ws` 参数，可以对维度级别的损失进行加权：
```python
dim_w = dim_ws[s:e] * w  # [1, 1, dim]
```

## 当前配置的启用情况

### ✅ 启用的损失项

1. **基础 PVA 损失** - 6 个部分 × 3 种损失 = 18 个损失项
   - scale_P, scale_V, scale_A
   - pitch_P, pitch_V, pitch_A
   - yaw_P, yaw_V, yaw_A
   - roll_P, roll_V, roll_A
   - t_P, t_V, t_A
   - exp_P, exp_V, exp_A

2. **Last Frame Loss** (`--use_last_frame_loss`)
   - 为每个部分额外添加 Last Frame 损失项（6 个）
   - 特别关注表情的初始值约束

### ❌ 未启用的损失项

1. **Regularization Loss** (`--use_reg_loss` 默认 False)
   - 如果需要约束运动幅度，可以添加 `--use_reg_loss`

## 总损失计算

```
total_loss = sum(all_loss_dict.values())
```

所有损失项直接求和，没有额外的加权组合。

## 建议调整

### 如果生成的表情太大或过度运动
```bash
添加 --use_reg_loss 标志
```

### 如果需要不同的部分权重
```bash
提供 --part_w_dict_json 参数
示例: {"scale": [0, 1, 0.5], "pitch": [1, 67, 1.0], ...}
```

### 如果需要维度级别的加权
```bash
提供 --dim_ws_npy 参数（NumPy 数组，形状 [265]）
```

## 总结

当前配置重点：
- ✅ 使用完整的 PVA 损失约束运动的位置、速度、加速度
- ✅ 约束初始表情帧与条件帧一致（Last Frame Loss）
- ✅ 使用所有条件特征（emotion、eye_open、eye_ball、source_canonical）
- ❌ 未使用正则化约束（如需要可启用）
- ❌ 未使用自定义的部分权重和维度加权
