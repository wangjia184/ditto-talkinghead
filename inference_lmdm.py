import torch
import numpy as np
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MotionDiT', 'src'))

from models.LMDM import LMDM

def load_data_sample(data_list_json, index=0):
    """从 data_list_train.json 加载一个数据样本"""
    with open(data_list_json, 'r') as f:
        data_list = json.load(f)
    
    if index >= len(data_list):
        index = 0
        print(f"Warning: index {index} out of range, using first sample")
    
    data_item = data_list[index]
    print(f"\n=== Loading data sample {index} ===")
    print(f"Data item keys: {data_item.keys()}")
    
    # 加载 motion 和 audio 数据
    mtn_path = data_item.get('mtn', data_item.get('kps_npy', ''))
    aud_path = data_item.get('aud', data_item.get('aud_npy', ''))
    frame_num = data_item.get('frame_num', 80)
    
    print(f"Motion file: {mtn_path}")
    print(f"Audio file: {aud_path}")
    print(f"Frame number: {frame_num}")
    
    # 加载数据
    mtn_data = np.load(mtn_path)  # [n_frames, motion_feat_dim]
    aud_data = np.load(aud_path)  # [n_frames, audio_feat_dim]
    
    print(f"Motion data shape: {mtn_data.shape}")
    print(f"Audio data shape: {aud_data.shape}")
    
    # 检查 motion 特征维度
    actual_motion_dim = mtn_data.shape[1] if len(mtn_data.shape) > 1 else mtn_data.shape[0]
    print(f"Actual motion feature dim: {actual_motion_dim}")
    
    return mtn_data, aud_data, frame_num, actual_motion_dim

def prepare_inputs(mtn_data, aud_data, seq_frames=80, motion_feat_dim=265, audio_feat_dim=1103):
    """准备模型输入"""
    # 选择条件帧（第一帧）
    kp_cond = mtn_data[0]  # [motion_feat_dim]
    
    # 处理音频特征维度
    if aud_data.shape[1] != audio_feat_dim:
        print(f"Warning: Audio feature dim mismatch! Expected {audio_feat_dim}, got {aud_data.shape[1]}")
        if aud_data.shape[1] < audio_feat_dim:
            # 如果维度不够，用零填充
            padding = np.zeros((aud_data.shape[0], audio_feat_dim - aud_data.shape[1]))
            aud_data = np.concatenate([aud_data, padding], axis=1)
            print(f"  Padded audio features to {audio_feat_dim} dims")
        else:
            # 如果维度太多，截断
            aud_data = aud_data[:, :audio_feat_dim]
            print(f"  Truncated audio features to {audio_feat_dim} dims")
    
    # 选择序列（前 seq_frames 帧）
    if len(mtn_data) < seq_frames:
        # 如果帧数不够，重复最后一帧
        kp_seq = mtn_data[:seq_frames]
        aud_cond = aud_data[:seq_frames]
        if len(kp_seq) < seq_frames:
            last_frame = kp_seq[-1:]
            last_aud = aud_cond[-1:]
            kp_seq = np.concatenate([kp_seq, np.repeat(last_frame, seq_frames - len(kp_seq), axis=0)], axis=0)
            aud_cond = np.concatenate([aud_cond, np.repeat(last_aud, seq_frames - len(aud_cond), axis=0)], axis=0)
    else:
        kp_seq = mtn_data[:seq_frames]
        aud_cond = aud_data[:seq_frames]
    
    # 转换为 tensor
    kp_cond_tensor = torch.from_numpy(kp_cond).float().unsqueeze(0)  # [1, motion_feat_dim]
    aud_cond_tensor = torch.from_numpy(aud_cond).float().unsqueeze(0)  # [1, seq_frames, audio_feat_dim]
    
    print(f"\n=== Prepared Inputs ===")
    print(f"kp_cond shape: {kp_cond_tensor.shape}")
    print(f"aud_cond shape: {aud_cond_tensor.shape}")
    print(f"kp_cond dtype: {kp_cond_tensor.dtype}")
    print(f"aud_cond dtype: {aud_cond_tensor.dtype}")
    
    return kp_cond_tensor, aud_cond_tensor

def print_tensor_info(name, tensor):
    """打印 tensor 的详细信息"""
    print(f"\n--- {name} ---")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Min: {tensor.min().item():.6f}")
    print(f"Max: {tensor.max().item():.6f}")
    print(f"Mean: {tensor.mean().item():.6f}")
    print(f"Std: {tensor.std().item():.6f}")
    print(f"First few values: {tensor.flatten()[:10].tolist()}")

def main():
    # 配置
    checkpoint_path = "checkpoints/ditto_pytorch/models/lmdm_v0.4_hubert.pth"
    data_list_json = "data_list_train.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("LMDM Model Inference Script")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data list: {data_list_json}")
    
    # 1. 加载模型
    print("\n" + "=" * 80)
    print("Step 1: Loading Model")
    print("=" * 80)
    
    lmdm = LMDM(
        motion_feat_dim=265,
        audio_feat_dim=1103,  # 1024 + 63 + 8 + 2 + 6 (HuBERT + SC + emo + eye_open + eye_ball)
        seq_frames=int(3.2 * 25),  # 80 frames
        checkpoint=checkpoint_path,
        device=device,
    )
    lmdm.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in lmdm.model.parameters()):,}")
    
    # 2. 加载数据
    print("\n" + "=" * 80)
    print("Step 2: Loading Data")
    print("=" * 80)
    
    mtn_data, aud_data, frame_num, actual_motion_dim = load_data_sample(data_list_json, index=0)
    
    # 检查 motion 特征维度是否匹配
    expected_motion_dim = 265
    print(f"\n=== Motion Feature Dimension Check ===")
    print(f"Expected: {expected_motion_dim}")
    print(f"Actual: {actual_motion_dim}")
    
    if actual_motion_dim != expected_motion_dim:
        print(f"\nWarning: Motion feature dim mismatch!")
        if actual_motion_dim > expected_motion_dim:
            print(f"  Truncating motion features from {actual_motion_dim} to {expected_motion_dim} dims")
            mtn_data = mtn_data[:, :expected_motion_dim]
        else:
            print(f"  Padding motion features from {actual_motion_dim} to {expected_motion_dim} dims")
            padding = np.zeros((mtn_data.shape[0], expected_motion_dim - actual_motion_dim))
            mtn_data = np.concatenate([mtn_data, padding], axis=1)
        print(f"  Final motion data shape: {mtn_data.shape}")
    
    # 3. 准备输入
    print("\n" + "=" * 80)
    print("Step 3: Preparing Inputs")
    print("=" * 80)
    
    kp_cond, aud_cond = prepare_inputs(mtn_data, aud_data, audio_feat_dim=1103)
    
    # 打印输入详细信息
    print_tensor_info("kp_cond (Condition Frame)", kp_cond)
    print_tensor_info("aud_cond (Audio Condition)", aud_cond)
    
    # 4. 推理
    print("\n" + "=" * 80)
    print("Step 4: Running Inference")
    print("=" * 80)
    
    with torch.no_grad():
        pred_kp_seq = lmdm._run_diffusion_render_sample(
            kp_cond=kp_cond,
            aud_cond=aud_cond,
            noise=None
        )
    
    # 5. 打印输出
    print("\n" + "=" * 80)
    print("Step 5: Output Analysis")
    print("=" * 80)
    
    print_tensor_info("pred_kp_seq (Predicted Keypoint Sequence)", pred_kp_seq)
    
    # 6. 对比输入输出
    print("\n" + "=" * 80)
    print("Step 6: Input/Output Comparison")
    print("=" * 80)
    
    print(f"\nInput shapes:")
    print(f"  kp_cond: {kp_cond.shape}")
    print(f"  aud_cond: {aud_cond.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  pred_kp_seq: {pred_kp_seq.shape}")
    
    print(f"\nExpected shapes:")
    print(f"  kp_cond: [1, 265]")
    print(f"  aud_cond: [1, 80, 1103]")
    print(f"  pred_kp_seq: [1, 80, 265]")
    
    # 7. 模型结构信息
    print("\n" + "=" * 80)
    print("Step 7: Model Structure Summary")
    print("=" * 80)
    
    print(f"\nModel Configuration:")
    print(f"  Motion feature dim: {lmdm.motion_feat_dim}")
    print(f"  Audio feature dim: {lmdm.audio_feat_dim}")
    print(f"  Sequence frames: {lmdm.seq_frames}")
    print(f"  Device: {lmdm.device}")
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
