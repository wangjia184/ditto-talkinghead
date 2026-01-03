"""
测试时间输入从 int 改为 float 的兼容性

该脚本验证：
1. 使用 int 时间输入的结果
2. 使用 float 时间输入的结果
3. 两者是否一致（应该一致，因为 float 会被四舍五入到最近的整数）
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MotionDiT', 'src'))

from models.LMDM import LMDM

def test_time_compatibility():
    """测试 int 和 float 时间输入的兼容性"""
    
    # 加载模型
    checkpoint_path = 'checkpoints/ditto_pytorch/models/lmdm_v0.4_hubert.pth'
    if not os.path.exists(checkpoint_path):
        print(f"警告: 检查点文件不存在: {checkpoint_path}")
        print("请确保检查点文件路径正确")
        return
    
    print("加载模型...")
    lmdm = LMDM(
        motion_feat_dim=265,
        audio_feat_dim=1103,
        seq_frames=80,
        checkpoint=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    lmdm.eval()
    
    device = lmdm.device
    batch_size = 1
    seq_len = 80
    motion_dim = 265
    audio_dim = 1103
    
    # 创建随机输入数据
    print("\n创建测试数据...")
    x = torch.randn(batch_size, seq_len, motion_dim, device=device)
    cond_frame = torch.randn(batch_size, motion_dim, device=device)
    cond = torch.randn(batch_size, seq_len, audio_dim, device=device)
    
    # 测试不同的时间值
    # 格式: (int值, float值, 期望的int值) - float值会被四舍五入到期望的int值
    # 注意: Python 的 round() 使用"银行家舍入"（round half to even）
    #   - round(250.5) = 250 (因为 250 是偶数)
    #   - round(251.5) = 252 (因为 252 是偶数)
    #   - round(250.3) = 250, round(250.7) = 251
    test_times = [
        (0, 0.0, 0),           # 整数 float，应该和 int 相同
        (100, 100.0, 100),     # 整数 float，应该和 int 相同
        (500, 500.0, 500),     # 整数 float，应该和 int 相同
        (999, 999.0, 999),     # 整数 float，应该和 int 相同
        (250, 250.0, 250),     # 整数 float，应该和 int 相同
        (250, 250.5, 250),     # 250.5 银行家舍入到 250（250是偶数），应该和 int=250 相同
        (750, 750.0, 750),     # 整数 float，应该和 int 相同
        (750, 750.3, 750),     # 750.3 四舍五入到 750，应该和 int=750 相同
        (751, 750.7, 751),     # 750.7 四舍五入到 751，应该和 int=751 相同
        (252, 251.5, 252),     # 251.5 银行家舍入到 252（252是偶数），应该和 int=252 相同
    ]
    
    print("\n开始测试...")
    print("=" * 80)
    
    all_passed = True
    
    for t_int_base, t_float, t_int_expected in test_times:
        # 计算 float 四舍五入后的值
        t_float_rounded = int(round(t_float))
        
        print(f"\n测试时间步: int={t_int_base}, float={t_float} (四舍五入到 {t_float_rounded})")
        
        # 验证四舍五入是否正确（Python 的 round 使用银行家舍入）
        if t_float_rounded != t_int_expected:
            print(f"  ⚠ 跳过: float={t_float} 实际四舍五入到 {t_float_rounded}，但测试期望 {t_int_expected}")
            print(f"  这是 Python round() 的银行家舍入行为（round half to even）")
            continue
        
        # 使用 int 时间（使用期望的 int 值进行比较）
        t_int_tensor = torch.full((batch_size,), t_int_expected, device=device, dtype=torch.long)
        with torch.no_grad():
            output_int = lmdm.diffusion.model(x, cond_frame, cond, t_int_tensor)
        
        # 使用 float 时间
        t_float_tensor = torch.full((batch_size,), t_float, device=device, dtype=torch.float32)
        with torch.no_grad():
            output_float = lmdm.diffusion.model(x, cond_frame, cond, t_float_tensor)
        
        # 比较输出
        max_diff = (output_int - output_float).abs().max().item()
        mean_diff = (output_int - output_float).abs().mean().item()
        
        # 检查是否一致（允许小的数值误差，由于浮点数精度）
        tolerance = 1e-6  # 更严格的容差，因为现在时间嵌入也会四舍五入
        is_close = max_diff < tolerance
        
        status = "✓ PASS" if is_close else "✗ FAIL"
        print(f"  最大差异: {max_diff:.2e}")
        print(f"  平均差异: {mean_diff:.2e}")
        print(f"  状态: {status}")
        
        if not is_close:
            all_passed = False
            print(f"  警告: 差异超过容差 {tolerance}")
            print(f"  注意: float={t_float} 应该四舍五入到 {t_float_rounded}，与 int={t_int_expected} 比较")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有测试通过！int 和 float 时间输入产生相同结果。")
    else:
        print("✗ 部分测试失败，请检查实现。")
    
    # 测试 extract 函数
    print("\n测试 extract 函数...")
    from models.modules.utils import extract
    
    # 创建一个测试数组
    test_array = torch.arange(1000, dtype=torch.float32)
    
    # 测试 int 索引
    t_int = torch.tensor([100, 500, 999], dtype=torch.long)
    result_int = extract(test_array, t_int, torch.zeros(3, 10, 10).shape)
    
    # 测试 float 索引（应该四舍五入）
    t_float = torch.tensor([100.0, 500.0, 999.0], dtype=torch.float32)
    result_float = extract(test_array, t_float, torch.zeros(3, 10, 10).shape)
    
    # 测试非整数 float（应该四舍五入）
    t_float_nonint = torch.tensor([100.3, 500.7, 999.4], dtype=torch.float32)
    result_float_nonint = extract(test_array, t_float_nonint, torch.zeros(3, 10, 10).shape)
    
    print(f"  int 索引结果: {result_int.squeeze().tolist()}")
    print(f"  float 索引结果: {result_float.squeeze().tolist()}")
    print(f"  float 非整数索引结果: {result_float_nonint.squeeze().tolist()}")
    
    # 验证四舍五入是否正确
    expected_nonint = torch.tensor([100, 501, 999])  # 100.3->100, 500.7->501, 999.4->999
    if torch.allclose(result_float_nonint.squeeze(), expected_nonint.float(), atol=1e-5):
        print("  ✓ extract 函数四舍五入测试通过")
    else:
        print("  ✗ extract 函数四舍五入测试失败")
        all_passed = False
    
    return all_passed

if __name__ == "__main__":
    success = test_time_compatibility()
    sys.exit(0 if success else 1)
