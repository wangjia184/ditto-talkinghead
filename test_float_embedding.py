"""
测试 float 数据是否真的进入模型内部

验证：
1. float 输入（如 255.2）是否真的进入 SinusoidalPosEmb
2. 还是只是做了转换
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MotionDiT', 'src'))

from models.modules.utils import SinusoidalPosEmb

def test_float_embedding():
    """测试 float 数据是否进入嵌入层"""
    
    print("=" * 80)
    print("测试 1: SinusoidalPosEmb 直接接收 float")
    print("=" * 80)
    
    emb = SinusoidalPosEmb(dim=512)
    
    # 测试整数
    t_int = torch.tensor([255.0], dtype=torch.float32)
    emb_int = emb(t_int)
    
    # 测试非整数 float
    t_float = torch.tensor([255.2], dtype=torch.float32)
    emb_float = emb(t_float)
    
    # 测试 round 后的 float
    t_rounded = t_float.round()
    emb_rounded = emb(t_rounded)
    
    print(f"\n输入值:")
    print(f"  t_int = {t_int.item()}")
    print(f"  t_float = {t_float.item()}")
    print(f"  t_rounded = {t_rounded.item()}")
    
    print(f"\n嵌入差异:")
    diff_int_float = (emb_int - emb_float).abs().max().item()
    diff_int_rounded = (emb_int - emb_rounded).abs().max().item()
    
    print(f"  |emb_int - emb_float| = {diff_int_float:.6e}")
    print(f"  |emb_int - emb_rounded| = {diff_int_rounded:.6e}")
    
    print(f"\n结论:")
    if diff_int_float > 1e-5:
        print(f"  ✓ float 数据（255.2）真的进入了嵌入层，产生了不同的嵌入")
        print(f"  ✓ 这就是为什么需要 round() 的原因")
    else:
        print(f"  ✗ float 和 int 产生了相同的嵌入（不应该发生）")
    
    if diff_int_rounded < 1e-6:
        print(f"  ✓ round() 后与整数产生相同的嵌入（正确）")
    
    print("\n" + "=" * 80)
    print("测试 2: 检查模型中的 round() 行为")
    print("=" * 80)
    
    # 模拟模型中的处理
    times_255_2 = torch.tensor([255.2], dtype=torch.float32)
    times_255_0 = torch.tensor([255.0], dtype=torch.float32)
    
    # 模拟 model.py 中的代码
    if times_255_2.dtype.is_floating_point:
        times_255_2_rounded = times_255_2.round()
    else:
        times_255_2_rounded = times_255_2.float()
    
    print(f"\n输入: times_255_2 = {times_255_2.item()}")
    print(f"round() 后: times_255_2_rounded = {times_255_2_rounded.item()}")
    print(f"比较: times_255_0 = {times_255_0.item()}")
    
    # 检查是否相同
    if torch.allclose(times_255_2_rounded, times_255_0, atol=1e-6):
        print(f"\n✓ round() 后与 255.0 相同")
        print(f"✓ 这意味着 255.2 会被转换为 255.0，然后进入嵌入层")
    else:
        print(f"\n✗ round() 后与 255.0 不同（不应该发生）")
    
    # 测试嵌入
    emb_255_2_rounded = emb(times_255_2_rounded)
    emb_255_0 = emb(times_255_0)
    
    diff = (emb_255_2_rounded - emb_255_0).abs().max().item()
    print(f"\n嵌入差异: |emb_255_2_rounded - emb_255_0| = {diff:.6e}")
    
    if diff < 1e-6:
        print(f"✓ 嵌入完全相同（正确）")
    else:
        print(f"✗ 嵌入不同（不应该发生）")
    
    print("\n" + "=" * 80)
    print("测试 3: 如果不用 round() 会怎样？")
    print("=" * 80)
    
    # 模拟不使用 round() 的情况
    times_no_round = torch.tensor([255.2], dtype=torch.float32)
    emb_no_round = emb(times_no_round)
    emb_with_round = emb(times_no_round.round())
    
    diff_no_round = (emb_no_round - emb_with_round).abs().max().item()
    
    print(f"\n不使用 round(): emb(255.2)")
    print(f"使用 round(): emb(round(255.2)) = emb(255.0)")
    print(f"差异: {diff_no_round:.6e}")
    
    if diff_no_round > 1e-5:
        print(f"\n✓ 如果不使用 round()，255.2 会产生不同的嵌入")
        print(f"✓ 这就是为什么需要 round() 的原因")
    else:
        print(f"\n✗ 不使用 round() 也产生相同的嵌入（不应该发生）")

if __name__ == "__main__":
    test_float_embedding()
