#!/usr/bin/env python3
"""
检查预处理所需的模型文件是否完整
"""

import os
import sys
from pathlib import Path


def check_model_files(ditto_pytorch_path):
    """检查所有必需的模型文件"""
    ditto_pytorch_path = Path(ditto_pytorch_path)
    
    # 必需的模型文件
    required_models = {
        'models/appearance_extractor.pth': '外观特征提取器',
        'models/decoder.pth': '解码器',
        'models/motion_extractor.pth': '运动特征提取器',
        'models/stitch_network.pth': '拼接网络',
        'models/warp_network.pth': '变形网络',
    }
    
    required_aux_models = {
        'aux_models/2d106det.onnx': 'InsightFace 人脸检测模型',
        'aux_models/det_10g.onnx': 'InsightFace 人脸检测模型',
        'aux_models/face_landmarker.task': 'MediaPipe 人脸关键点模型',
        'aux_models/hubert_streaming_fix_kv.onnx': 'Hubert 音频特征提取模型',
        'aux_models/landmark203.onnx': '关键点检测模型',
    }
    
    # InsightFace 软链接（可选，会自动创建）
    insightface_links = {
        'aux_models/insightface/models/buffalo_l/2d106det.onnx': 'InsightFace 软链接',
        'aux_models/insightface/models/buffalo_l/det_10g.onnx': 'InsightFace 软链接',
    }
    
    print("=" * 60)
    print("Checking model files completeness")
    print("=" * 60)
    print(f"检查路径: {ditto_pytorch_path}\n")
    
    all_ok = True
    
    # 检查 models/ 目录
    print("[models/] directory:")
    for rel_path, desc in required_models.items():
        full_path = ditto_pytorch_path / rel_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {rel_path:50s} ({size_mb:.1f} MB) - {desc}")
        else:
            print(f"  [MISSING] {rel_path:50s} - {desc}")
            all_ok = False
    
    # 检查 aux_models/ 目录
    print("\n[aux_models/] directory:")
    for rel_path, desc in required_aux_models.items():
        full_path = ditto_pytorch_path / rel_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {rel_path:50s} ({size_mb:.1f} MB) - {desc}")
        else:
            print(f"  [MISSING] {rel_path:50s} - {desc}")
            all_ok = False
    
    # 检查 InsightFace 软链接
    print("\n[InsightFace] symlinks:")
    for rel_path, desc in insightface_links.items():
        full_path = ditto_pytorch_path / rel_path
        if full_path.exists() or full_path.is_symlink():
            print(f"  [OK] {rel_path:50s} - {desc}")
        else:
            print(f"  [WARN] {rel_path:50s} - {desc} (will be auto-created)")
    
    # 检查情绪检测模型（Python 包）
    print("\n[Python packages] dependencies:")
    try:
        import hsemotion
        print(f"  [OK] hsemotion - emotion detection library (installed)")
        print(f"       Note: model will be auto-downloaded on first use")
    except ImportError:
        print(f"  [MISSING] hsemotion - emotion detection library")
        print(f"       Install: pip install hsemotion")
        all_ok = False
    
    try:
        import facenet_pytorch
        print(f"  [OK] facenet-pytorch - MTCNN face detection (installed)")
    except ImportError:
        print(f"  [MISSING] facenet-pytorch - MTCNN face detection")
        print(f"       Install: pip install facenet-pytorch")
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("[SUCCESS] All required model files are ready!")
        print("\n[Tips]:")
        print("   - InsightFace symlinks will be auto-created by check_ckpt_path.py")
        print("   - Emotion detection model will be auto-downloaded on first use")
    else:
        print("[ERROR] Missing model files detected, please check the list above")
        print("\n[Download missing models]:")
        print("   git lfs install")
        print("   git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints")
    print("=" * 60)
    
    return all_ok


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ditto_pytorch_path = sys.argv[1]
    else:
        # 默认路径
        script_dir = Path(__file__).parent
        ditto_pytorch_path = script_dir / 'checkpoints' / 'ditto_pytorch'
    
    if not os.path.exists(ditto_pytorch_path):
        print(f"Error: Path does not exist: {ditto_pytorch_path}")
        print(f"\nUsage: python {sys.argv[0]} [ditto_pytorch_path]")
        sys.exit(1)
    
    success = check_model_files(ditto_pytorch_path)
    sys.exit(0 if success else 1)

