#!/usr/bin/env python3
"""
Export LMDM checkpoint to ONNX format
"""
import os
import sys
import torch
import torch.nn as nn
import argparse

# Add MotionDiT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MotionDiT'))

from src.models.LMDM import LMDM


class ModelWrapper(nn.Module):
    """Wrapper class for ONNX export that returns both pred_noise and x_start"""
    def __init__(self, model, diffusion):
        super().__init__()
        self.model = model
        # Register buffers needed for noise prediction
        self.register_buffer('sqrt_recip_alphas_cumprod', diffusion.sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', diffusion.sqrt_recipm1_alphas_cumprod)
        self.guidance_weight = diffusion.guidance_weight
        self.clip_denoised = diffusion.clip_denoised
    
    def extract(self, a, t, x_shape):
        """Extract values from buffer at timestep t"""
        b, *_ = t.shape
        # Ensure a is at least 2D for gather operation
        # a shape: [n_timestep], t shape: [B]
        # We need to unsqueeze a to [1, n_timestep] for gather(-1, t)
        a = a.unsqueeze(0)  # [n_timestep] -> [1, n_timestep]
        out = a.gather(-1, t.unsqueeze(-1))  # t: [B] -> [B, 1], gather -> [B, 1]
        out = out.squeeze(-1)  # [B, 1] -> [B]
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def predict_noise_from_start(self, x_t, t, x0):
        """Predict noise from x_start"""
        return (
            (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / 
            self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def forward(self, x, cond_frame, cond_embed, times):
        """
        Forward pass that returns both pred_noise and x_start
        
        Args:
            x: noisy input [B, L, D]
            cond_frame: condition frame [B, D]
            cond_embed: condition embedding [B, L, C]
            times: timesteps [B]
        
        Returns:
            pred_noise: predicted noise [B, L, D]
            x_start: predicted x_start [B, L, D]
        """
        # Get guided forward output (x_start)
        # This calls forward twice: once with cond_drop_prob=1 (unconditioned) and once with cond_drop_prob=0 (conditioned)
        unc = self.model(x, cond_frame, cond_embed, times, cond_drop_prob=1.0)
        conditioned = self.model(x, cond_frame, cond_embed, times, cond_drop_prob=0.0)
        x_start = unc + (conditioned - unc) * self.guidance_weight
        
        # Clip if needed
        if self.clip_denoised:
            x_start = torch.clamp(x_start, min=-1.0, max=1.0)
        
        # Predict noise from x_start
        pred_noise = self.predict_noise_from_start(x, times, x_start)
        
        return pred_noise, x_start


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    motion_feat_dim: int = 265,
    audio_feat_dim: int = 1103,
    seq_frames: int = 80,
    device: str = 'cuda',
    opset_version: int = 14,
    use_dynamic_axes: bool = True,
):
    """
    Load checkpoint and export to ONNX format
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        output_path: Path to save the ONNX model
        motion_feat_dim: Motion feature dimension (default: 265)
        audio_feat_dim: Audio feature dimension (default: 1103)
        seq_frames: Sequence frames (default: 80, which is 3.2 * 25)
        device: Device to use ('cuda' or 'cpu')
        opset_version: ONNX opset version (default: 14)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    print(f"Creating model with parameters:")
    print(f"  motion_feat_dim: {motion_feat_dim}")
    print(f"  audio_feat_dim: {audio_feat_dim}")
    print(f"  seq_frames: {seq_frames}")
    
    lmdm = LMDM(
        motion_feat_dim=motion_feat_dim,
        audio_feat_dim=audio_feat_dim,
        seq_frames=seq_frames,
        checkpoint=checkpoint_path,
        device=device,
    )
    
    # Set model to eval mode
    lmdm.eval()
    model = lmdm.model.to(device)
    diffusion = lmdm.diffusion
    
    # Create dummy inputs for export
    batch_size = 1
    x = torch.randn(batch_size, seq_frames, motion_feat_dim).to(device)
    cond_frame = torch.randn(batch_size, motion_feat_dim).to(device)
    cond_embed = torch.randn(batch_size, seq_frames, audio_feat_dim).to(device)
    times = torch.randint(0, 1000, (batch_size,)).to(device).long()
    
    print(f"\nExporting model to ONNX format...")
    print(f"Output path: {output_path}")
    print(f"Model will output: pred_noise and x_start")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Export to ONNX
    # The wrapper returns both pred_noise and x_start
    with torch.no_grad():
        # Create a wrapper module that includes diffusion buffers and returns both outputs
        wrapped_model = ModelWrapper(model, diffusion).to(device)
        wrapped_model.eval()
        
        # Prepare export arguments
        export_kwargs = {
            'opset_version': opset_version,
            'do_constant_folding': True,
            'verbose': False,
            'input_names': ['x', 'cond_frame', 'cond', 'time_cond'],
            'output_names': ['pred_noise', 'x_start'],
        }
        
        # Add dynamic axes only if requested (fixed size is better for Netron)
        if use_dynamic_axes:
            export_kwargs['dynamic_axes'] = {
                'x': {0: 'batch_size', 1: 'seq_len'},
                'cond_frame': {0: 'batch_size'},
                'cond': {0: 'batch_size', 1: 'seq_len'},
                'time_cond': {0: 'batch_size'},
                'pred_noise': {0: 'batch_size', 1: 'seq_len'},
                'x_start': {0: 'batch_size', 1: 'seq_len'},
            }
            print("Using dynamic axes (batch_size and seq_len can vary)")
        else:
            print("Using fixed size (better for Netron visualization)")
        
        torch.onnx.export(
            wrapped_model,
            (x, cond_frame, cond_embed, times),
            output_path,
            **export_kwargs
        )
    
    print(f"\n✓ Successfully exported model to: {output_path}")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed")
    except ImportError:
        print("Warning: onnx package not found, skipping model verification")
    except Exception as e:
        print(f"Warning: ONNX model verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export LMDM checkpoint to ONNX format')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save ONNX model (.onnx)'
    )
    parser.add_argument(
        '--motion_feat_dim',
        type=int,
        default=265,
        help='Motion feature dimension (default: 265)'
    )
    parser.add_argument(
        '--audio_feat_dim',
        type=int,
        default=1103,
        help='Audio feature dimension (default: 1103)'
    )
    parser.add_argument(
        '--seq_frames',
        type=int,
        default=80,
        help='Sequence frames (default: 80, which is 3.2 * 25)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )
    parser.add_argument(
        '--use_dynamic_axes',
        action='store_true',
        help='Use dynamic axes (batch_size, seq_len). If not set, uses fixed size (better for Netron)'
    )
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        motion_feat_dim=args.motion_feat_dim,
        audio_feat_dim=args.audio_feat_dim,
        seq_frames=args.seq_frames,
        device=args.device,
        opset_version=args.opset_version,
        use_dynamic_axes=args.use_dynamic_axes,
    )


if __name__ == '__main__':
    main()
