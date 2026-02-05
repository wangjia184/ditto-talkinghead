import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from tqdm import tqdm

from .utils import prob_mask_like


class MotionMeanFlow(nn.Module):
    """
    Improved MeanFlow for motion generation.
    """

    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        guidance_weight=3,
        cond_drop_prob=0.2,
        part_w_dict=None,
        use_last_frame_loss=False,
        use_reg_loss=False,
        dim_ws=None,
        loss_type="l2",
        # iMF parameters
        P_mean=-0.4,
        P_std=1.0,
        class_dropout_prob=0.1,
        norm_p=1.0,
        norm_eps=0.01,
        # JVP / attention options
        use_jvp=False,
        disable_flash_attention=False,
        debug_pva=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model

        self.cond_drop_prob = cond_drop_prob
        self.guidance_weight = guidance_weight

        # Loss configuration
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss
        self.part_w_dict = part_w_dict
        self.use_last_frame_loss = use_last_frame_loss
        self.use_reg_loss = use_reg_loss

        if self.part_w_dict is None:
            self.part_w_dict = {
                "scale": [0, 1, 1],
                "pitch": [1, 67, 1],
                "yaw": [67, 133, 1],
                "roll": [133, 199, 1],
                "t": [199, 202, 1],
                "exp": [202, 265, 1],
            }

        if dim_ws is not None:
            self.register_buffer("dim_ws", torch.from_numpy(dim_ws))
        else:
            self.dim_ws = None

        # iMF training parameters
        self.P_mean = P_mean
        self.P_std = P_std
        self.class_dropout_prob = class_dropout_prob
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.use_jvp = use_jvp
        self.debug_pva = debug_pva or bool(int(os.getenv("MEANFLOW_DEBUG_PVA", "0")))

        if disable_flash_attention:
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

    # ------------------------------------------ sampling ------------------------------------------#

    def sample_one_step(self, z_t, cond_frame, cond, i, t_steps):
        """
        Perform one sampling step given current state z_t at time step i.
        """
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = z_t.shape[0]

        t = t.expand(bsz).to(z_t.device)
        r = r.expand(bsz).to(z_t.device)
        h = t - r

        u, _ = self.model(z_t, cond_frame, cond, h, cond_drop_prob=0.0)

        # Flow update: z_next = z_t - (t - r) * u
        return z_t - (t - r)[:, None, None] * u

    @torch.no_grad()
    def flow_sample(self, shape, cond_frame, cond, num_steps=1, noise=None):
        """
        Flow-based sampling.
        """
        batch, device = shape[0], cond_frame.device

        if noise is not None:
            z_t = noise.to(device)
        else:
            z_t = torch.randn(shape, device=device, dtype=cond_frame.dtype)

        cond_frame = cond_frame.to(device)
        cond = cond.to(device)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            z_t = self.sample_one_step(z_t, cond_frame, cond, i, t_steps)

        return z_t

    # ------------------------------------------ training ------------------------------------------#

    def logit_normal_dist(self, bz, device):
        rnd_normal = torch.randn(bz, 1, 1, 1, device=device, dtype=torch.float32)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz, device):
        """
        Sample t and r from logit-normal distribution (no Flow Matching).
        """
        t = self.logit_normal_dist(bz, device)  # [B, 1, 1, 1]
        r = self.logit_normal_dist(bz, device)  # [B, 1, 1, 1]
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        # Return as [B] tensors
        t = t.squeeze()
        r = r.squeeze()
        return t, r

    def v_cond_fn(self, z_t, t, cond_frame, cond):
        """
        Get conditioned v from v-head (for JVP tangent).
        Use h=0 following imeanflow.
        """
        h = torch.zeros_like(t)
        _, v = self.model(z_t, cond_frame, cond, h, cond_drop_prob=0.0)
        return v

    def v_fn(self, z_t, t, cond_frame, cond):
        """
        Get both conditioned and unconditioned v predictions.
        """
        h = torch.zeros_like(t)

        # Conditioned v
        _, v_c = self.model(z_t, cond_frame, cond, h, cond_drop_prob=0.0)
        # Unconditioned v
        _, v_u = self.model(z_t, cond_frame, cond, h, cond_drop_prob=1.0)

        # If v is None (no v-head), fall back to u
        if v_c is None:
            u_c, _ = self.model(z_t, cond_frame, cond, h, cond_drop_prob=0.0)
            u_u, _ = self.model(z_t, cond_frame, cond, h, cond_drop_prob=1.0)
            return u_c, u_u

        return v_c, v_u

    def cond_drop(self, v_t, v_g, bz, device):
        """
        Conditional dropout: for some samples, replace v_g with v_t.
        """
        rand_mask = torch.rand(bz, device=device) < self.class_dropout_prob
        num_drop = rand_mask.sum().int()
        drop_mask = torch.arange(bz, device=device) < num_drop
        drop_mask = drop_mask[:, None, None]

        v_g = torch.where(drop_mask, v_t, v_g)
        return v_g, drop_mask.squeeze()

    def guidance_fn(self, v_t, z_t, t, r, cond_frame, cond):
        """
        Compute CFG-guided velocity v_g.
        """
        omega = self.guidance_weight

        # Get conditioned and unconditioned v predictions
        v_c, v_u = self.v_fn(z_t, t, cond_frame, cond)

        # CFG: v_g = v_t + (1 - 1/ω) * (v_c - v_u)
        v_g = v_t + (1 - 1.0 / omega) * (v_c - v_u)

        return v_g, v_c

    def q_sample(self, x_start, t, noise=None):
        """
        Linear flow forward: z_t = (1-t)*x + t*e
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        if t.dim() == 1:
            t = t[:, None, None]
        elif t.dim() == 2:
            t = t[:, :, None]

        z_t = (1 - t) * x_start + t * noise
        return z_t

    def p_losses(self, x_start, cond_frame, cond):
        """
        iMF training loss with JVP.
        """
        bz = x_start.shape[0]
        device = x_start.device

        # Sample t, r
        t, r = self.sample_tr(bz, device)

        # Forward flow: z_t = (1-t)*x + t*e
        e = torch.randn_like(x_start)
        z_t = self.q_sample(x_start, t, noise=e)
        v_t = e - x_start  # Instantaneous velocity

        # Compute guided velocity v_g and conditioned velocity v_c
        v_g, v_c = self.guidance_fn(v_t, z_t, t, r, cond_frame, cond)

        # Conditional dropout: for some samples, v_g = v_t
        v_g, _ = self.cond_drop(v_t, v_g, bz, device)

        # Warped u-function for JVP computation
        def u_fn(z_t, t, r):
            h = t - r
            u, v = self.model(z_t, cond_frame, cond, h, cond_drop_prob=0.0)
            return u, v

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(r)

        if self.use_jvp:
            # Compute JVP (match imeanflow). Handle API differences across PyTorch versions.
            jvp_out = jvp(
                u_fn,
                (z_t, t, r),
                (v_c, dtdt, dtdr),
                has_aux=True,
            )
            if isinstance(jvp_out, tuple) and len(jvp_out) >= 2:
                main_out = jvp_out[0]
                du = jvp_out[1]
                # main_out can be (u, v) or just u; handle both
                if isinstance(main_out, (tuple, list)):
                    u = main_out[0]
                    v = main_out[1] if len(main_out) > 1 else None
                else:
                    u = main_out
                    v = None
            else:
                raise RuntimeError(
                    f"Unexpected jvp output: {type(jvp_out)} / len={len(jvp_out)}"
                )
        else:
            # Finite difference DDE: du/dh
            h = t - r
            u, v = self.model(z_t, cond_frame, cond, h, cond_drop_prob=0.0)
            eps = 1e-4
            h_eps = h + eps
            u_eps, _ = self.model(z_t, cond_frame, cond, h_eps, cond_drop_prob=0.0)
            du = (u_eps - u) / eps

        # Compound function V = u + (t - r) * stop_gradient(du)
        t_r = (t - r)[:, None, None]
        V = u + t_r * du.detach()

        v_g = v_g.detach()

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / adp_wt.detach()

        loss_u = torch.sum((V - v_g) ** 2, dim=(1, 2))
        loss_u = adp_wt_fn(loss_u)

        # Ensure v is available for auxiliary loss
        if v is None:
            _, v = self.model(z_t, cond_frame, cond, t - r, cond_drop_prob=0.0)

        loss_v = torch.zeros_like(loss_u)
        if v is not None:
            loss_v = torch.sum((v - v_g) ** 2, dim=(1, 2))
            loss_v = adp_wt_fn(loss_v)

        loss = (loss_u + loss_v).mean()

        # PVA monitoring (no grad)
        with torch.no_grad():
            pva_dict = self._get_pva_loss_monitor(u, v_g, cond_frame)

        loss_dict = {
            "loss": loss,
            "loss_u": torch.mean((V - v_g) ** 2),
            "loss_v": torch.mean((v - v_g) ** 2)
            if v is not None
            else torch.tensor(0.0, device=device),
        }
        loss_dict.update(pva_dict)

        return loss, loss_dict

    def _get_pva_loss_monitor(self, pred, gt, cond_frame):
        """
        Compute PVA losses for monitoring only (no gradient).
        """
        loss_dict = {}

        def _crop_time(a, b):
            t = min(a.shape[1], b.shape[1])
            return a[:, :t], b[:, :t]

        def _crop_dim(a, b):
            if a.dim() < 3 or b.dim() < 3:
                if self.debug_pva:
                    print(f"[PVA] bad dim: a.shape={tuple(a.shape)} b.shape={tuple(b.shape)}")
                return a, b
            d = min(a.shape[2], b.shape[2])
            return a[:, :, :d], b[:, :, :d]

        def _part_loss(s, e, w):
            if s <= 0:
                s = 0
            if e <= 0:
                e = gt.shape[-1]

            p1 = pred[..., s:e]
            p2 = gt[..., s:e]

            if self.debug_pva:
                print(f"[PVA] k dims s={s} e={e} pred={tuple(pred.shape)} gt={tuple(gt.shape)} p1={tuple(p1.shape)} p2={tuple(p2.shape)}")

            # Align time and feature dims
            p1, p2 = _crop_time(p1, p2)
            p1, p2 = _crop_dim(p1, p2)

            if p1.dim() < 3 or p1.shape[1] < 2 or p1.shape[2] == 0:
                zero = torch.tensor(0.0, device=p1.device, dtype=p1.dtype)
                return zero, zero, zero

            dim_w = w
            if self.dim_ws is not None:
                dim_w = self.dim_ws[s : s + p1.shape[2]][None, None] * w

            v1 = p1[:, 1:] - p1[:, :-1]
            v2 = p2[:, 1:] - p2[:, :-1]
            v1, v2 = _crop_time(v1, v2)
            v1, v2 = _crop_dim(v1, v2)

            if v1.shape[1] < 2 or v1.shape[2] == 0:
                _p_loss = (self.loss_fn(p1, p2, reduction="none") * dim_w).mean()
                zero = torch.tensor(0.0, device=p1.device, dtype=p1.dtype)
                return _p_loss, zero, zero

            a1 = v1[:, 1:] - v1[:, :-1]
            a2 = v2[:, 1:] - v2[:, :-1]
            a1, a2 = _crop_time(a1, a2)
            a1, a2 = _crop_dim(a1, a2)

            _p_loss = (self.loss_fn(p1, p2, reduction="none") * dim_w).mean()
            _v_loss = (self.loss_fn(v1, v2, reduction="none") * dim_w).mean()

            if a1.shape[1] == 0 or a1.shape[2] == 0:
                zero = torch.tensor(0.0, device=p1.device, dtype=p1.dtype)
                return _p_loss, _v_loss, zero

            _a_loss = (self.loss_fn(a1, a2, reduction="none") * dim_w).mean()
            return _p_loss, _v_loss, _a_loss

        p_vals = []
        v_vals = []
        a_vals = []
        for _, (s, e, w) in self.part_w_dict.items():
            p_loss, v_loss, a_loss = _part_loss(s, e, w)
            p_vals.append(p_loss)
            v_vals.append(v_loss)
            a_vals.append(a_loss)

        # Aggregate P/V/A monitor values (mean across parts)
        if len(p_vals) > 0:
            loss_dict["monitor_P"] = torch.stack(p_vals).mean()
            loss_dict["monitor_V"] = torch.stack(v_vals).mean()
            loss_dict["monitor_A"] = torch.stack(a_vals).mean()
        else:
            zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            loss_dict["monitor_P"] = zero
            loss_dict["monitor_V"] = zero
            loss_dict["monitor_A"] = zero

        return loss_dict

    def loss(self, x, cond_frame, cond, t_override=None):
        return self.p_losses(x, cond_frame, cond)

    def forward(self, x, cond_frame, cond, t_override=None):
        return self.loss(x, cond_frame, cond, t_override)

    def render_sample(
        self,
        shape,
        cond_frame,
        cond,
        normalizer=None,
        epoch=None,
        render_out=None,
        last_half=None,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
    ):
        """
        Render samples using flow-based sampling.
        """
        if isinstance(shape, tuple):
            samples = (
                self.flow_sample(
                    shape,
                    cond_frame,
                    cond,
                    num_steps=1,  # 1-NFE by default
                    noise=noise,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape

        if render_out is None:
            return samples

        os.makedirs(render_out, exist_ok=True)
        for i in range(samples.shape[0]):
            np.save(
                f"{render_out}/{epoch}_{os.path.basename(name[i])[:-4]}.npy",
                samples[i].numpy(),
            )

        return samples
