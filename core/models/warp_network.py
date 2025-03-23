import numpy as np
import torch
from ..utils.load_model import load_model

import a2h

class WarpNetwork:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "WarpingNetwork",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, feature_3d, kp_source, kp_driving):
        """
        feature_3d: np.ndarray, shape (1, 32, 16, 64, 64)
        kp_source | kp_driving: np.ndarray, shape (1, 21, 3)
        """
        print( kp_source.shape, kp_driving.shape)
        pred = a2h.warp( np.ascontiguousarray(feature_3d), np.ascontiguousarray(kp_source), np.ascontiguousarray(kp_driving) )
        
        return pred
