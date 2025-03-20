import numpy as np
import torch
from ..utils.load_model import load_model

import a2h

class StitchNetwork:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "StitchingNetwork",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, kp_source, kp_driving):
        if self.model_type == "onnx":
            pred = self.model.run(None, {"kp_source": kp_source, "kp_driving": kp_driving})[0]
        elif self.model_type == "tensorrt":
            #print( kp_source.shape, kp_source.dtype, kp_driving.shape, kp_driving.dtype)
            #self.model.setup({"kp_source": kp_source, "kp_driving": kp_driving})
            #self.model.infer()
            #pred = self.model.buffer["out"][0].copy()
            #print( pred.shape, pred.dtype )
            pred = a2h.stitch_face( np.ascontiguousarray(kp_source), np.ascontiguousarray(kp_driving) )
        elif self.model_type == 'pytorch':
            with torch.no_grad():
                pred = self.model(
                    torch.from_numpy(kp_source).to(self.device), 
                    torch.from_numpy(kp_driving).to(self.device)
                ).cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return pred
