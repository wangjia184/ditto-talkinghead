import numpy as np
import torch
from ..utils.load_model import load_model

import a2h

class Decoder:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "SPADEDecoder",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device
        
    def __call__(self, feature):

        pred = a2h.decode_face( np.ascontiguousarray(feature) )
        
        pred = np.transpose(pred, [1, 2, 0]).clip(0, 1) * 255    # [h, w, c]
        
        return pred
