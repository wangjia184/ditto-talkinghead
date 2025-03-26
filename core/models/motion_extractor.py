import numpy as np
import torch
from ..utils.load_model import load_model

import a2h

class MotionExtractor:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "MotionExtractor",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

        self.output_names = [
            "pitch",
            "yaw",
            "roll",
            "t",
            "exp",
            "scale",
            "kp",
        ]

    def __call__(self, image):
        """
        image: np.ndarray, shape (1, 3, 256, 256), RGB, 0-1
        """
        outputs = {}
        net_outputs = a2h.extract_motion( np.ascontiguousarray(image) )
        outputs["pitch"] = net_outputs[0]
        outputs["yaw"] = net_outputs[1]
        outputs["roll"] = net_outputs[2]
        outputs["t"] = net_outputs[3]
        outputs["exp"] = net_outputs[4]
        outputs["scale"] = net_outputs[5]
        outputs["kp"] = net_outputs[6]
        outputs["exp"] = outputs["exp"].reshape(1, -1)
        outputs["kp"] = outputs["kp"].reshape(1, -1)
        return outputs


