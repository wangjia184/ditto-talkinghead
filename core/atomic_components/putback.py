import cv2
import numpy as np
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask

import a2h

class PutBackNumpy:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            self.mask_ori_float = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.mask_ori_float = mask.astype(np.float32) / 255.0

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        result = mask_warped * frame_warped + (1 - mask_warped) * frame_rgb
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result
    

def save_mask_as_grayscale(mask, output_path):
    # 1. 归一化到 [0, 255]
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # 2. 如果是三通道，取第一个通道（假设所有通道相同）
    if len(mask_uint8.shape) == 3:
        mask_uint8 = mask_uint8[..., 0]  # 取单通道
    
    # 3. 保存为灰度图
    cv2.imwrite(output_path, mask_uint8)


class PutBack:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9) # (512, 512, 1)
            #save_mask_as_grayscale( mask, "./mask.png")
            mask = np.concatenate([mask] * 3, 2) # (512, 512, 3)
            print("np.concatenate", mask.shape, mask.dtype)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:,:,0] # (512, 512)
        print("self.mask_ori_float", self.mask_ori_float.shape, self.mask_ori_float.dtype)
        self.result_buffer = None

    def __call__(self, frame_rgb, render_image, M_c2o):
        # frame_rgb [ H, W, 3 ]  range 0-255
        # render_image [H, W, 3] range 0-255
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Use Cython implementation for blending
        blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)

        return self.result_buffer
    