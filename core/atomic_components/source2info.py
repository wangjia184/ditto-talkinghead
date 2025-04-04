import numpy as np
import os
from PIL import Image
import cv2

#from ..aux_models.insightface_det import InsightFaceDet
from ..aux_models.insightface_landmark106 import Landmark106
from ..aux_models.landmark203 import Landmark203
from ..aux_models.mediapipe_landmark478 import Landmark478
from ..models.appearance_extractor import AppearanceExtractor
from ..models.motion_extractor import MotionExtractor

from ..utils.crop import crop_image
from ..utils.eye_info import EyeAttrUtilsByMP

import a2h

"""
insightface_det_cfg = {
    "model_path": "",
    "device": "cuda",
    "force_ori_type": False,
}
landmark106_cfg = {
    "model_path": "",
    "device": "cuda",
    "force_ori_type": False,
}
landmark203_cfg = {
    "model_path": "",
    "device": "cuda",
    "force_ori_type": False,
}
landmark478_cfg = {
    "blaze_face_model_path": "", 
    "face_mesh_model_path": "", 
    "device": "cuda",
    "force_ori_type": False,
    "task_path": "",
}
appearance_extractor_cfg = {
    "model_path": "",
    "device": "cuda",
}
motion_extractor_cfg = {
    "model_path": "",
    "device": "cuda",
}
"""


def _transform_pts(pts, M):
    """ conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]

def save_hwc_image(
    array: np.ndarray,
    save_path: str,
    *,
    channel_order: str = 'RGB',
    clamp_float: bool = True,
    check_bounds: bool = True
) -> None:
    """
    通用 HWC 图像保存函数（支持 uint8 和 0-255 float）
    
    参数:
        array (np.ndarray): 输入数组 [H, W, C]
        save_path (str): 保存路径
        channel_order (str): 通道顺序 (RGB/BGR)
        clamp_float (bool): float类型时自动截断到0-255
        check_bounds (bool): 检查float数值是否在有效范围
    
    支持:
        - 数据类型: uint8 或 float32/64 (0-255范围)
        - 通道数: 1 (灰度), 3 (RGB/RGBA), 4 (RGBA)
    """
    # ================= 基础验证 =================
    if not isinstance(array, np.ndarray):
        raise TypeError(f"需要numpy数组，输入类型: {type(array)}")
        
    if array.ndim != 3:
        raise ValueError(f"需要HWC格式输入，当前维度: {array.ndim}D")
        
    h, w, c = array.shape
    if c not in (1, 3, 4):
        raise ValueError(f"无效通道数 {c}，支持: 1/3/4")

    # ================= 类型处理 =================
    if array.dtype == np.uint8:
        # uint8 直接处理
        img_array = array
    elif np.issubdtype(array.dtype, np.floating):
        # float 类型处理流程
        if check_bounds:
            min_val = np.min(array)
            max_val = np.max(array)
            if min_val < 0 or max_val > 255:
                if clamp_float:
                    array = np.clip(array, 0, 255)
                else:
                    raise ValueError(
                        f"数值越界: min={min_val:.2f}, max={max_val:.2f}\n"
                        "使用 clamp_float=True 自动截断"
                    )
        
        # 优化过的类型转换 (避免复制)
        img_array = array.astype(np.uint8, copy=False)
    else:
        raise TypeError(f"不支持的dtype: {array.dtype}")

    # ================= 通道处理 =================
    if c in (3, 4) and channel_order.upper() == 'BGR':
        # BGR转换 (原地操作优化)
        img_array = img_array[..., ::-1] if c == 3 else img_array[..., [2,1,0,3]]

    # ================= 保存处理 =================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        if c == 1:
            Image.fromarray(img_array.squeeze(axis=2), 'L').save(save_path)
        else:
            mode = 'RGB' if c == 3 else 'RGBA'
            Image.fromarray(img_array, mode).save(save_path)
    except Exception as e:
        raise RuntimeError(f"图像保存失败: {str(e)}")

class Source2Info:
    def __init__(
        self,
        insightface_det_cfg,
        landmark106_cfg,
        landmark203_cfg,
        landmark478_cfg,
        appearance_extractor_cfg,
        motion_extractor_cfg,
    ):
        #self.insightface_det = InsightFaceDet(**insightface_det_cfg)
        self.landmark106 = Landmark106(**landmark106_cfg)
        self.landmark203 = Landmark203(**landmark203_cfg)
        self.landmark478 = Landmark478(**landmark478_cfg)

        self.appearance_extractor = AppearanceExtractor(**appearance_extractor_cfg)
        self.motion_extractor = MotionExtractor(**motion_extractor_cfg)

    def _crop(self, img, last_lmk=None, **kwargs):
        """
        # img_rgb -> det->landmark106->landmark203->crop
        if last_lmk is None:  # det for first frame or image
            det, _ = a2h.insightface_detect(img)
            # [[578.951      308.7893     659.48627    419.8099       0.81669873]]
            boxes = det[np.argsort(-(det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))]
            if len(boxes) == 0:
                return None
            #lmk_for_track = self.landmark106(img, boxes[0])  # 106
            lmk_for_track = a2h.detect_landmark106(img, boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
        else:  # track for video frames
            lmk_for_track = last_lmk  # 203
        """
        """
        lmk_for_track = a2h.extract_face(img)
        center, cropping_size, angle = a2h.parse_face_cropping_area( lmk_for_track, 1.5, 0.0, -0.1, False, True)
        cropped_img, o2c, c2o = a2h.crop_face( img, center, cropping_size, self.landmark203.dsize, angle)
        #print( "o2c=", o2c )
        #print( "c2o=", c2o )
        

        crop_dct = crop_image(
            img,
            lmk_for_track,
            dsize=self.landmark203.dsize,
            scale=1.5,
            vy_ratio=-0.1,
            pt_crop_flag=False,
        )
        #save_hwc_image( cropped_img, "./cropped_img.png")
        #save_hwc_image( crop_dct["img_crop"], "./img_crop.png")
        #lmk203 = self.landmark203(crop_dct["img_crop"], c2o)
        lmk203 = a2h.detect_landmark203(cropped_img)
        lmk203 = _transform_pts(lmk203, c2o)
        #print(lmk203)
        """
        cropped_img, inv = a2h.crop_face(img)
        return cropped_img, inv, None

        center, cropping_size, angle = a2h.parse_face_cropping_area( lmk203, kwargs.get("crop_scale", 2.3), kwargs.get("crop_vx_ratio", 0), kwargs.get("crop_vy_ratio", -0.125), False, True)
        cropped_img, o2c, c2o = a2h.crop_face( img, center, cropping_size, 512, angle)

        ret_dct = crop_image(
            img,
            lmk203,
            dsize=512,
            scale=kwargs.get("crop_scale", 2.3),
            vx_ratio=kwargs.get("crop_vx_ratio", 0),
            vy_ratio=kwargs.get("crop_vy_ratio", -0.125),
            flag_do_rot=kwargs.get("crop_flag_do_rot", True),
            pt_crop_flag=False,
        )

        save_hwc_image( cropped_img, "./cropped_img.png")
        save_hwc_image( ret_dct["img_crop"], "./img_crop.png")

        img_crop = ret_dct["img_crop"]
        M_c2o = ret_dct["M_c2o"]

        return cropped_img, c2o, lmk203
    
    @staticmethod
    def _img_crop_to_bchw256(img_crop):
        rgb_256 = cv2.resize(img_crop, (256, 256), interpolation=cv2.INTER_AREA)
        rgb_256_bchw = (rgb_256.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)
        return rgb_256_bchw

    def _get_kp_info(self, img):
        # rgb_256_bchw_norm01
        kp_info = self.motion_extractor(img)
        return kp_info

    def _get_f3d(self, img):
        # rgb_256_bchw_norm01
        fs = self.appearance_extractor(img)
        return fs

    def _get_eye_info(self, img):
        # rgb uint8
        lmk478 = self.landmark478(img)  # [1, 478, 3]
        attr = EyeAttrUtilsByMP(lmk478)
        lr_open = attr.LR_open().reshape(-1, 2)   # [1, 2]
        lr_ball = attr.LR_ball_move().reshape(-1, 6)   # [1, 3, 2] -> [1, 6]
        return [lr_open, lr_ball]

    def __call__(self, img, last_lmk=None, **kwargs):
        """
        img: rgb, uint8
        last_lmk: last frame lmk203, for video tracking
        kwargs: optional crop cfg
            crop_scale: 2.3
            crop_vx_ratio: 0
            crop_vy_ratio: -0.125
            crop_flag_do_rot: True
        """
        #img_crop, M_c2o, lmk203 = self._crop(img, last_lmk=last_lmk, **kwargs)
        img_crop, M_c2o = a2h.crop_face(img)

        eye_open, eye_ball = self._get_eye_info(img_crop)
        #print( "eye_open", eye_open)
        #print( "eye_ball", eye_ball)

        rgb_256_bchw = self._img_crop_to_bchw256(img_crop)
        kp_info = self._get_kp_info(rgb_256_bchw) #motion_extractor
        fs = self._get_f3d(rgb_256_bchw) # appearance_extractor
        eye_open = np.array([[1.0678004, 1.083962 ]])
        eye_ball = np.array([[ 0.00860407, -0.01075477, -0.0042886,  -0.0055787,  -0.01107246, -0.01152826]])
        source_info = {
            "x_s_info": kp_info,
            "f_s": fs,  # (1, 32, 16, 64, 64)
            "M_c2o": M_c2o, 
            "eye_open": eye_open,   # [1, 2]
            "eye_ball": eye_ball,    # [1, 6]
            "lmk203": None,  # for track
        }

        xxx = a2h.extract_face(img)

        print("x_s_info", xxx["x_s_info"])
        print("f_s", xxx["f_s"].shape)
        print("M_c2o", xxx["M_c2o"].shape, M_c2o.shape)
        print("eye_open", xxx["eye_open"].shape)
        print("eye_ball", xxx["eye_ball"].shape) 

        return source_info
