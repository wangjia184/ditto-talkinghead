import copy
import random
import numpy as np
from scipy.special import softmax

from ..models.stitch_network import StitchNetwork
import a2h

"""
# __init__
stitch_network_cfg = {
    "model_path": "",
    "device": "cuda",
}

# __call__
kwargs:
    fade_alpha
    fade_out_keys

    delta_pitch
    delta_yaw
    delta_roll

"""


def ctrl_motion(x_d_info, **kwargs):
    # pose + offset
    for kk in ["delta_pitch", "delta_yaw", "delta_roll"]:
        if kk in kwargs:
            k = kk[6:]
            x_d_info[k] = bin66_to_degree(x_d_info[k]) + kwargs[kk]

    # pose * alpha
    for kk in ["alpha_pitch", "alpha_yaw", "alpha_roll"]:
        if kk in kwargs:
            k = kk[6:]
            x_d_info[k] = x_d_info[k] * kwargs[kk]

    # exp + offset
    if "delta_exp" in kwargs:
        k = "exp"
        x_d_info[k] = x_d_info[k] + kwargs["delta_exp"]

    return x_d_info


def fade(x_d_info, dst, alpha, keys=None):
    if keys is None:
        keys = x_d_info.keys()
    for k in keys:
        if k == 'kp':
            continue
        x_d_info[k] = x_d_info[k] * alpha + dst[k] * (1 - alpha)
    return x_d_info


def ctrl_vad(x_d_info, dst, alpha):
    exp = x_d_info["exp"]
    exp_dst = dst["exp"]

    _lip = [6, 12, 14, 17, 19, 20]
    _a1 = np.zeros((21, 3), dtype=np.float32)
    _a1[_lip] = alpha
    _a1 = _a1.reshape(1, -1)
    x_d_info["exp"] = exp * alpha + exp_dst * (1 - alpha)

    return x_d_info
    
    

def _mix_s_d_info(
    x_s_info,
    x_d_info,
    use_d_keys=("exp", "pitch", "yaw", "roll", "t"),
    d0=None
):
    #if d0 is not None:
    #    if isinstance(use_d_keys, dict):
    #        x_d_info = {
    #            k: x_s_info[k] + (v - d0[k]) * use_d_keys.get(k, 1)
    #            for k, v in x_d_info.items()
    #        }
    #        print(use_d_keys, use_d_keys.get('exp', 1))
    #    else:
    #        x_d_info = {k: x_s_info[k] + (v - d0[k]) for k, v in x_d_info.items()}

    #for k, v in x_s_info.items():
    #    if k not in x_d_info or k not in use_d_keys:
    #        x_d_info[k] = v
    
    x_d_info = {
        #"exp" : x_s_info["exp"] + (x_d_info["exp"] - d0["exp"]) * 0.927,
        "exp" :  x_d_info["exp"],
        "pitch" :  x_d_info["pitch"],
        "yaw" :  x_d_info["yaw"],
        "roll" :  x_d_info["roll"],
        "t" :  x_d_info["t"],
        "scale" :  x_d_info["scale"],
        #"pitch" : x_s_info["pitch"] + (x_d_info["pitch"] - d0["pitch"]) * use_d_keys.get("pitch", 1),
        #"yaw" : x_s_info["yaw"] + (x_d_info["yaw"] - d0["yaw"]) * use_d_keys.get("yaw", 1),
        #"roll" : x_s_info["roll"] + (x_d_info["roll"] - d0["roll"]) * use_d_keys.get("roll", 1),
        #"t" : x_s_info["t"] + (x_d_info["t"] - d0["t"]),
        #"kp" : x_s_info["kp"],
        #"scale" : x_s_info["scale"],
    }
  
    #if isinstance(use_d_keys, dict) and d0 is None:
    #    for k, alpha in use_d_keys.items():
    #        x_d_info[k] *= alpha
    return x_d_info


def _set_eye_blink_idx(N, blink_n=15, open_n=-1):
    """
    open_n:
        -1: no blink
        0: random open_n
        >0: fix open_n
        list: loop open_n
    """
    OPEN_MIN = 60
    OPEN_MAX = 100

    idx = [0] * N
    if isinstance(open_n, int):
        if open_n < 0:  # no blink
            return idx
        elif open_n > 0:  # fix open_n
            open_ns = [open_n]
        else:  # open_n == 0:  # random open_n, 60-100
            open_ns = []
    elif isinstance(open_n, list):
        open_ns = open_n  # loop open_n
    else:
        raise ValueError()

    blink_idx = list(range(blink_n))

    start_n = open_ns[0] if open_ns else random.randint(OPEN_MIN, OPEN_MAX)
    end_n = open_ns[-1] if open_ns else random.randint(OPEN_MIN, OPEN_MAX)
    max_i = N - max(end_n, blink_n)
    cur_i = start_n
    cur_n_i = 1
    while cur_i < max_i:
        idx[cur_i : cur_i + blink_n] = blink_idx

        if open_ns:
            cur_n = open_ns[cur_n_i % len(open_ns)]
            cur_n_i += 1
        else:
            cur_n = random.randint(OPEN_MIN, OPEN_MAX)

        cur_i = cur_i + blink_n + cur_n

    return idx


def _fix_exp_for_x_d_info(x_d_info, x_s_info, delta_eye=None, drive_eye=True):
    _eye = [11, 13, 15, 16, 18]
    _lip = [6, 12, 14, 17, 19, 20]
    alpha = np.zeros((21, 3), dtype=x_d_info["exp"].dtype)
    alpha[_lip] = 1
    if delta_eye is None and drive_eye:  # use d eye
        alpha[_eye] = 1
    alpha = alpha.reshape(1, -1)
    x_d_info["exp"] = x_d_info["exp"] * alpha + x_s_info["exp"] * (1 - alpha)

    if delta_eye is not None and drive_eye:
        alpha = np.zeros((21, 3), dtype=x_d_info["exp"].dtype)
        alpha[_eye] = 1
        alpha = alpha.reshape(1, -1)
        x_d_info["exp"] = (delta_eye + x_s_info["exp"]) * alpha + x_d_info["exp"] * (
            1 - alpha
        )

    return x_d_info


def _fix_exp_for_x_d_info_v2(x_d_info, x_s_info, delta_eye, a1, a2, a3):
    print( a1, a2)
    x_d_info["exp"] = x_d_info["exp"] * a1 + x_s_info["exp"] * a2 + delta_eye * a3
    return x_d_info


def bin66_to_degree(pred):
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx = np.arange(66).astype(np.float32)
        pred = softmax(pred, axis=1)
        degree = np.sum(pred * idx, axis=1) * 3 - 97.5
        #degree = a2h.bin66_to_degree(pred)
        #print(degree)
        return degree
    return pred


def _eye_delta(exp, dx=0, dy=0):
    if dx > 0:
        exp[0, 33] += dx * 0.0007
        exp[0, 45] += dx * 0.001
    else:
        exp[0, 33] += dx * 0.001
        exp[0, 45] += dx * 0.0007

    exp[0, 34] += dy * -0.001
    exp[0, 46] += dy * -0.001
    return exp

def _fix_gaze(pose_s, x_d_info):
    x_ratio = 0.26
    y_ratio = 0.28
    
    yaw_s, pitch_s = pose_s
    yaw_d = bin66_to_degree(x_d_info['yaw']).item()
    pitch_d = bin66_to_degree(x_d_info['pitch']).item()

    delta_yaw = yaw_d - yaw_s
    delta_pitch = pitch_d - pitch_s

    dx = delta_yaw * x_ratio
    dy = delta_pitch * y_ratio
    
    x_d_info['exp'] = _eye_delta(x_d_info['exp'], dx, dy)
    return x_d_info


def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    return a2h.construct_rotation_matrix( pitch_, yaw_, roll_)



def convert_to_rust_array(delta_eye_arr):
    # 定义格式字符串
    fmt = lambda x: f"{x:.8e}f32".replace("e-0", "e-").replace("e+0", "e+")

    # 逐层构建数组字符串
    rust_array = "pub static DELTA_EYE_ARR: [[[f32; 3]; 21]; 15] = [\n"
    
    # 遍历第一维度 (15)
    for i in range(delta_eye_arr.shape[0]):
        rust_array += "    [\n"
        
        # 遍历第二维度 (21)
        for j in range(delta_eye_arr.shape[1]):
            elements = ", ".join([fmt(x) for x in delta_eye_arr[i, j]])
            rust_array += f"        [{elements}],\n"
        
        rust_array += "    ],\n"
    
    rust_array += "];"
    return rust_array


class MotionStitch:
    def __init__(
        self,
        stitch_network_cfg,
    ):
        self.stitch_net = StitchNetwork(**stitch_network_cfg)

    def set_Nd(self, N_d=-1):
        # only for offline (make start|end eye open)
        if N_d == self.N_d:
            return
        
        self.N_d = N_d
        if self.drive_eye and self.delta_eye_arr is not None:
            N = 3000 if self.N_d == -1 else self.N_d
            self.delta_eye_idx_list = _set_eye_blink_idx(
                N, len(self.delta_eye_arr), self.delta_eye_open_n
            )

    def setup(
        self,
        N_d=-1,
        use_d_keys=None,
        relative_d=True,
        drive_eye=None,  # use d eye or s eye
        delta_eye_arr=None,  # fix eye
        delta_eye_open_n=-1,  # int|list
        fade_out_keys=("exp",),
        fade_type="",   # "" | "d0" | "s"
        flag_stitching=True,
        is_image_flag=True,
        x_s_info=None,
        d0=None,
        ch_info=None,
        overall_ctrl_info=None,
    ):
        self.is_image_flag = is_image_flag
        if use_d_keys is None:
            if self.is_image_flag:
                self.use_d_keys = ("exp", "pitch", "yaw", "roll", "t")
            else:
                self.use_d_keys = ("exp", )
        else:
            self.use_d_keys = use_d_keys

        print( "self.is_image_flag", self.is_image_flag)
        print( "self.use_d_keys", self.use_d_keys)

        if drive_eye is None:
            if self.is_image_flag:
                self.drive_eye = True
            else:
                self.drive_eye = False
        else:
            self.drive_eye = drive_eye

        print( "self.drive_eye", self.drive_eye)


        self.N_d = N_d
        self.relative_d = relative_d
        self.delta_eye_arr = delta_eye_arr
        self.delta_eye_open_n = delta_eye_open_n
        self.fade_out_keys = fade_out_keys
        self.fade_type = fade_type
        self.flag_stitching = flag_stitching

        print( "self.relative_d", self.relative_d)
        
        print( "self.delta_eye_open_n", self.delta_eye_open_n)
        print( "self.fade_out_keys", self.fade_out_keys)
        print( "self.fade_type", self.fade_type)
        print( "self.flag_stitching", self.flag_stitching)


        #print( "self.delta_eye_arr", self.delta_eye_arr.reshape(15, 21, 3))


        _eye = [11, 13, 15, 16, 18]
        _lip = [6, 12, 14, 17, 19, 20]
        _a1 = np.zeros((21, 3), dtype=np.float32)
        _a1[_lip] = 1
        _a2 = 0
        if self.drive_eye:
            if self.delta_eye_arr is None:
                _a1[_eye] = 1
            else:
                _a2 = np.zeros((21, 3), dtype=np.float32)
                _a2[_eye] = 1
                _a2 = _a2.reshape(1, -1)
        _a1 = _a1.reshape(1, -1)

        self.fix_exp_a1 = _a1 * (1 - _a2)
        self.fix_exp_a2 = (1 - _a1) + _a1 * _a2
        self.fix_exp_a3 = _a2

        print( "self.fix_exp_a1", self.fix_exp_a1.shape)
        print( "self.fix_exp_a2", self.fix_exp_a2.shape)
        print( "self.fix_exp_a3", self.fix_exp_a3.shape)

        if self.drive_eye and self.delta_eye_arr is not None:
            N = 3000 if self.N_d == -1 else self.N_d
            self.delta_eye_idx_list = _set_eye_blink_idx(
                N, len(self.delta_eye_arr), self.delta_eye_open_n
            )

        self.pose_s = None
        self.x_s = None
        self.fade_dst = None
        if self.is_image_flag and x_s_info is not None:
            yaw_s = bin66_to_degree(x_s_info['yaw']).item()
            pitch_s = bin66_to_degree(x_s_info['pitch']).item()
            self.pose_s = [yaw_s, pitch_s]
            #self.x_s = a2h.compute_implicit_keypoints(x_s_info)

            if self.fade_type == "s":
                self.fade_dst = copy.deepcopy(x_s_info)

        if ch_info is not None:
            self.scale_a = ch_info['x_s_info_lst'][0]['scale'].item()
            if x_s_info is not None:
                self.scale_b = x_s_info['scale'].item()
                self.scale_ratio = self.scale_a / self.scale_b
                self._set_scale_ratio(self.scale_ratio)
            else:
                self.scale_ratio = None
        else:
            self.scale_ratio = 1

        print( "self.scale_a", self.scale_a)

        self.overall_ctrl_info = overall_ctrl_info

        print("d0", d0)
        #self.d0 = d0
        self.idx = 0

    def _set_scale_ratio(self, scale_ratio=1):
        if scale_ratio == 1:
            return
        if isinstance(self.use_d_keys, dict):
            self.use_d_keys = {k: v * (scale_ratio if k in {"exp", "pitch", "yaw", "roll"} else 1) for k, v in self.use_d_keys.items()}
        else:
            self.use_d_keys = {k: scale_ratio if k in {"exp", "pitch", "yaw", "roll"} else 1 for k in self.use_d_keys}

    @staticmethod
    def _merge_kwargs(default_kwargs, run_kwargs):
        if default_kwargs is None:
            return run_kwargs
        
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs
    
    def __call__(self, x_s_info, x_d_info, **kwargs):
        
        """
        Compute implicit keypoints using these two fomular:

        x_s = s_s·(x_{c,s}R_s + δ_s) + t_s
        x_d = s_d·(x_{c,s}R_d + δ_d) + t_d

        * s_s : scale of source. x_s_info.scale
        * s_d : scale of driving. x_d_info.scale
        * x_{c,s} : 21 canonical keypoints. x_s_info.kp 
        * R_s : rotation matrix of source. generated from x_s_info.pitch/yaw/roll
        * R_d : rotation matrix of driving. generated from x_d_info.pitch/yaw/roll
        * δ_s : x_s_info.exp
        * δ_d : x_s_info.exp
        * t_s : x_s_info.t
        * t_d : x_d_info.t

        x_s_info : 
           * scale : (1, 1)
           * pitch : (1, 66)
           * yaw : (1, 66)
           * roll : (1, 66)
           * t : (1, 3)
           * exp : (1, 63)
           * kp : (1, 63)

        x_d_info : 
           * scale : (1, 1)
           * pitch : (1, 66)
           * yaw : (1, 66)
           * roll : (1, 66)
           * t : (1, 3)
           * exp : (1, 63)
        """
        #print(self.overall_ctrl_info)
        #kwargs = self._merge_kwargs(self.overall_ctrl_info, kwargs)

        
        # scale
        if self.scale_ratio is None:
            self.scale_b = x_s_info['scale'].item()
            self.scale_ratio = self.scale_a / self.scale_b
            self._set_scale_ratio(self.scale_ratio)

        if self.relative_d and self.d0 is None:
            self.d0 = copy.deepcopy(x_d_info)
            print("self.d0[exp]", self.d0["exp"])
            print("x_d_info[exp]", x_d_info["exp"])


        """
        x_d_info = _mix_s_d_info(
            x_s_info,
            x_d_info,
            self.use_d_keys,
            self.d0,
        )
        """

        delta_eye = 0



        

        if self.drive_eye and self.delta_eye_arr is not None:
            delta_eye = self.delta_eye_arr[
                self.delta_eye_idx_list[self.idx % len(self.delta_eye_idx_list)]
            ][None]


        """
        x_d_info = _fix_exp_for_x_d_info_v2(
            x_d_info,
            x_s_info,
            delta_eye,
            self.fix_exp_a1,
            self.fix_exp_a2,
            self.fix_exp_a3,
        )
        """
        
        return a2h.stitch_motion(x_s_info, self.d0, x_d_info)
        """
        if kwargs.get("vad_alpha", 1) < 1:
            x_d_info = ctrl_vad(x_d_info, x_s_info, kwargs.get("vad_alpha", 1))
        """
        #x_d_info = ctrl_motion(x_d_info, **kwargs)

        """
        if self.fade_type == "d0" and self.fade_dst is None:
            self.fade_dst = copy.deepcopy(x_d_info)

        
        # fade
        if "fade_alpha" in kwargs and self.fade_type in ["d0", "s"]:
            fade_alpha = kwargs["fade_alpha"]
            fade_keys = kwargs.get("fade_out_keys", self.fade_out_keys)
            if self.fade_type == "d0":
                fade_dst = self.fade_dst
            elif self.fade_type == "s":
                if self.fade_dst is not None:
                    fade_dst = self.fade_dst
                else:
                    fade_dst = copy.deepcopy(x_s_info)
                    if self.is_image_flag:
                        self.fade_dst = fade_dst
            x_d_info = fade(x_d_info, fade_dst, fade_alpha, fade_keys)
        """
        """
        if self.drive_eye:
            if self.pose_s is None:
                yaw_s = bin66_to_degree(x_s_info['yaw']).item()
                pitch_s = bin66_to_degree(x_s_info['pitch']).item()
                self.pose_s = [yaw_s, pitch_s]
            x_d_info = _fix_gaze(self.pose_s, x_d_info)
        """

        #print( a2h.compute_implicit_keypoints(x_s_info) )

        if self.x_s is not None:
            x_s = self.x_s
        else:
            x_s = a2h.compute_implicit_keypoints(x_s_info)
            if self.is_image_flag:
                self.x_s = x_s
        
        x_d = a2h.compute_implicit_keypoints(x_d_info)
        
        if self.flag_stitching:
            x_d = self.stitch_net(x_s, x_d)

        self.idx += 1

        return x_s, x_d
