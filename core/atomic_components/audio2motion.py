import numpy as np
from ..models.lmdm import LMDM

import a2h

"""
lmdm_cfg = {
    "model_path": "",
    "device": "cuda",
    "motion_feat_dim": 265,
    "audio_feat_dim": 1024+35,
    "seq_frames": 80,
}
"""


def _cvt_LP_motion_info(inp, mode, ignore_keys=()):
    ks_shape_map = [
        ['scale', (1, 1), 1], 
        ['pitch', (1, 66), 66],
        ['yaw',   (1, 66), 66],
        ['roll',  (1, 66), 66],
        ['t',     (1, 3), 3], 
        ['exp', (1, 63), 63],
        ['kp',  (1, 63), 63],
    ]
    
    def _dic2arr(_dic):
        arr = []
        for k, _, ds in ks_shape_map:
            if k not in _dic or k in ignore_keys:
                continue
            v = _dic[k].reshape(ds)
            if k == 'scale':
                v = v - 1
            arr.append(v)
        arr = np.concatenate(arr, -1)  # (133)
        return arr
    
    def _arr2dic(_arr):
        dic = {}
        s = 0
        for k, ds, ss in ks_shape_map:
            if k in ignore_keys:
                continue
            v = _arr[s:s + ss].reshape(ds)
            if k == 'scale':
                v = v + 1
            dic[k] = v
            s += ss
            if s >= len(_arr):
                break
        return dic
    
    if mode == 'dic2arr':
        assert isinstance(inp, dict)
        return _dic2arr(inp)   # (dim)
    elif mode == 'arr2dic':
        assert inp.shape[0] >= 265, f"{inp.shape}"
        return _arr2dic(inp)   # {k: (1, dim)}
    else:
        raise ValueError()
    

class Audio2Motion:
    def __init__(
        self,
        lmdm_cfg,
    ):
        self.lmdm = LMDM(**lmdm_cfg)

    def setup(
        self, 
        x_s_info, 
        overlap_v2=70,
        fix_kp_cond=0,
        fix_kp_cond_dim=None,
        sampling_timesteps=50,
        online_mode=False,
        v_min_max_for_clip=None,
        smo_k_d=3,
    ):
        self.smo_k_d = smo_k_d  #3
        self.overlap_v2 = overlap_v2 #70
        self.seq_frames = 80
        self.valid_clip_len = self.seq_frames - self.overlap_v2 #10

        self.online_mode = online_mode

        self.fuse_length = min(self.overlap_v2, self.valid_clip_len) #10
        self.fuse_alpha = np.arange(self.fuse_length, dtype=np.float32).reshape(1, -1, 1) / self.fuse_length

        self.fix_kp_cond = fix_kp_cond  #1
        self.fix_kp_cond_dim = fix_kp_cond_dim # [0, 202]
        self.sampling_timesteps = sampling_timesteps #10
        
        self.v_min_max_for_clip = v_min_max_for_clip
        if self.v_min_max_for_clip is not None:
            self.v_min = self.v_min_max_for_clip[0][None]    # [dim, 1]
            self.v_max = self.v_min_max_for_clip[1][None]

        kp_source = _cvt_LP_motion_info(x_s_info, mode='dic2arr', ignore_keys={'kp'})[None]
        self.s_kp_cond = kp_source.copy().reshape(1, -1)
        self.kp_cond = self.s_kp_cond.copy()

        self.lmdm.setup(sampling_timesteps)

        self.clip_idx = 0


        print("overlap_v2 : ", overlap_v2 )
        print("smo_k_d : ", self.smo_k_d)
        print("seq_frames : ", self.seq_frames)
        print("fix_kp_cond : ", self.fix_kp_cond)
        print("fix_kp_cond_dim : ", self.fix_kp_cond_dim)
        print("v_min_max_for_clip : ", self.v_min_max_for_clip)
        print("fuse_length : ", self.fuse_length )
        print("fuse_alpha : ", self.fuse_alpha )
        print("valid_clip_len : ", self.valid_clip_len )

    def _fuse(self, res_kp_seq, pred_kp_seq):
        ## ========================
        ## online fuse mode
        ## last clip:  -------
        ## fuse part:       **
        ## curr clip:    -------
        ## output:          ^^
        ## ========================
        #out = a2h.fuse( res_kp_seq, pred_kp_seq, self.seq_frames, self.valid_clip_len, self.fuse_length)

        fuse_r1_s = res_kp_seq.shape[1] - self.fuse_length
        fuse_r1_e = res_kp_seq.shape[1]
        fuse_r2_s = self.seq_frames - self.valid_clip_len - self.fuse_length
        fuse_r2_e = self.seq_frames - self.valid_clip_len

        r1 = res_kp_seq[:, fuse_r1_s:fuse_r1_e]     # [1, fuse_len, dim]
        r2 = pred_kp_seq[:, fuse_r2_s: fuse_r2_e]   # [1, fuse_len, dim]
        r_fuse = r1 * (1 - self.fuse_alpha) + r2 * self.fuse_alpha

        res_kp_seq[:, fuse_r1_s:fuse_r1_e] = r_fuse    # fuse last
        res_kp_seq = np.concatenate([res_kp_seq, pred_kp_seq[:, fuse_r2_e:]], 1)  # len(res_kp_seq) + valid_clip_len

        #if out.shape == res_kp_seq.shape:
        #    print( np.mean((res_kp_seq - out) ** 2))
        #else:
        #    print( res_kp_seq.shape, out.shape )

        return res_kp_seq
    
    def _update_kp_cond(self, res_kp_seq, idx):
        if self.clip_idx % self.fix_kp_cond == 0:  # self.fix_kp_cond == 1 here
            self.kp_cond = self.s_kp_cond.copy()  # 重置所有
            if self.fix_kp_cond_dim is not None:
                ds, de = self.fix_kp_cond_dim
                self.kp_cond[:, ds:de] = res_kp_seq[:, idx-1, ds:de]
        else:
            self.kp_cond = res_kp_seq[:, idx-1]
            

    def _smo(self, res_kp_seq, s, e):
        if self.smo_k_d <= 1:
            return res_kp_seq
        
        new_res_kp_seq = res_kp_seq.copy()
        n = res_kp_seq.shape[1]
        half_k = self.smo_k_d // 2
        for i in range(s, e):
            ss = max(0, i - half_k)
            ee = min(n, i + half_k + 1)
            res_kp_seq[:, i, :202] = np.mean(new_res_kp_seq[:, ss:ee, :202], axis=1)

    
            
        return res_kp_seq
    
    def __call__(self, aud_cond, res_kp_seq=None):
        """
        aud_cond: (1, seq_frames, dim)
        """
        """
        #pred_kp_seq = self.lmdm(self.kp_cond, aud_cond, self.sampling_timesteps)
        assert self.sampling_timesteps == 10
        pred_kp_seq = a2h.predict_motion(self.kp_cond, aud_cond)
        if res_kp_seq is None:
            res_kp_seq = pred_kp_seq   # [1, seq_frames, dim]
            res_kp_seq =  a2h.moving_average_smooth(res_kp_seq, 0, res_kp_seq.shape[1], self.smo_k_d)
        else:
            # len(res_kp_seq) + valid_clip_len
            res_kp_seq = a2h.fuse(res_kp_seq, pred_kp_seq, self.valid_clip_len, self.fuse_length)
            res_kp_seq =  a2h.moving_average_smooth(res_kp_seq, res_kp_seq.shape[1] - self.valid_clip_len - self.fuse_length, res_kp_seq.shape[1] - self.valid_clip_len + 1, self.smo_k_d)
        """
        if res_kp_seq is None:
            res_kp_seq = np.zeros((1, 0, 265), dtype=np.float32) 
        res_kp_seq = a2h.audio2motion( self.s_kp_cond, np.ascontiguousarray(aud_cond), np.ascontiguousarray(res_kp_seq) )
        #self.clip_idx += 1

        #idx = res_kp_seq.shape[1] - self.overlap_v2
        #self._update_kp_cond(res_kp_seq, idx)

        return res_kp_seq
    
    def cvt_fmt(self, res_kp_seq):
        # res_kp_seq: [1, n, dim]
        if self.v_min_max_for_clip is not None:
            tmp_res_kp_seq = np.clip(res_kp_seq[0], self.v_min, self.v_max)
        else:
            tmp_res_kp_seq = res_kp_seq[0]

        x_d_info_list = []
        for i in range(tmp_res_kp_seq.shape[0]):
            x_d_info = _cvt_LP_motion_info(tmp_res_kp_seq[i], 'arr2dic')   # {k: (1, dim)}
            x_d_info_list.append(x_d_info)
        return x_d_info_list
