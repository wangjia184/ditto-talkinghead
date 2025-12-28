"""
Process one video through all steps:
1. Crop video
2. Extract audio
3. Extract audio features
4. Extract motion features (normal + flip)
5. Extract eye features (normal + flip)
6. Extract emotion features
"""
import os
import sys
import traceback
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

import numpy as np
from utils.utils import load_json
from scripts.crop_video_by_LP import init_cropper, crop_one
from scripts.extract_audio_from_video import extract_audio
from scripts.extract_audio_feat_by_Hubert import init_W2F
from scripts.extract_motion_feat_by_LP import init_LP, video_to_motion_pkl, cvt_live_motion_info
from scripts.extract_eye_ratio_from_video import MediaPipeUtils, det_mp_lmks_for_video, lmks_to_eye_attr
from scripts.extract_emo_feat_from_video import EmoRec, process_one_video as process_one_video_emo


@dataclass
class Options:
    fps25_video: str = ""  # input video path
    video: str = ""  # output cropped video path
    wav: str = ""  # output audio path
    hubert_aud_npy: str = ""  # output audio feature path
    LP_pkl: str = ""  # output motion pkl path
    LP_npy: str = ""  # output motion npy path
    LP_pkl_flip: str = ""  # output motion pkl flip path
    LP_npy_flip: str = ""  # output motion npy flip path
    MP_lmk_npy: str = ""  # output landmark path
    eye_open_npy: str = ""  # output eye open path
    eye_ball_npy: str = ""  # output eye ball path
    MP_lmk_npy_flip: str = ""  # output landmark flip path
    eye_open_npy_flip: str = ""  # output eye open flip path
    eye_ball_npy_flip: str = ""  # output eye ball flip path
    emo_npy: str = ""  # output emotion path
    
    ditto_pytorch_path: str = ""
    Hubert_onnx: str = ""
    MP_face_landmarker_task_path: str = ""
    
    skip_existing: bool = True  # skip if output files already exist


def process_one_video_complete(opt: Options):
    """Process one video through all steps"""
    video_name = os.path.basename(opt.fps25_video)
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Initialize models (only once, reuse for all videos)
    if not hasattr(process_one_video_complete, '_cropper'):
        print("Initializing cropper...", flush=True)
        process_one_video_complete._cropper = init_cropper(opt.ditto_pytorch_path)
    
    if not hasattr(process_one_video_complete, '_w2f'):
        print("Initializing HuBERT...", flush=True)
        process_one_video_complete._w2f = init_W2F(opt.Hubert_onnx)
    
    if not hasattr(process_one_video_complete, '_lp'):
        print("Initializing LivePortrait...", flush=True)
        process_one_video_complete._lp = init_LP(opt.ditto_pytorch_path)
    
    if not hasattr(process_one_video_complete, '_mp_utils'):
        print("Initializing MediaPipe...", flush=True)
        process_one_video_complete._mp_utils = MediaPipeUtils(opt.MP_face_landmarker_task_path)
    
    if not hasattr(process_one_video_complete, '_emo_rec'):
        print("Initializing Emotion Recognition...", flush=True)
        process_one_video_complete._emo_rec = EmoRec()
    
    cropper = process_one_video_complete._cropper
    w2f = process_one_video_complete._w2f
    lp = process_one_video_complete._lp
    mp_utils = process_one_video_complete._mp_utils
    emo_rec = process_one_video_complete._emo_rec
    
    # Check if all output files already exist
    if opt.skip_existing:
        all_files_exist = (
            os.path.isfile(opt.video) and
            os.path.isfile(opt.wav) and
            os.path.isfile(opt.hubert_aud_npy) and
            os.path.isfile(opt.LP_pkl) and
            os.path.isfile(opt.LP_npy) and
            os.path.isfile(opt.LP_pkl_flip) and
            os.path.isfile(opt.LP_npy_flip) and
            os.path.isfile(opt.MP_lmk_npy) and
            os.path.isfile(opt.eye_open_npy) and
            os.path.isfile(opt.eye_ball_npy) and
            os.path.isfile(opt.MP_lmk_npy_flip) and
            os.path.isfile(opt.eye_open_npy_flip) and
            os.path.isfile(opt.eye_ball_npy_flip) and
            os.path.isfile(opt.emo_npy)
        )
        
        if all_files_exist:
            print(f"  ✓ All output files already exist, skipping entire video", flush=True)
            return True
    
    success_steps = []
    failed_steps = []
    
    # Step 1: Crop video
    try:
        if opt.skip_existing and os.path.isfile(opt.video):
            print(f"  [1/9] ✓ Video already exists, skipping")
            success_steps.append(1)
        else:
            print(f"  [1/9] Cropping video...", flush=True)
            crop_one(opt.fps25_video, opt.video, cropper)
            if os.path.isfile(opt.video):
                print(f"  [1/9] ✓ Video cropped: {os.path.basename(opt.video)}", flush=True)
                success_steps.append(1)
            else:
                raise Exception(f"Video file not created: {opt.video}")
    except Exception as e:
        print(f"  [1/9] ✗ Failed to crop video: {e}", flush=True)
        failed_steps.append(1)
        traceback.print_exc()
        return False
    
    # Step 2: Extract audio
    try:
        if opt.skip_existing and os.path.isfile(opt.wav):
            print(f"  [2/9] ✓ Audio already exists, skipping")
            success_steps.append(2)
        else:
            print(f"  [2/9] Extracting audio...", flush=True)
            extract_audio(opt.video, opt.wav)
            if os.path.isfile(opt.wav):
                print(f"  [2/9] ✓ Audio extracted: {os.path.basename(opt.wav)}", flush=True)
                success_steps.append(2)
            else:
                raise Exception(f"Audio file not created: {opt.wav}")
    except Exception as e:
        print(f"  [2/9] ✗ Failed to extract audio: {e}", flush=True)
        failed_steps.append(2)
        traceback.print_exc()
    
    # Step 3: Extract audio features
    try:
        if opt.skip_existing and os.path.isfile(opt.hubert_aud_npy):
            print(f"  [3/9] ✓ Audio features already exist, skipping")
            success_steps.append(3)
        else:
            print(f"  [3/9] Extracting audio features...", flush=True)
            os.makedirs(os.path.dirname(opt.hubert_aud_npy), exist_ok=True)
            w2f(opt.wav, opt.hubert_aud_npy)
            if os.path.isfile(opt.hubert_aud_npy):
                print(f"  [3/9] ✓ Audio features extracted: {os.path.basename(opt.hubert_aud_npy)}", flush=True)
                success_steps.append(3)
            else:
                raise Exception(f"Audio features file not created: {opt.hubert_aud_npy}")
    except Exception as e:
        print(f"  [3/9] ✗ Failed to extract audio features: {e}", flush=True)
        failed_steps.append(3)
        traceback.print_exc()
    
    # Step 4a: Extract motion features (normal)
    try:
        if opt.skip_existing and os.path.isfile(opt.LP_pkl) and os.path.isfile(opt.LP_npy):
            print(f"  [4a/9] ✓ Motion features (normal) already exist, skipping")
            success_steps.append(4)
        else:
            print(f"  [4a/9] Extracting motion features (normal)...", flush=True)
            if not os.path.isfile(opt.LP_pkl):
                video_to_motion_pkl(lp, opt.video, opt.LP_pkl, flip=False)
            if not os.path.isfile(opt.LP_npy):
                cvt_live_motion_info(opt.LP_pkl, opt.LP_npy)
            if os.path.isfile(opt.LP_pkl) and os.path.isfile(opt.LP_npy):
                print(f"  [4a/9] ✓ Motion features (normal) extracted", flush=True)
                success_steps.append(4)
            else:
                raise Exception(f"Motion features files not created")
    except Exception as e:
        print(f"  [4a/9] ✗ Failed to extract motion features (normal): {e}", flush=True)
        failed_steps.append(4)
        traceback.print_exc()
    
    # Step 4b: Extract motion features (flip)
    try:
        if opt.skip_existing and os.path.isfile(opt.LP_pkl_flip) and os.path.isfile(opt.LP_npy_flip):
            print(f"  [4b/9] ✓ Motion features (flip) already exist, skipping")
            success_steps.append(5)
        else:
            print(f"  [4b/9] Extracting motion features (flip)...", flush=True)
            if not os.path.isfile(opt.LP_pkl_flip):
                video_to_motion_pkl(lp, opt.video, opt.LP_pkl_flip, flip=True)
            if not os.path.isfile(opt.LP_npy_flip):
                cvt_live_motion_info(opt.LP_pkl_flip, opt.LP_npy_flip)
            if os.path.isfile(opt.LP_pkl_flip) and os.path.isfile(opt.LP_npy_flip):
                print(f"  [4b/9] ✓ Motion features (flip) extracted", flush=True)
                success_steps.append(5)
            else:
                raise Exception(f"Motion features (flip) files not created")
    except Exception as e:
        print(f"  [4b/9] ✗ Failed to extract motion features (flip): {e}", flush=True)
        failed_steps.append(5)
        traceback.print_exc()
    
    # Step 5a: Extract eye features (normal)
    try:
        if opt.skip_existing and os.path.isfile(opt.MP_lmk_npy) and os.path.isfile(opt.eye_open_npy) and os.path.isfile(opt.eye_ball_npy):
            print(f"  [5a/9] ✓ Eye features (normal) already exist, skipping")
            success_steps.append(6)
        else:
            print(f"  [5a/9] Extracting eye features (normal)...", flush=True)
            if not os.path.isfile(opt.MP_lmk_npy):
                lmks = det_mp_lmks_for_video(opt.video, mp_utils, npy=opt.MP_lmk_npy, flip=False)
            else:
                lmks = np.load(opt.MP_lmk_npy)
            lmks_to_eye_attr(lmks, open_npy=opt.eye_open_npy, ball_npy=opt.eye_ball_npy)
            if os.path.isfile(opt.MP_lmk_npy) and os.path.isfile(opt.eye_open_npy) and os.path.isfile(opt.eye_ball_npy):
                print(f"  [5a/9] ✓ Eye features (normal) extracted", flush=True)
                success_steps.append(6)
            else:
                raise Exception(f"Eye features (normal) files not created")
    except Exception as e:
        print(f"  [5a/9] ✗ Failed to extract eye features (normal): {e}", flush=True)
        failed_steps.append(6)
        traceback.print_exc()
    
    # Step 5b: Extract eye features (flip)
    try:
        if opt.skip_existing and os.path.isfile(opt.MP_lmk_npy_flip) and os.path.isfile(opt.eye_open_npy_flip) and os.path.isfile(opt.eye_ball_npy_flip):
            print(f"  [5b/9] ✓ Eye features (flip) already exist, skipping")
            success_steps.append(7)
        else:
            print(f"  [5b/9] Extracting eye features (flip)...", flush=True)
            if not os.path.isfile(opt.MP_lmk_npy_flip):
                lmks = det_mp_lmks_for_video(opt.video, mp_utils, npy=opt.MP_lmk_npy_flip, flip=True)
            else:
                lmks = np.load(opt.MP_lmk_npy_flip)
            lmks_to_eye_attr(lmks, open_npy=opt.eye_open_npy_flip, ball_npy=opt.eye_ball_npy_flip)
            if os.path.isfile(opt.MP_lmk_npy_flip) and os.path.isfile(opt.eye_open_npy_flip) and os.path.isfile(opt.eye_ball_npy_flip):
                print(f"  [5b/9] ✓ Eye features (flip) extracted", flush=True)
                success_steps.append(7)
            else:
                raise Exception(f"Eye features (flip) files not created")
    except Exception as e:
        print(f"  [5b/9] ✗ Failed to extract eye features (flip): {e}", flush=True)
        failed_steps.append(7)
        traceback.print_exc()
    
    # Step 6: Extract emotion features
    try:
        if opt.skip_existing and os.path.isfile(opt.emo_npy):
            print(f"  [6/9] ✓ Emotion features already exist, skipping")
            success_steps.append(8)
        else:
            print(f"  [6/9] Extracting emotion features...", flush=True)
            process_one_video_emo(opt.video, opt.emo_npy, ER=emo_rec)
            if os.path.isfile(opt.emo_npy):
                print(f"  [6/9] ✓ Emotion features extracted: {os.path.basename(opt.emo_npy)}", flush=True)
                success_steps.append(8)
            else:
                raise Exception(f"Emotion features file not created: {opt.emo_npy}")
    except Exception as e:
        print(f"  [6/9] ✗ Failed to extract emotion features: {e}", flush=True)
        failed_steps.append(8)
        traceback.print_exc()
    
    # Summary
    print(f"\n  Summary: {len(success_steps)} steps succeeded, {len(failed_steps)} steps failed")
    if failed_steps:
        print(f"  Failed steps: {failed_steps}")
        return False
    else:
        print(f"  ✓ All steps completed successfully!")
        return True


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    
    if not opt.fps25_video or not os.path.isfile(opt.fps25_video):
        print(f"Error: Input video not found: {opt.fps25_video}")
        sys.exit(1)
    
    success = process_one_video_complete(opt)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

