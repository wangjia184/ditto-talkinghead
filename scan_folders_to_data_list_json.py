#!/usr/bin/env python3
"""
Scan processed data folder(s) and generate training JSON.
Output matches prepare_data Step 7: data_list.json (81-frame filtered, all inputs per sample).

Each output entry: {mtn, aud, frame_num, emo?, eye_open?, eye_ball?}
- frame_num >= 81 (samples below threshold are filtered)
- All modality files must exist

Usage:
    # data_list.json (training format, default) - same as prepare_data Step 7
    python scan_folders_to_data_list_json.py -f /data/HDTF_processed -o data_list_train.json

    # With emo, eye, flip (match train.sh / prepare_data_parallel)
    python scan_folders_to_data_list_json.py -f /data/HDTF_processed -o data_list_train.json --use-emo --use-eye-open --use-eye-ball --with-flip

    # Multiple folders merged
    python scan_folders_to_data_list_json.py -f /data/HDTF_processed /data/hallo3_processed -o data_list_train.json --use-emo --use-eye-open --use-eye-ball --with-flip

    # data_info.json (intermediate, for gather_data_list_json input)
    python scan_folders_to_data_list_json.py -f /data/HDTF_processed -o data_info.json --format data_info
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Dirs used to discover sample names (union, same as train.sh)
NAME_SOURCE_DIRS = ["LP_npy", "hubert_aud_npy", "video"]

N_FRAME_THRESH = 81  # same as gather_data_list_json_for_train.check_one_v2


def get_names_from_folder(folder: Path) -> set[str]:
    """Get base names from LP_npy, hubert_aud_npy, video (same as train.sh)."""
    names = set()
    for dirname in NAME_SOURCE_DIRS:
        dirpath = folder / dirname
        if dirpath.is_dir():
            for f in dirpath.iterdir():
                if "." in f.name:
                    name = f.name.rsplit(".", 1)[0]
                    names.add(name)
    return names


def scan_folder(folder: Path) -> dict[str, list[str]]:
    """Scan one folder, build data_info (path lists)."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        return {}

    names = sorted(get_names_from_folder(folder))
    if not names:
        return {}

    data_root = str(folder)
    data_info = {
        "video_list": [os.path.join(data_root, "video", n + ".mp4") for n in names],
        "LP_npy_list": [os.path.join(data_root, "LP_npy", n + ".npy") for n in names],
        "hubert_aud_npy_list": [os.path.join(data_root, "hubert_aud_npy", n + ".npy") for n in names],
        "wav_list": [os.path.join(data_root, "wav", n + ".wav") for n in names],
        "LP_pkl_list": [os.path.join(data_root, "LP_pkl", n + ".pkl") for n in names],
        "MP_lmk_npy_list": [os.path.join(data_root, "MP_lmk_npy", n + ".npy") for n in names],
        "eye_open_npy_list": [os.path.join(data_root, "eye_open_npy", n + ".npy") for n in names],
        "eye_ball_npy_list": [os.path.join(data_root, "eye_ball_npy", n + ".npy") for n in names],
        "emo_npy_list": [os.path.join(data_root, "emo_npy", n + ".npy") for n in names],
    }
    return data_info


def check_one(data: dict, n_thre: int = N_FRAME_THRESH) -> tuple:
    """Check all files exist and min frame count >= n_thre. Skip unreadable/corrupt files."""
    ns = []
    for k, v in data.items():
        if not os.path.isfile(v):
            return False, None
        try:
            n = np.load(v, allow_pickle=False).shape[0]
        except Exception:
            # Corrupt or mismatched npy; skip this sample
            return False, None
        ns.append(n)
    n_min = min(ns)
    if n_min < n_thre:
        return False, None
    return True, n_min


def flip_path(p: str) -> str:
    """Same as gather_data_list_json_for_train.flip_path."""
    items = p.replace("\\", "/").split("/")
    if len(items) >= 2:
        items[-2] = items[-2] + "_flip"
    return "/".join(items)


def gather_data_list(
    data_info: dict,
    use_emo: bool = False,
    use_eye_open: bool = False,
    use_eye_ball: bool = False,
    flip: bool = False,
) -> list[dict]:
    """Filter and build data_list. Same logic as gather_data_list_json_for_train."""
    lst = []
    num_v = len(data_info["hubert_aud_npy_list"])
    for i in range(num_v):
        data = {
            "mtn": data_info["LP_npy_list"][i],
            "aud": data_info["hubert_aud_npy_list"][i],
        }
        if use_emo:
            data["emo"] = data_info["emo_npy_list"][i]
        if use_eye_open:
            data["eye_open"] = data_info["eye_open_npy_list"][i]
        if use_eye_ball:
            data["eye_ball"] = data_info["eye_ball_npy_list"][i]
        if flip:
            for k in ["mtn", "eye_open", "eye_ball"]:
                if k in data:
                    data[k] = flip_path(data[k])

        ok, n = check_one(data)
        if not ok:
            continue
        data["frame_num"] = n
        lst.append(data)
    return lst


def merge_data_infos(infos: list[dict]) -> dict:
    """Merge multiple data_info dicts by concatenating lists."""
    if not infos:
        return {}
    keys = list(infos[0].keys())
    merged = {k: [] for k in keys}
    for info in infos:
        for k in keys:
            merged[k].extend(info.get(k, []))
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Scan processed folder(s) and generate training JSON (data_list or data_info)."
    )
    parser.add_argument(
        "-f", "--folders",
        nargs="+",
        required=True,
        help="One or more root folders containing LP_npy, hubert_aud_npy, etc.",
    )
    parser.add_argument(
        "-o", "--output",
        default="data_list_train.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--format",
        choices=["data_list", "data_info"],
        default="data_list",
        help="data_list: training format (81-frame filtered, mtn/aud/frame_num/...). data_info: path lists.",
    )
    parser.add_argument("--use-emo", action="store_true")
    parser.add_argument("--use-eye-open", action="store_true")
    parser.add_argument("--use-eye-ball", action="store_true")
    parser.add_argument("--with-flip", action="store_true")
    args = parser.parse_args()

    all_infos = []
    for folder in args.folders:
        p = Path(folder)
        if not p.exists():
            print(f"Warning: folder does not exist, skipping: {p}")
            continue
        info = scan_folder(p)
        if not info:
            print(f"Warning: no valid samples in {p}")
            continue
        n = len(info["LP_npy_list"])
        print(f"  {p}: {n} path entries")
        all_infos.append(info)

    if not all_infos:
        print("Error: no valid data found in any folder.")
        return 1

    merged_info = merge_data_infos(all_infos)

    if args.format == "data_info":
        out_data = merged_info
        total = len(merged_info["LP_npy_list"])
    else:
        lst = gather_data_list(
            merged_info,
            use_emo=args.use_emo,
            use_eye_open=args.use_eye_open,
            use_eye_ball=args.use_eye_ball,
            flip=False,
        )
        if args.with_flip:
            flip_lst = gather_data_list(
                merged_info,
                use_emo=args.use_emo,
                use_eye_open=args.use_eye_open,
                use_eye_ball=args.use_eye_ball,
                flip=True,
            )
            lst = lst + flip_lst
        out_data = lst
        total = len(lst)
        print(f"  Filtered (frame_num>={N_FRAME_THRESH}): {total} training samples")

    print(f"Total: {total}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    exit(main())
