import os
from tqdm.contrib import tzip
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import load_json


def extract_audio(path, out_path, sample_rate=16000, ffmpeg_bin='ffmpeg'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = f"{ffmpeg_bin} -loglevel error -y -i {path} -f wav -ar {sample_rate} {out_path}"
    os.system(cmd)


def process_data_list(video_list, wav_list, ffmpeg_bin='ffmpeg'):
    total = len(video_list)
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for idx, (video, wav) in enumerate(tzip(video_list, wav_list), 1):
        try:
            if os.path.isfile(wav):
                skip_count += 1
                continue
            print(f"[{idx}/{total}] Extracting audio: {os.path.basename(video)}")
            extract_audio(video, wav, ffmpeg_bin=ffmpeg_bin)
            if os.path.isfile(wav):
                success_count += 1
            else:
                print(f"Warning: Audio file not created: {wav}")
                fail_count += 1
        except Exception as e:
            print(f"[{idx}/{total}] Failed to extract audio: {os.path.basename(video)}")
            traceback.print_exc()
            fail_count += 1
    
    print(f"\nAudio extraction summary: {success_count} succeeded, {skip_count} skipped, {fail_count} failed out of {total} total")
    if fail_count > 0:
        print("Warning: Some audio extractions failed. Check the error messages above.")


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data list json: {'video_list': video_list, 'wav_list': wav_list}
    ffmpeg_bin: str = "ffmpeg"  # ffmpeg_bin path


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json

    data_info = load_json(opt.input_data_json)

    video_list = data_info['video_list']
    wav_list = data_info['wav_list']

    process_data_list(video_list, wav_list, opt.ffmpeg_bin)



if __name__ == '__main__':
    main()
