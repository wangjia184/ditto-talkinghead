"""
Process videos in parallel using multiprocessing
Each process handles one video completely (all steps) before moving to the next
"""
import os
import sys
import json
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
from tqdm import tqdm
import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))


def flip_path(p):
    """Generate flip path for a file"""
    items = p.split('/')
    if len(items) >= 2:
        items[-2] = items[-2] + '_flip'
    return '/'.join(items)


def process_single_video(args):
    """Process a single video - this function will be called by worker processes"""
    (idx, total, fps25_video, video, wav, hubert_aud_npy,
     LP_pkl, LP_npy, LP_pkl_flip, LP_npy_flip,
     MP_lmk_npy, eye_open_npy, eye_ball_npy,
     MP_lmk_npy_flip, eye_open_npy_flip, eye_ball_npy_flip,
     emo_npy, ditto_pytorch_path, Hubert_onnx, MP_face_landmarker_task_path, skip_existing) = args
    
    script_path = os.path.join(os.path.dirname(CUR_DIR), "scripts", "process_one_video_complete.py")
    
    cmd = [
        sys.executable, script_path,
        "--fps25_video", fps25_video,
        "--video", video,
        "--wav", wav,
        "--hubert_aud_npy", hubert_aud_npy,
        "--LP_pkl", LP_pkl,
        "--LP_npy", LP_npy,
        "--LP_pkl_flip", LP_pkl_flip,
        "--LP_npy_flip", LP_npy_flip,
        "--MP_lmk_npy", MP_lmk_npy,
        "--eye_open_npy", eye_open_npy,
        "--eye_ball_npy", eye_ball_npy,
        "--MP_lmk_npy_flip", MP_lmk_npy_flip,
        "--eye_open_npy_flip", eye_open_npy_flip,
        "--eye_ball_npy_flip", eye_ball_npy_flip,
        "--emo_npy", emo_npy,
        "--ditto_pytorch_path", ditto_pytorch_path,
        "--Hubert_onnx", Hubert_onnx,
        "--MP_face_landmarker_task_path", MP_face_landmarker_task_path
    ]
    
    if skip_existing:
        cmd.append("--skip_existing")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per video
        )
        
        success = result.returncode == 0
        video_name = os.path.basename(fps25_video)
        
        return {
            'idx': idx,
            'total': total,
            'video': video_name,
            'success': success,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'idx': idx,
            'total': total,
            'video': os.path.basename(fps25_video),
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Timeout: video processing took longer than 1 hour'
        }
    except Exception as e:
        return {
            'idx': idx,
            'total': total,
            'video': os.path.basename(fps25_video),
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""  # data_info.json path
    ditto_pytorch_path: str = ""
    Hubert_onnx: str = ""
    MP_face_landmarker_task_path: str = ""
    num_workers: int = 0  # 0 means auto-detect (use CPU count)
    skip_existing: bool = True


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    
    if not opt.input_data_json or not os.path.isfile(opt.input_data_json):
        print(f"Error: data_info.json not found: {opt.input_data_json}")
        sys.exit(1)
    
    # Load data_info.json
    with open(opt.input_data_json, 'r') as f:
        data_info = json.load(f)
    
    fps25_video_list = data_info.get('fps25_video_list', [])
    video_list = data_info.get('video_list', [])
    wav_list = data_info.get('wav_list', [])
    hubert_aud_npy_list = data_info.get('hubert_aud_npy_list', [])
    LP_pkl_list = data_info.get('LP_pkl_list', [])
    LP_npy_list = data_info.get('LP_npy_list', [])
    MP_lmk_npy_list = data_info.get('MP_lmk_npy_list', [])
    eye_open_npy_list = data_info.get('eye_open_npy_list', [])
    eye_ball_npy_list = data_info.get('eye_ball_npy_list', [])
    emo_npy_list = data_info.get('emo_npy_list', [])
    
    # Generate flip paths
    LP_pkl_flip_list = [flip_path(p) for p in LP_pkl_list]
    LP_npy_flip_list = [flip_path(p) for p in LP_npy_list]
    MP_lmk_npy_flip_list = [flip_path(p) for p in MP_lmk_npy_list]
    eye_open_npy_flip_list = [flip_path(p) for p in eye_open_npy_list]
    eye_ball_npy_flip_list = [flip_path(p) for p in eye_ball_npy_list]
    
    total = len(fps25_video_list)
    
    if total == 0:
        print("Error: No videos found in data_info.json")
        sys.exit(1)
    
    # Determine number of workers
    if opt.num_workers == 0:
        num_workers = multiprocessing.cpu_count()
        # Limit to reasonable number to avoid overwhelming system
        num_workers = min(num_workers, 8)
    else:
        num_workers = opt.num_workers
    
    print("="*70)
    print("Parallel Video Processing")
    print("="*70)
    print(f"Total videos: {total}")
    print(f"Number of workers: {num_workers}")
    print(f"Skip existing files: {opt.skip_existing}")
    print("="*70)
    print("")
    
    # Prepare arguments for each video
    tasks = []
    skipped_count = 0
    
    for idx in range(total):
        # Quick check: if skip_existing is enabled and all files exist, skip this video entirely
        if opt.skip_existing:
            all_exist = (
                os.path.isfile(video_list[idx]) and
                os.path.isfile(wav_list[idx]) and
                os.path.isfile(hubert_aud_npy_list[idx]) and
                os.path.isfile(LP_pkl_list[idx]) and
                os.path.isfile(LP_npy_list[idx]) and
                os.path.isfile(LP_pkl_flip_list[idx]) and
                os.path.isfile(LP_npy_flip_list[idx]) and
                os.path.isfile(MP_lmk_npy_list[idx]) and
                os.path.isfile(eye_open_npy_list[idx]) and
                os.path.isfile(eye_ball_npy_list[idx]) and
                os.path.isfile(MP_lmk_npy_flip_list[idx]) and
                os.path.isfile(eye_open_npy_flip_list[idx]) and
                os.path.isfile(eye_ball_npy_flip_list[idx]) and
                os.path.isfile(emo_npy_list[idx])
            )
            
            if all_exist:
                skipped_count += 1
                continue  # Skip adding to tasks
        
        tasks.append((
            idx + 1, total,
            fps25_video_list[idx],
            video_list[idx],
            wav_list[idx],
            hubert_aud_npy_list[idx],
            LP_pkl_list[idx],
            LP_npy_list[idx],
            LP_pkl_flip_list[idx],
            LP_npy_flip_list[idx],
            MP_lmk_npy_list[idx],
            eye_open_npy_list[idx],
            eye_ball_npy_list[idx],
            MP_lmk_npy_flip_list[idx],
            eye_open_npy_flip_list[idx],
            eye_ball_npy_flip_list[idx],
            emo_npy_list[idx],
            opt.ditto_pytorch_path,
            opt.Hubert_onnx,
            opt.MP_face_landmarker_task_path,
            opt.skip_existing
        ))
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} videos (all files already exist)")
        print(f"Remaining videos to process: {len(tasks)}")
        print("")
    
    if len(tasks) == 0:
        print("All videos are already processed! Nothing to do.")
        sys.exit(0)
    
    # Process videos in parallel
    start_time = time.time()
    success_count = skipped_count  # Count skipped videos as successful
    fail_count = 0
    failed_videos = []
    failed_results = []  # Store detailed failure information
    
    # Use multiprocessing Pool
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = []
        with tqdm(total=len(tasks), desc="Processing videos", unit="video") as pbar:
            for result in pool.imap_unordered(process_single_video, tasks):
                results.append(result)
                if result['success']:
                    success_count += 1
                    pbar.set_postfix({"Success": success_count, "Failed": fail_count})
                else:
                    fail_count += 1
                    failed_videos.append(result['video'])
                    failed_results.append(result)  # Store detailed failure info
                    
                    # Print failure immediately
                    print(f"\n{'='*70}", flush=True)
                    print(f"✗ FAILED: [{result['idx']}/{result['total']}] {result['video']}", flush=True)
                    print(f"Return code: {result['returncode']}", flush=True)
                    
                    # Special handling for common error codes
                    if result['returncode'] == -9:
                        print(f"\n⚠️  WARNING: Process was killed (SIGKILL, return code -9)", flush=True)
                        print(f"   This usually indicates:", flush=True)
                        print(f"   - Out of Memory (OOM) - system killed the process", flush=True)
                        print(f"   - System resource exhaustion", flush=True)
                        print(f"   - Consider reducing number of workers (current: {num_workers})", flush=True)
                        print(f"   - Or process videos in smaller batches", flush=True)
                    elif result['returncode'] == -1:
                        if 'Timeout' in result.get('stderr', ''):
                            print(f"\n⚠️  WARNING: Process timed out (>1 hour)", flush=True)
                        else:
                            print(f"\n⚠️  WARNING: Process exception occurred", flush=True)
                    
                    if result['stderr']:
                        print(f"\nSTDERR:", flush=True)
                        stderr_text = result['stderr'][-2000:]  # Last 2000 chars
                        print(stderr_text, flush=True)
                        
                        # Check for memory-related errors
                        if any(keyword in stderr_text.lower() for keyword in 
                               ['memory', 'oom', 'killed', 'sigkill', 'cuda out of memory']):
                            print(f"\n💡 SUGGESTION: This appears to be a memory issue.", flush=True)
                            print(f"   Try reducing --num_workers or processing fewer videos at once.", flush=True)
                    
                    if result['stdout']:
                        # Extract error lines from stdout
                        stdout_lines = result['stdout'].split('\n')
                        error_lines = [line for line in stdout_lines if any(keyword in line.lower() 
                                    for keyword in ['error', 'failed', 'exception', 'traceback', '✗', 'memory', 'oom'])]
                        if error_lines:
                            print(f"\nSTDOUT (errors):", flush=True)
                            print('\n'.join(error_lines[-50:]), flush=True)  # Last 50 error lines
                    
                    print(f"{'='*70}\n", flush=True)
                    
                    pbar.set_postfix({"Success": success_count, "Failed": fail_count})
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("Processing Summary")
    print("="*70)
    print(f"Total videos: {total}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Processed successfully: {success_count - skipped_count}")
    print(f"Failed: {fail_count}")
    if len(tasks) > 0:
        print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Average time per video: {elapsed_time/len(tasks):.2f} seconds")
    print("="*70)
    
    if failed_videos:
        print(f"\n{'='*70}")
        print(f"Failed Videos Summary ({len(failed_videos)} total)")
        print(f"{'='*70}")
        
        # Group failures by error type
        oom_count = sum(1 for r in failed_results if r['returncode'] == -9)
        timeout_count = sum(1 for r in failed_results if r['returncode'] == -1 and 'Timeout' in r.get('stderr', ''))
        other_count = len(failed_results) - oom_count - timeout_count
        
        if oom_count > 0:
            print(f"\n⚠️  Memory issues (OOM): {oom_count} videos")
            print(f"   These processes were killed due to memory exhaustion.")
            print(f"   Consider reducing --num_workers or processing in smaller batches.")
        
        if timeout_count > 0:
            print(f"\n⏱️  Timeout issues: {timeout_count} videos")
            print(f"   These videos took longer than 1 hour to process.")
        
        if other_count > 0:
            print(f"\n❌ Other errors: {other_count} videos")
        
        # Show first 10 failed videos with details
        print(f"\nDetailed failures (showing first 10):")
        for i, result in enumerate(failed_results[:10], 1):
            print(f"\n[{i}] {result['video']}")
            print(f"    Return code: {result['returncode']}", end='')
            
            if result['returncode'] == -9:
                print(" (OOM - Out of Memory)")
            elif result['returncode'] == -1:
                if 'Timeout' in result.get('stderr', ''):
                    print(" (Timeout)")
                else:
                    print(" (Exception)")
            else:
                print()
            
            if result['stderr']:
                stderr_short = result['stderr'].split('\n')[-5:]  # Last 5 lines
                error_lines = [line for line in stderr_short if line.strip()]
                if error_lines:
                    print(f"    Last error lines:")
                    for line in error_lines:
                        print(f"      {line}")
        
        if len(failed_videos) > 10:
            print(f"\n  ... and {len(failed_videos) - 10} more failed videos")
        
        # Save detailed failure log to file
        failed_log_file = opt.input_data_json.replace('.json', '_failed_detailed.log')
        with open(failed_log_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("Detailed Failure Log\n")
            f.write("="*70 + "\n\n")
            
            for i, result in enumerate(failed_results, 1):
                f.write(f"\n{'='*70}\n")
                f.write(f"[{i}/{len(failed_results)}] {result['video']}\n")
                f.write(f"Index: {result['idx']}/{result['total']}\n")
                f.write(f"Return code: {result['returncode']}\n")
                f.write(f"{'='*70}\n\n")
                
                if result['stdout']:
                    f.write("STDOUT:\n")
                    f.write("-"*70 + "\n")
                    f.write(result['stdout'])
                    f.write("\n\n")
                
                if result['stderr']:
                    f.write("STDERR:\n")
                    f.write("-"*70 + "\n")
                    f.write(result['stderr'])
                    f.write("\n\n")
        
        # Save simple list of failed videos
        failed_list_file = opt.input_data_json.replace('.json', '_failed.txt')
        with open(failed_list_file, 'w', encoding='utf-8') as f:
            for video in failed_videos:
                f.write(f"{video}\n")
        
        print(f"\n{'='*70}")
        print(f"Failure logs saved:")
        print(f"  - Detailed log: {failed_log_file}")
        print(f"  - Simple list: {failed_list_file}")
        print(f"{'='*70}")
    
    if fail_count > 0:
        print(f"\nWarning: {fail_count} videos failed to process. Check the logs above.")
        sys.exit(1)
    else:
        print("\n✓ All videos processed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    # Required for Windows compatibility
    multiprocessing.freeze_support()
    main()

