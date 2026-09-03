#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import time

import torch


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def completed_epochs(checkpoint):
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (FileNotFoundError, EOFError, RuntimeError):
        return None
    epoch = state.get("epoch")
    return None if epoch is None else int(epoch) + 1


def wait_for_limit(pid, checkpoint, max_epochs, interval, dry_run):
    while process_exists(pid):
        completed = completed_epochs(checkpoint)
        print(
            f"baseline_pid={pid} completed_epochs={completed} limit={max_epochs}",
            flush=True,
        )
        if completed is not None and completed >= max_epochs:
            if dry_run:
                return False
            process_group = os.getpgid(pid)
            print(f"Stopping completed baseline process group {process_group}", flush=True)
            os.killpg(process_group, signal.SIGINT)
            break
        if dry_run:
            return False
        time.sleep(interval)

    if dry_run:
        return False

    deadline = time.time() + 120
    while process_exists(pid) and time.time() < deadline:
        time.sleep(2)
    if process_exists(pid):
        process_group = os.getpgid(pid)
        print(f"Baseline did not exit after SIGINT; sending SIGTERM to {process_group}", flush=True)
        os.killpg(process_group, signal.SIGTERM)
        while process_exists(pid):
            time.sleep(2)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pid", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--next-config",
        default="configs/semantic_kitti/Matter3DToken_B-density-20.yaml",
    )
    parser.add_argument(
        "--next-log-root",
        default="logs/research/semantic_kitti/Matter3DToken_B-density-20",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stopped = wait_for_limit(
        args.baseline_pid,
        os.path.abspath(args.checkpoint),
        args.max_epochs,
        args.interval,
        args.dry_run,
    )
    if args.dry_run:
        return

    if not stopped and process_exists(args.baseline_pid):
        raise RuntimeError("Baseline is still running")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["CONFIG"] = args.next_config
    env["LOG_ROOT"] = args.next_log_root
    print(f"Launching density experiment with config {args.next_config}", flush=True)
    result = subprocess.run(
        ["bash", "train_semantickitti_component.sh"],
        cwd=repo,
        env=env,
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
