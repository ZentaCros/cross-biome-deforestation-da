"""
Full experiment sweep for Paper A (empirical study).
Runs source-only, FDA mix50 (50% styled) and FDA scale=1.0 (100% styled),
each with and without BN-TTA, for all 6 pairs x 3 seeds (108 runs).

Safe to run as several parallel workers (python run_sweep_cd.py --worker A,
then --worker B): each run is claimed with a lock file in results/_locks so no
two workers take the same run. A finished run (final_summary.csv) is skipped.
An eval run waits until its training run has finished, because best_model.pt
is rewritten while training is still going.
If a worker is killed, its lock stays: delete results/_locks/<tag>.lock by hand.
"""
import os
import sys
import time
import argparse
import subprocess

PYTHON = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
LOCKS = os.path.join(RESULTS, "_locks")

PAIRS = [
    ("PA", "RO"), ("RO", "PA"),
    ("MA", "RO"), ("RO", "MA"),
    ("PA", "MA"), ("MA", "PA"),
]
SEEDS = [42, 123, 456]
TRAIN_ARGS = ["--batch_size", "32", "--num_workers", "2"]


def done(tag):
    return os.path.exists(os.path.join(RESULTS, tag, "final_summary.csv"))


def failed(tag):
    return os.path.exists(os.path.join(LOCKS, f"{tag}.failed"))


def ckpt_path(tag):
    return os.path.join(BASE, "checkpoints", tag, "best_model.pt")


def build_tasks():
    """Ordered list of (tag, desc, cmd, depends_on_tag)."""
    tasks = []
    for src, tgt in PAIRS:
        for seed in SEEDS:
            tag = f"cd_source_only_{src}_to_{tgt}_seed{seed}"
            tasks.append((tag, f"Source-only {src}->{tgt} seed{seed}", [
                PYTHON, "-u", "train_cd.py",
                "--source_domain", src, "--target_domain", tgt,
                "--seed", str(seed), "--experiment_name", "cd_source_only",
                *TRAIN_ARGS], None))
    for src, tgt in PAIRS:
        for seed in SEEDS:
            dep = f"cd_source_only_{src}_to_{tgt}_seed{seed}"
            tag = f"cd_source_only_bntta_{src}_to_{tgt}_seed{seed}"
            tasks.append((tag, f"BN-TTA {src}->{tgt} seed{seed}", [
                PYTHON, "-u", "eval_bn_tta.py",
                "--checkpoint", ckpt_path(dep),
                "--source_domain", src, "--target_domain", tgt,
                "--output_dir", os.path.join(RESULTS, tag),
                "--bn_tta", "--bn_steps", "200"], dep))
    # FDA variants, all 6 pairs x 3 seeds:
    #   cd_fda_mix50 = 50% styled patches (transfer_scale 0.5)
    #   cd_fda       = 100% styled patches (transfer_scale 1.0)
    fda = [("cd_fda_mix50", "FDA mix50", "0.5"), ("cd_fda", "FDA scale=1.0", "1.0")]
    for exp, label, scale in fda:
        for src, tgt in PAIRS:
            for seed in SEEDS:
                tag = f"{exp}_{src}_to_{tgt}_seed{seed}"
                tasks.append((tag, f"{label} {src}->{tgt} seed{seed}", [
                    PYTHON, "-u", "train_cd.py",
                    "--source_domain", src, "--target_domain", tgt,
                    "--seed", str(seed), "--experiment_name", exp,
                    "--use_fda", "--fda_beta", "0.01", "--n_styled", "3",
                    "--transfer_scale", scale, *TRAIN_ARGS], None))
    for exp, label, scale in fda:
        for src, tgt in PAIRS:
            for seed in SEEDS:
                dep = f"{exp}_{src}_to_{tgt}_seed{seed}"
                tag = f"{exp}_bntta_{src}_to_{tgt}_seed{seed}"
                tasks.append((tag, f"{label}+BN-TTA {src}->{tgt} seed{seed}", [
                    PYTHON, "-u", "eval_bn_tta.py",
                    "--checkpoint", ckpt_path(dep),
                    "--source_domain", src, "--target_domain", tgt,
                    "--output_dir", os.path.join(RESULTS, tag),
                    "--bn_tta", "--bn_steps", "200"], dep))
    return tasks


def try_lock(tag, worker):
    try:
        fd = os.open(os.path.join(LOCKS, f"{tag}.lock"),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{worker} pid={os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
    os.close(fd)
    return True


def release(tag):
    try:
        os.remove(os.path.join(LOCKS, f"{tag}.lock"))
    except FileNotFoundError:
        pass


def run_cmd(args, desc, worker):
    print(f"\n{'=' * 60}")
    print(f"  [{worker}] {desc}")
    print(f"  {' '.join(args)}")
    print(f"{'=' * 60}\n", flush=True)
    t0 = time.time()
    r = subprocess.run(args, cwd=BASE)
    dt = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"\n  [{status}] {desc} ({dt:.0f}s)\n", flush=True)
    return r.returncode == 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", default="A")
    args = p.parse_args(argv)
    os.makedirs(LOCKS, exist_ok=True)
    tasks = build_tasks()
    ran = n_failed = 0
    waiting_said = False

    while True:
        todo = [t for t in tasks if not done(t[0]) and not failed(t[0])
                and not (t[3] and failed(t[3]))]
        if not todo:
            break
        claimed = None
        for tag, desc, cmd, dep in todo:
            if dep and not done(dep):
                continue
            if try_lock(tag, args.worker):
                # Another worker may have finished it between our scan and the lock.
                if done(tag) or failed(tag):
                    release(tag)
                    continue
                claimed = (tag, desc, cmd)
                break
        if claimed is None:
            # Everything left is being run by another worker or waits on one.
            if not waiting_said:
                print(f"  [{args.worker}] waiting: {len(todo)} runs held by other workers",
                      flush=True)
                waiting_said = True
            time.sleep(30)
            continue
        waiting_said = False
        tag, desc, cmd = claimed
        t0 = time.time()
        try:
            ok = run_cmd(cmd, desc, args.worker)
        finally:
            release(tag)
        with open(os.path.join(LOCKS, "durations.csv"), "a") as f:
            kind = "eval" if "eval_bn_tta.py" in cmd[2] else "train"
            f.write(f"{tag},{kind},{time.time() - t0:.0f},{args.worker},{int(ok)}\n")
        ran += 1
        if not ok:
            n_failed += 1
            with open(os.path.join(LOCKS, f"{tag}.failed"), "w") as f:
                f.write(f"{args.worker} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("\n" + "=" * 60)
    print(f"  SWEEP COMPLETE [{args.worker}]: {ran} runs by this worker, {n_failed} failed")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
