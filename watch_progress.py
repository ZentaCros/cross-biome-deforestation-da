"""
FADE-Net Paper A -- Experiment Sweep Progress Monitor (works with parallel workers)
Running runs come from results/_locks/*.lock; live epochs from each run's metrics.csv.
ETA uses run durations measured by the sweep (results/_locks/durations.csv).
Usage: python watch_progress.py
"""
import os
import csv
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
LOCKS = os.path.join(RESULTS, "_locks")
PATIENCE = 10

PAIRS = [("PA", "RO"), ("RO", "PA"), ("MA", "RO"),
         ("RO", "MA"), ("PA", "MA"), ("MA", "PA")]
SEEDS = [42, 123, 456]

G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"; D = "\033[90m"; RD = "\033[91m"
B = "\033[1m"; R = "\033[0m"
BG_G = "\033[42m"; BG_Y = "\033[43m"; BG_D = "\033[100m"


def build_phases():
    p1 = [("train", f"{s}->{t} s{sd}", f"cd_source_only_{s}_to_{t}_seed{sd}", t)
          for s, t in PAIRS for sd in SEEDS]
    p2 = [("eval", f"{s}->{t} s{sd} +BN-TTA", f"cd_source_only_bntta_{s}_to_{t}_seed{sd}", t)
          for s, t in PAIRS for sd in SEEDS]
    p3 = [("train", f"{s}->{t} s{sd} mix50", f"cd_fda_mix50_{s}_to_{t}_seed{sd}", t)
          for s, t in PAIRS for sd in SEEDS]
    p4 = [("train", f"{s}->{t} s{sd} s=1.0", f"cd_fda_{s}_to_{t}_seed{sd}", t)
          for s, t in PAIRS for sd in SEEDS]
    p5 = [("eval", f"{s}->{t} s{sd} mix50+TTA", f"cd_fda_mix50_bntta_{s}_to_{t}_seed{sd}", t)
          for s, t in PAIRS for sd in SEEDS]
    p6 = [("eval", f"{s}->{t} s{sd} s=1.0+TTA", f"cd_fda_bntta_{s}_to_{t}_seed{sd}", t)
          for s, t in PAIRS for sd in SEEDS]
    return [("Phase 1: Source-only training (6 pairs x 3 seeds)", p1),
            ("Phase 2: BN-TTA evaluation", p2),
            ("Phase 3: FDA mix50 training", p3),
            ("Phase 4: FDA scale=1.0 training", p4),
            ("Phase 5: FDA mix50 + BN-TTA evaluation", p5),
            ("Phase 6: FDA scale=1.0 + BN-TTA evaluation", p6)]


def target_f1(tag, tgt):
    path = os.path.join(RESULTS, tag, "final_summary.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["domain"] == tgt and row["region"] == "test":
                    return float(row["F1"])
    except Exception:
        pass
    return None


def lock_owner(tag):
    try:
        with open(os.path.join(LOCKS, f"{tag}.lock")) as f:
            return f.read().split()[0]
    except OSError:
        return None


def live_epoch(tag):
    """(epoch, epochs since best val loss) from the run's per-epoch metrics.csv."""
    try:
        with open(os.path.join(RESULTS, tag, "metrics.csv")) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None, 0
    if not rows:
        return 0, 0
    best, since = float("inf"), 0
    for r in rows:
        v = float(r["SrcVal_Loss"])
        if v < best - 1e-5:
            best, since = v, 0
        else:
            since += 1
    return int(rows[-1]["Epoch"]), since


def durations():
    """Measured seconds: per kind (for runs never timed) and per tag (latest)."""
    train, ev, by_tag = [], [], {}
    try:
        with open(os.path.join(LOCKS, "durations.csv")) as f:
            for line in f:
                tag, kind, secs, worker, ok = line.strip().split(",")
                if ok == "1":
                    (ev if kind == "eval" else train).append(float(secs))
                    by_tag[tag] = float(secs)
    except (OSError, ValueError):
        pass
    return train, ev, by_tag


def lock_age(tag):
    try:
        return time.time() - os.path.getmtime(os.path.join(LOCKS, f"{tag}.lock"))
    except OSError:
        return 0.0


def bar(frac, w=22):
    n = int(max(0.0, min(frac, 1.0)) * w)
    if frac >= 1.0:
        return BG_G + " " * w + R
    if frac > 0:
        return BG_Y + " " * n + BG_D + " " * (w - n) + R
    return D + "." * w + R


def fmt(sec):
    sec = int(sec)
    h, m = divmod(sec // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {sec % 60:02d}s"


def render():
    phases = build_phases()
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{B}{C}{'=' * 80}\n  FADE-Net Paper A -- Experiment Sweep Monitor\n{'=' * 80}{R}")

    train_d, eval_d, by_tag = durations()
    avg = {"train": sum(train_d) / len(train_d) if train_d else None,
           "eval": sum(eval_d) / len(eval_d) if eval_d else None}
    total = done = 0
    running, workers = [], set()
    left_train = left_eval = 0
    work, n_timed, n_avg, n_unknown = 0.0, 0, 0, 0
    for name, items in phases:
        print(f"\n  {D}-- {name} --{R}")
        for kind, label, tag, tgt in items:
            total += 1
            f1 = target_f1(tag, tgt)
            if f1 is not None:
                done += 1
                print(f"  {label:<26s} {bar(1.0)}  {G}DONE{R}  F1 {f1:.4f}")
                continue
            if os.path.exists(os.path.join(LOCKS, f"{tag}.failed")):
                print(f"  {label:<26s} {bar(0.0)}  {RD}FAILED{R}")
                continue
            if kind == "train":
                left_train += 1
            else:
                left_eval += 1
            # Same run timed before (same tag) beats the overall average.
            est = by_tag.get(tag)
            if est is not None:
                n_timed += 1
            elif avg[kind] is not None:
                est, n_avg = avg[kind], n_avg + 1
            else:
                n_unknown += 1
            owner = lock_owner(tag)
            if est is not None:
                work += max(est - lock_age(tag), 0.0) if owner else est
            if owner:
                running.append(label)
                workers.add(owner)
                if kind == "train":
                    ep, since = live_epoch(tag)
                    frac = since / PATIENCE
                    info = (f"epoch {ep}, no-improve {since}/{PATIENCE}"
                            if ep else "loading data")
                else:
                    frac, info = 0.5, "evaluating"
                print(f"  {label:<26s} {bar(frac)}  {Y}RUNNING [{owner}]{R}  {info}")
            else:
                print(f"  {label:<26s} {bar(0.0)}  {D}pending{R}")

    n_workers = max(1, len(workers))
    print(f"\n{C}{'=' * 80}{R}")
    print(f"  {B}{total} experiments:{R}  {G}{done} done{R}  "
          f"{Y}{len(running)} running{R}  {D}{total - done - len(running)} pending{R}"
          f"   workers active: {len(workers)}")

    if done == total:
        print(f"\n  {G}{B}ALL EXPERIMENTS COMPLETE{R}")
    elif not (train_d or eval_d):
        print("  ETA: waiting for the first run to finish under the new setup")
    else:
        eta = work / n_workers
        finish = time.strftime('%H:%M', time.localtime(time.time() + eta))
        print(f"  ETA: ~{fmt(eta)}  (finish ~{finish})  over {n_workers} workers, "
              f"running runs counted from elapsed time")
        note = f"       {n_timed} runs use their own earlier timing, {n_avg} use the kind average"
        if n_unknown:
            note += f", {n_unknown} not yet timeable (excluded)"
        print(note)

    print(f"\n  {D}Refreshes every 10s. Ctrl+C to stop.{R}")


if __name__ == "__main__":
    try:
        while True:
            render()
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n{D}  Monitor stopped.{R}")
