"""Run the full experiment matrix (methods x datasets) + LoRA-rank ablation.

- Each run is a separate subprocess so GPU memory is fully released between runs.
- Results append incrementally to outputs/results.csv (train.py does the writing).
- Resumable: completed (tag, method, dataset, lora_r) combos are skipped.
- Robust: a crashed run is logged and the sweep continues.
"""
import os
import sys
import csv
import time
import subprocess

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJ, "src")
RESULTS = os.path.join(PROJ, "outputs", "results.csv")
LOG_DIR = os.path.join(PROJ, "outputs", "logs")
PY = sys.executable

# Datasets ordered fastest -> slowest, so a complete sub-matrix appears early.
DATASET_ORDER = ["flowers102", "dtd", "pets", "cifar100"]
EPOCHS = {"flowers102": 30, "dtd": 20, "pets": 20, "cifar100": 6}
# larger test sets -> evaluate less often to save time (best_acc still tracked)
EVAL_INTERVAL = {"flowers102": 3, "dtd": 2, "pets": 2, "cifar100": 2}
# Methods ordered cheap/important first, heaviest (full_ft) last.
METHOD_ORDER = ["linear_probe", "bitfit", "ssf", "vpt", "lora", "adaptformer", "lora_ssf", "full_ft"]

BATCH = 128
WORKERS = 8
RUN_TIMEOUT = 90 * 60  # seconds


def done_keys():
    keys = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                keys.add((r["tag"], r["method"], r["dataset"], str(r.get("lora_r", "8"))))
    return keys


def launch(method, dataset, epochs, tag, lora_r=8, eval_interval=1, extra=None):
    env = dict(os.environ)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        PY, os.path.join(SRC, "train.py"),
        "--method", method, "--dataset", dataset,
        "--epochs", str(epochs), "--batch_size", str(BATCH),
        "--num_workers", str(WORKERS), "--tag", tag,
        "--lora_r", str(lora_r), "--lora_alpha", str(lora_r),
        "--eval_interval", str(eval_interval),
        "--results_csv", RESULTS,
    ]
    if extra:
        cmd += extra
    os.makedirs(LOG_DIR, exist_ok=True)
    logf = os.path.join(LOG_DIR, f"{tag}_{method}_{dataset}_r{lora_r}.log")
    print(f">>> [{time.strftime('%H:%M:%S')}] RUN tag={tag} {method}/{dataset} "
          f"epochs={epochs} r={lora_r}", flush=True)
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as lf:
        try:
            subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=RUN_TIMEOUT, check=True)
            print(f"    OK ({time.time()-t0:.0f}s) -> {logf}", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"    FAILED rc={e.returncode} (see {logf})", flush=True)
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {RUN_TIMEOUT}s (see {logf})", flush=True)


def main():
    done = done_keys()
    print(f"[MATRIX START] {time.strftime('%Y-%m-%d %H:%M:%S')}  already done: {len(done)}", flush=True)

    # ---- 1) main matrix ----
    total = len(DATASET_ORDER) * len(METHOD_ORDER)
    idx = 0
    for ds in DATASET_ORDER:
        for m in METHOD_ORDER:
            idx += 1
            key = ("main", m, ds, "8")
            if key in done:
                print(f"[{idx}/{total}] skip (done) main {m}/{ds}", flush=True)
                continue
            print(f"[{idx}/{total}]", end=" ", flush=True)
            launch(m, ds, EPOCHS[ds], "main", eval_interval=EVAL_INTERVAL[ds])

    # ---- 2) LoRA rank ablation on a small/fast dataset ----
    for r in [1, 2, 4, 8, 16, 32]:
        key = ("abl_loraR", "lora", "dtd", str(r))
        if key in done:
            print(f"skip (done) abl_loraR lora/dtd r={r}", flush=True)
            continue
        launch("lora", "dtd", EPOCHS["dtd"], "abl_loraR", lora_r=r, eval_interval=1)

    print(f"[MATRIX DONE] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("MATRIX_DONE", flush=True)


if __name__ == "__main__":
    main()
