"""Master experiment campaign for the expanded PEFT study.

Groups (priority order: cheap/high-value first, expensive last):
  1. ViT-B on 4 new datasets (breadth)
  2. ViT-S full 8-dataset matrix (scaling, small end)
  3. Ablations (LoRA rank/alpha, adapter dim, VPT prompts, LR sensitivity)
  4. Data-efficiency (train-fraction sweep)
  5. Interpretability checkpoints (save trainable weights on Pets)
  6. ViT-B core multi-seed (seed 123)  -> statistical variance
  7. ViT-L core 4-dataset matrix (scaling, large end)
  8. ViT-B core multi-seed (seed 2024) -> 3-seed statistics

Each run is a subprocess (frees GPU). Resumable via run_id. Robust to crashes.
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

VIT_S = "vit_small_patch16_224.augreg_in21k"
VIT_B = "vit_base_patch16_224.augreg_in21k"
VIT_L = "vit_large_patch16_224.augreg_in21k"

CORE = ["flowers102", "dtd", "pets", "cifar100"]
NEW = ["cifar10", "svhn", "eurosat", "gtsrb"]
ALL8 = CORE + NEW
METHODS = ["linear_probe", "bitfit", "ssf", "vpt", "lora", "adaptformer", "lora_ssf", "full_ft"]

# large/easy datasets converge in 2-3 epochs with a pretrained ViT -> fewer epochs
EPOCHS = {"flowers102": 30, "dtd": 20, "pets": 20, "cifar100": 6,
          "cifar10": 3, "svhn": 2, "eurosat": 5, "gtsrb": 3}
EVAL_INTERVAL = {"flowers102": 3, "dtd": 2, "pets": 2, "cifar100": 2,
                 "cifar10": 1, "svhn": 1, "eurosat": 2, "gtsrb": 1}

RUN_TIMEOUT = 150 * 60


def model_short(name):
    if "small" in name:
        return "vit_s"
    if "large" in name:
        return "vit_l"
    if "base" in name:
        return "vit_b"
    return name.split("_")[0]


def batch_for(model, method):
    if "large" in model:
        return 32 if method == "full_ft" else 64
    return 128


def job(method, dataset, model, seed=42, tag="main", lr=None, lora_r=8,
        lora_alpha=None, adapter_dim=64, vpt_prompts=20, train_fraction=1.0,
        save_ckpt=False):
    return {
        "method": method, "dataset": dataset, "model": model, "seed": seed,
        "tag": tag, "lr": lr, "lora_r": lora_r,
        "lora_alpha": lora_alpha if lora_alpha is not None else lora_r,
        "adapter_dim": adapter_dim, "vpt_prompts": vpt_prompts,
        "train_fraction": train_fraction, "save_ckpt": save_ckpt,
    }


def run_id_of(j):
    ms = model_short(j["model"])
    frac = int(round(j["train_fraction"] * 100))
    return (f"{j['tag']}__{ms}__{j['method']}__{j['dataset']}__s{j['seed']}"
            f"__f{frac}__r{j['lora_r']}__a{j['adapter_dim']}__p{j['vpt_prompts']}")


# --------------------------------------------------------------------------- #
def build_jobs():
    jobs = []
    # 1. ViT-B broad (new datasets) -- already done, kept for resume safety
    for ds in NEW:
        for m in METHODS:
            jobs.append(job(m, ds, VIT_B))
    # 2. Interpretability checkpoints (report-critical, only 8 runs ~1h) -- FRONT-LOADED
    for m in METHODS:
        jobs.append(job(m, "pets", VIT_B, tag="interp", save_ckpt=True))
    # 3. Data-efficiency (compelling low-data curves)
    for ds in ["dtd", "cifar100"]:
        for m in ["linear_probe", "bitfit", "lora", "adaptformer", "lora_ssf"]:
            for frac in [0.05, 0.1, 0.25, 0.5]:
                jobs.append(job(m, ds, VIT_B, tag="fracEff", train_fraction=frac))
    # 4. Ablations (ViT-B, on DTD)
    for r in [1, 2, 4, 8, 16, 32]:
        jobs.append(job("lora", "dtd", VIT_B, tag="abl_loraR", lora_r=r, lora_alpha=r))
    for a in [4, 8, 16, 32]:
        jobs.append(job("lora", "dtd", VIT_B, tag=f"abl_loraAlpha_a{a}", lora_r=8, lora_alpha=a))
    for d in [8, 16, 32, 64, 128, 256]:
        jobs.append(job("adaptformer", "dtd", VIT_B, tag="abl_adapterDim", adapter_dim=d))
    for p in [1, 5, 10, 20, 50, 100]:
        jobs.append(job("vpt", "dtd", VIT_B, tag="abl_vptPrompts", vpt_prompts=p))
    for m in ["lora", "ssf", "full_ft"]:
        for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
            jobs.append(job(m, "dtd", VIT_B, tag=f"abl_lr_{m}_{lr:g}", lr=lr))
    # 5. ViT-S full matrix (core4 done -> scaling ready; new4 broadens it)
    for ds in ALL8:
        for m in METHODS:
            jobs.append(job(m, ds, VIT_S))
    # 6. ViT-B core multi-seed (seed 123) -- statistical variance
    for ds in CORE:
        for m in METHODS:
            jobs.append(job(m, ds, VIT_B, seed=123))
    # 7. ViT-L core matrix -- scaling top-end (slow, bonus)
    for ds in CORE:
        for m in METHODS:
            jobs.append(job(m, ds, VIT_L))
    # 8. ViT-B core multi-seed (seed 2024)
    for ds in CORE:
        for m in METHODS:
            jobs.append(job(m, ds, VIT_B, seed=2024))
    return jobs


def done_run_ids():
    ids = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("run_id"):
                    ids.add(r["run_id"])
    return ids


def launch(j):
    rid = run_id_of(j)
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    batch = batch_for(j["model"], j["method"])
    cmd = [
        PY, os.path.join(SRC, "train.py"),
        "--method", j["method"], "--dataset", j["dataset"], "--model", j["model"],
        "--seed", str(j["seed"]), "--tag", j["tag"],
        "--epochs", str(EPOCHS[j["dataset"]]),
        "--eval_interval", str(EVAL_INTERVAL[j["dataset"]]),
        "--batch_size", str(batch), "--num_workers", "8",
        "--lora_r", str(j["lora_r"]), "--lora_alpha", str(j["lora_alpha"]),
        "--adapter_dim", str(j["adapter_dim"]), "--vpt_prompts", str(j["vpt_prompts"]),
        "--train_fraction", str(j["train_fraction"]),
        "--results_csv", RESULTS,
    ]
    if j["lr"] is not None:
        cmd += ["--lr", str(j["lr"])]
    if "large" in j["model"]:
        cmd += ["--grad_ckpt"]
    if j["save_ckpt"]:
        cmd += ["--save_ckpt"]
    os.makedirs(LOG_DIR, exist_ok=True)
    logf = os.path.join(LOG_DIR, f"{rid}.log")
    print(f">>> [{time.strftime('%H:%M:%S')}] {rid}  (batch={batch})", flush=True)
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as lf:
        try:
            subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=RUN_TIMEOUT, check=True)
            print(f"    OK ({time.time()-t0:.0f}s)", flush=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"    FAILED rc={e.returncode} (see {os.path.basename(logf)})", flush=True)
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT (see {os.path.basename(logf)})", flush=True)
    return False


def main():
    jobs = build_jobs()
    done = done_run_ids()
    todo = [j for j in jobs if run_id_of(j) not in done]
    print(f"[CAMPAIGN START] {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"total={len(jobs)} done={len(jobs)-len(todo)} todo={len(todo)}", flush=True)
    for i, j in enumerate(todo, 1):
        # re-check done set live (in case of overlap / manual runs)
        if run_id_of(j) in done_run_ids():
            continue
        print(f"[{i}/{len(todo)}]", end=" ", flush=True)
        launch(j)
    print(f"[CAMPAIGN DONE] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("CAMPAIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
