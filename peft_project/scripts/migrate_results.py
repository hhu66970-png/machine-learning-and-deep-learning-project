"""One-off: migrate old results.csv (pre-expansion schema) to the new schema
so train.py can keep appending with aligned columns. Idempotent & backed up."""
import os
import csv
import shutil

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "outputs", "results.csv")

NEW_FIELDS = ["run_id", "tag", "method", "dataset", "backbone", "train_fraction",
              "num_classes", "epochs", "batch_size", "lr", "seed",
              "trainable_params", "total_params", "pct_trainable",
              "best_acc", "final_acc", "peak_mem_mb", "train_time_s",
              "time_per_epoch_s", "throughput_img_s", "model",
              "lora_r", "adapter_dim", "vpt_prompts"]


def model_short(name):
    if "small" in name:
        return "vit_s"
    if "large" in name:
        return "vit_l"
    if "base" in name:
        return "vit_b"
    return name.split("_")[0]


def main():
    if not os.path.exists(RES):
        print("no results.csv; nothing to migrate")
        return
    with open(RES, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "run_id" in rows[0]:
        print("already migrated")
        return
    shutil.copy(RES, RES + ".bak")
    out = []
    for r in rows:
        model = r.get("model", "vit_base_patch16_224.augreg_in21k")
        bk = model_short(model)
        r["backbone"] = bk
        r["train_fraction"] = 1.0
        r["run_id"] = (f"{r['tag']}__{bk}__{r['method']}__{r['dataset']}__s{r['seed']}"
                       f"__f100__r{r.get('lora_r', 8)}__a{r.get('adapter_dim', 64)}"
                       f"__p{r.get('vpt_prompts', 20)}")
        out.append({k: r.get(k, "") for k in NEW_FIELDS})
    with open(RES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NEW_FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"migrated {len(out)} rows -> new schema (backup at results.csv.bak)")


if __name__ == "__main__":
    main()
