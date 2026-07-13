"""Train/evaluate one (method, dataset) configuration and log all metrics.

Run as CLI; appends one result row to a CSV and dumps the per-epoch curve.
The matrix runner launches this as a subprocess per run so GPU memory is freed.
"""
import os
import sys
import csv
import json
import math
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# weights are cached locally -> go offline to skip slow HF Hub network checks
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import timm
from models.peft import build_model, apply_peft, DEFAULT_MODEL
from datasets import get_loaders, DATASET_INFO
import utils

DEFAULT_LR = {
    "full_ft": 1e-4, "linear_probe": 1e-3, "bitfit": 1e-3,
    "lora": 1e-3, "adaptformer": 1e-3, "ssf": 2e-3, "vpt": 5e-3, "lora_ssf": 1e-3,
}
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_short(name):
    if "small" in name:
        return "vit_s"
    if "large" in name:
        return "vit_l"
    if "base" in name:
        return "vit_b"
    return name.split("_")[0]


def run_id(args):
    ms = model_short(args.model)
    frac = int(round(args.train_fraction * 100))
    return (f"{args.tag}__{ms}__{args.method}__{args.dataset}__s{args.seed}"
            f"__f{frac}__r{args.lora_r}__a{args.adapter_dim}__p{args.vpt_prompts}")


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(k in n for k in ("prompt", "cls_token", "pos_embed", "bias")):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=(0.9, 0.999))


def make_scheduler(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(warmup_ratio * total_steps))

    def fn(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # build the model ONCE (num_classes known from dataset registry)
    num_classes = DATASET_INFO[args.dataset]
    model = build_model(num_classes, model_name=args.model)
    data_cfg = timm.data.resolve_model_data_config(model)
    mean, std = data_cfg["mean"], data_cfg["std"]

    train_loader, test_loader, _ = get_loaders(
        args.dataset, mean, std, batch_size=args.batch_size,
        num_workers=args.num_workers, img_size=224,
        train_fraction=args.train_fraction, seed=args.seed,
    )
    cfg = {
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "adapter_dim": args.adapter_dim, "vpt_prompts": args.vpt_prompts,
    }
    model = apply_peft(model, args.method, cfg)
    if args.grad_ckpt:
        try:
            model.set_grad_checkpointing(True)
        except Exception as e:
            print(f"[warn] grad_ckpt not enabled: {e}", flush=True)
    model.to(device)

    trainable, total_p, pct = utils.count_parameters(model)
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.method]
    optimizer = build_optimizer(model, lr, args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print(f"[RUN] method={args.method} dataset={args.dataset} classes={num_classes} "
          f"trainable={trainable:,} ({pct:.3f}%) lr={lr} epochs={args.epochs} "
          f"amp={amp_dtype}", flush=True)

    utils.reset_peak_memory()
    best_acc = 0.0
    curve = []
    n_images = 0
    with utils.Timer() as total_timer:
        for epoch in range(args.epochs):
            model.train()
            loss_meter = utils.AverageMeter()
            with utils.Timer() as ep_timer:
                for x, y in train_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(x)
                        loss = criterion(logits, y)
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0)
                        optimizer.step()
                    scheduler.step()
                    loss_meter.update(loss.item(), x.size(0))
                    n_images += x.size(0)
            # evaluate every eval_interval epochs, plus the first and last 3 epochs
            do_eval = (((epoch + 1) % args.eval_interval == 0) or (epoch == 0)
                       or (epoch + 1 > args.epochs - 3))
            if do_eval:
                acc = evaluate(model, test_loader, device, amp_dtype)
                best_acc = max(best_acc, acc)
                curve.append({"epoch": epoch + 1, "train_loss": round(loss_meter.avg, 4),
                              "test_acc": round(acc, 3), "epoch_time_s": round(ep_timer.elapsed, 1)})
                print(f"  epoch {epoch+1}/{args.epochs}  loss={loss_meter.avg:.4f}  "
                      f"test_acc={acc:.2f}  best={best_acc:.2f}  ({ep_timer.elapsed:.0f}s)", flush=True)
            else:
                print(f"  epoch {epoch+1}/{args.epochs}  loss={loss_meter.avg:.4f}  "
                      f"(no eval)  ({ep_timer.elapsed:.0f}s)", flush=True)

    peak_mem = utils.peak_memory_mb()
    train_time = total_timer.elapsed
    throughput = n_images / train_time if train_time > 0 else 0.0

    rid = run_id(args)
    row = {
        "run_id": rid, "tag": args.tag, "method": args.method, "dataset": args.dataset,
        "backbone": model_short(args.model), "train_fraction": args.train_fraction,
        "num_classes": num_classes, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": lr, "seed": args.seed,
        "trainable_params": trainable, "total_params": total_p,
        "pct_trainable": round(pct, 4),
        "best_acc": round(best_acc, 3), "final_acc": round(curve[-1]["test_acc"], 3),
        "peak_mem_mb": round(peak_mem, 1), "train_time_s": round(train_time, 1),
        "time_per_epoch_s": round(train_time / args.epochs, 1),
        "throughput_img_s": round(throughput, 1), "model": args.model,
        "lora_r": args.lora_r, "adapter_dim": args.adapter_dim, "vpt_prompts": args.vpt_prompts,
    }

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    write_header = not os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    curve_dir = os.path.join(PROJ, "outputs", "curves")
    os.makedirs(curve_dir, exist_ok=True)
    with open(os.path.join(curve_dir, f"{rid}.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": row, "curve": curve}, f, indent=2)

    if args.save_ckpt:
        ckpt_dir = os.path.join(PROJ, "outputs", "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainable_sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"trainable": trainable_sd, "meta": row}, os.path.join(ckpt_dir, f"{rid}.pt"))
        print(f"[CKPT] saved {rid}.pt ({len(trainable_sd)} tensors)", flush=True)

    print(f"[DONE] {args.method}/{args.dataset}  best_acc={best_acc:.2f}  "
          f"trainable%={pct:.3f}  mem={peak_mem:.0f}MB  time={train_time:.0f}s", flush=True)
    return row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tag", default="main")
    p.add_argument("--results_csv", default=os.path.join(PROJ, "outputs", "results.csv"))
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--adapter_dim", type=int, default=64)
    p.add_argument("--vpt_prompts", type=int, default=20)
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--num_classes_placeholder", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
