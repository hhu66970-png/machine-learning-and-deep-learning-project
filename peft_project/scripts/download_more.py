"""Download the extra datasets (cifar10, svhn, eurosat, fgvc_aircraft, gtsrb)
and the ViT-S / ViT-L pretrained weights. ASCII-only logging.
"""
import os
import sys
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(90)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_ROOT, exist_ok=True)

import torchvision.datasets as D


def dl_cifar10():
    D.CIFAR10(DATA_ROOT, train=True, download=True)
    D.CIFAR10(DATA_ROOT, train=False, download=True)
    return "cifar10"


def dl_svhn():
    D.SVHN(DATA_ROOT, split="train", download=True)
    D.SVHN(DATA_ROOT, split="test", download=True)
    return "svhn"


def dl_eurosat():
    D.EuroSAT(DATA_ROOT, download=True)
    return "eurosat"


def dl_aircraft():
    D.FGVCAircraft(DATA_ROOT, split="trainval", annotation_level="variant", download=True)
    D.FGVCAircraft(DATA_ROOT, split="test", annotation_level="variant", download=True)
    return "fgvc_aircraft"


def dl_gtsrb():
    D.GTSRB(DATA_ROOT, split="train", download=True)
    D.GTSRB(DATA_ROOT, split="test", download=True)
    return "gtsrb"


def dl_weights():
    import timm
    for n in ("vit_small_patch16_224.augreg_in21k", "vit_large_patch16_224.augreg_in21k"):
        m = timm.create_model(n, pretrained=True, num_classes=10)
        del m
    return "vit_s_l_weights"


TASKS = {
    "cifar10": dl_cifar10, "svhn": dl_svhn, "eurosat": dl_eurosat,
    "fgvc_aircraft": dl_aircraft, "gtsrb": dl_gtsrb, "weights": dl_weights,
}


def run_with_retry(name, fn, retries=3):
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            fn()
            print(f"[OK] {name} in {time.time()-t0:.0f}s", flush=True)
            return name, True
        except Exception as e:
            print(f"[RETRY {attempt}/{retries}] {name}: {e}", flush=True)
            time.sleep(4)
    print(f"[FAIL] {name}\n{traceback.format_exc()}", flush=True)
    return name, False


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else list(TASKS.keys())
    print(f"[START] {which} -> {DATA_ROOT}", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=len(which)) as ex:
        futs = {ex.submit(run_with_retry, n, TASKS[n]): n for n in which}
        for f in as_completed(futs):
            name, ok = f.result()
            results[name] = ok
    print("[SUMMARY] " + ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()), flush=True)
    print("DOWNLOAD_MORE_DONE", flush=True)


if __name__ == "__main__":
    main()
