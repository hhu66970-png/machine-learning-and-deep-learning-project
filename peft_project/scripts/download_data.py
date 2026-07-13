"""Download all downstream datasets in parallel, with retries.
Logs ASCII-only status so Windows console stays readable.
"""
import os
import sys
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

socket.setdefaulttimeout(60)

# Use HF mirror if any HF download happens (harmless otherwise)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_ROOT, exist_ok=True)

import torchvision.datasets as D

# Optional: Tsinghua mirror fallback for CIFAR-100
TSINGHUA_CIFAR = "https://mirrors.tuna.tsinghua.edu.cn/cifar/cifar-100-python.tar.gz"


def dl_cifar100():
    D.CIFAR100(DATA_ROOT, train=True, download=True)
    D.CIFAR100(DATA_ROOT, train=False, download=True)
    return "cifar100"


def dl_flowers():
    for sp in ("train", "val", "test"):
        D.Flowers102(DATA_ROOT, split=sp, download=True)
    return "flowers102"


def dl_pets():
    for sp in ("trainval", "test"):
        D.OxfordIIITPet(DATA_ROOT, split=sp, download=True)
    return "pets"


def dl_dtd():
    for sp in ("train", "val", "test"):
        D.DTD(DATA_ROOT, split=sp, download=True)
    return "dtd"


TASKS = {
    "cifar100": dl_cifar100,
    "flowers102": dl_flowers,
    "pets": dl_pets,
    "dtd": dl_dtd,
}


def run_with_retry(name, fn, retries=3):
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            fn()
            print(f"[OK] {name} downloaded in {time.time()-t0:.0f}s", flush=True)
            return name, True, ""
        except Exception as e:
            print(f"[RETRY {attempt}/{retries}] {name}: {e}", flush=True)
            if name == "cifar100" and attempt == 1:
                try:
                    D.CIFAR100.url = TSINGHUA_CIFAR
                    D.CIFAR100.filename = "cifar-100-python.tar.gz"
                    print("[INFO] switching CIFAR-100 to Tsinghua mirror", flush=True)
                except Exception:
                    pass
            time.sleep(3)
    return name, False, traceback.format_exc()


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else list(TASKS.keys())
    print(f"[START] downloading: {which}  -> {DATA_ROOT}", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=len(which)) as ex:
        futs = {ex.submit(run_with_retry, n, TASKS[n]): n for n in which}
        for f in as_completed(futs):
            name, ok, err = f.result()
            results[name] = ok
            if not ok:
                print(f"[FAIL] {name}\n{err}", flush=True)
    print("[SUMMARY] " + ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()), flush=True)
    print("DOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
