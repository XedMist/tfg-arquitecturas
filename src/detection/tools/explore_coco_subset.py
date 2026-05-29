"""
explore_coco_subset.py — Elige nº de clases y % de datos para detección COCO.

Uso:
    # Solo estadísticas (no necesita GPU):
    python src/detection/tools/explore_coco_subset.py \\
        --ann-file /data/coco/annotations/instances_train2017.json

    # Con benchmark GPU real (mide velocidad real de tu hardware):
    python src/detection/tools/explore_coco_subset.py \\
        --ann-file /data/coco/annotations/instances_train2017.json \\
        --benchmark --arch gated_cnn --batch-size 8

    # Con velocidad conocida (it/s):
    python src/detection/tools/explore_coco_subset.py \\
        --ann-file /data/coco/annotations/instances_train2017.json \\
        --ref-its 3.2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Candidatos a explorar
CLASS_CANDIDATES = [10, 20, 40, 80]
SUBSET_CANDIDATES = [0.10, 0.25, 0.50, 1.00]
SCHEDULES = {"1x": 12, "2x": 24, "3x": 36}
_BACKBONE_FRACTION = 0.65  # backbone ≈ 65% del tiempo total del detector


# ─── COCO parsing ─────────────────────────────────────────────────────────────

def load_coco(ann_file: Path) -> dict:
    print(f"Cargando {ann_file} ...")
    with open(ann_file) as f:
        return json.load(f)


def class_stats(coco: dict) -> list[dict]:
    """Clases ordenadas de mayor a menor por instancias."""
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    inst: dict[int, int] = defaultdict(int)
    imgs: dict[int, set] = defaultdict(set)
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            continue
        inst[a["category_id"]] += 1
        imgs[a["category_id"]].add(a["image_id"])
    stats = [{"id": cid, "name": cat_map[cid],
               "instances": inst[cid], "images": len(imgs[cid])}
              for cid in cat_map]
    stats.sort(key=lambda x: x["instances"], reverse=True)
    return stats


def count_images(coco: dict, class_ids: set, frac: float, seed: int = 42) -> int:
    """Imágenes con ≥1 anotación de class_ids, aplicando subset frac."""
    found: set[int] = set()
    for a in coco["annotations"]:
        if not a.get("iscrowd") and a["category_id"] in class_ids:
            found.add(a["image_id"])
    lst = sorted(found)
    rng = random.Random(seed)
    rng.shuffle(lst)
    return max(1, int(len(lst) * frac))


# ─── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(arch: str, batch_size: int, n_warmup: int = 15, n_run: int = 60) -> float:
    """Mide it/s backbone (forward+backward). Devuelve it/s estimadas del detector."""
    import torch
    _SRC = Path(__file__).resolve().parents[2]
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from detection.backbones.metaformer_backbone import MetaFormerBackbone

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  GPU: {_gpu_name()} | device={dev}")
    model = MetaFormerBackbone(arch=arch, out_indices=(0,1,2,3),
                               frozen_stages=-1, pretrained=None).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Ajustar tamaño si OOM
    bs = batch_size
    H, W = 800, 1280
    for try_bs in [batch_size, batch_size // 2, 1]:
        try:
            x = torch.randn(try_bs, 3, H, W, device=dev)
            with torch.no_grad():
                model(x)
            bs = try_bs
            break
        except RuntimeError:
            if dev == "cuda":
                torch.cuda.empty_cache()
    if bs != batch_size:
        print(f"  OOM con bs={batch_size}, usando bs={bs} para escalar.")

    def step():
        x = torch.randn(bs, 3, H, W, device=dev)
        outs = model(x)
        loss = sum(o.mean() for o in outs)
        opt.zero_grad(); loss.backward(); opt.step()

    print(f"  Calentando {n_warmup} iters...", end="", flush=True)
    for _ in range(n_warmup):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    print(" OK")

    print(f"  Midiendo {n_run} iters...", end="", flush=True)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_run):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    bb_its = n_run / elapsed        # it/s backbone (al bs real)
    scale = batch_size / bs
    bb_its_full = bb_its / scale    # escalar a bs configurado
    det_its = bb_its_full * _BACKBONE_FRACTION  # detector completo estimado

    print(f" OK\n  backbone {bb_its:.2f} it/s (bs={bs})"
          f"  →  detector ~{det_its:.2f} it/s (bs={batch_size})")
    return det_its


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "desconocido"


# ─── Tiempo ───────────────────────────────────────────────────────────────────

def est_hours(n_imgs: int, bs: int, epochs: int, its: float) -> float:
    if its <= 0:
        return float("nan")
    return math.ceil(n_imgs / bs) * epochs / its / 3600


def fmt_h(h: float) -> str:
    if math.isnan(h):
        return "  ?"
    if h < 1:
        return f"{h*60:.0f}m"
    d = int(h // 24)
    return f"{d}d{h%24:.0f}h" if d else f"{h:.1f}h"


# ─── Tablas ───────────────────────────────────────────────────────────────────

def print_ranking(stats: list[dict], top: int = 25) -> None:
    print(f"\n{'─'*60}")
    print(f"  {'#':>3}  {'Clase':<22}  {'Instancias':>11}  {'Imágenes':>9}")
    print(f"{'─'*60}")
    for i, s in enumerate(stats[:top], 1):
        print(f"  {i:>3}  {s['name']:<22}  {s['instances']:>11,}  {s['images']:>9,}")
    if len(stats) > top:
        print(f"  ... ({len(stats)-top} más)")


def print_candidate_classes(stats: list[dict]) -> None:
    print(f"\n{'─'*60}")
    print("  CLASES POR CANDIDATO")
    print(f"{'─'*60}")
    for n in CLASS_CANDIDATES:
        names = ", ".join(s["name"] for s in stats[:n])
        print(f"  Top-{n:>2}: {names}")


def print_time_table(coco: dict, stats: list[dict], bs: int, its: float) -> None:
    cols = list(SCHEDULES.keys())
    W = 8
    print(f"\n{'─'*70}")
    label = f"it/s={its:.2f}" if its > 0 else "sin medición"
    print(f"  TIEMPOS ESTIMADOS ({label}, bs={bs})")
    print(f"{'─'*70}")
    hdr = f"  {'Cls':>4} {'Sub':>5} {'Imgs':>8} {'It/Ep':>6}"
    hdr += "".join(f"  {c:>{W}}" for c in cols)
    print(hdr)
    print(f"  {'─'*68}")

    for n_cls in CLASS_CANDIDATES:
        ids = {s["id"] for s in stats[:n_cls]}
        for frac in SUBSET_CANDIDATES:
            n = count_images(coco, ids, frac)
            iep = max(1, math.ceil(n / bs))
            row = f"  {n_cls:>4} {frac:>4.0%}  {n:>8,} {iep:>6,}"
            row += "".join(
                f"  {fmt_h(est_hours(n, bs, ep, its)):>{W}}"
                for ep in SCHEDULES.values()
            )
            print(row)
        print(f"  {'─'*68}")

    if its <= 0:
        print("\n  ⚠ Tiempo desconocido. Usa --benchmark o --ref-its VALUE")
    else:
        print(f"\n  1x={SCHEDULES['1x']}ep  2x={SCHEDULES['2x']}ep  3x={SCHEDULES['3x']}ep")


def recommend(coco: dict, stats: list[dict], bs: int, its: float) -> None:
    if its <= 0:
        return
    print(f"\n{'═'*70}")
    print("  RECOMENDACIÓN (objetivo: 2x en 8–24h)")
    print(f"{'─'*70}")
    for n_cls in CLASS_CANDIDATES:
        ids = {s["id"] for s in stats[:n_cls]}
        for frac in SUBSET_CANDIDATES:
            n = count_images(coco, ids, frac)
            h2 = est_hours(n, bs, SCHEDULES["2x"], its)
            if 6 <= h2 <= 30:
                h1 = est_hours(n, bs, SCHEDULES["1x"], its)
                h3 = est_hours(n, bs, SCHEDULES["3x"], its)
                print(f"\n  ✓ Top-{n_cls} clases @ {frac:.0%}  ({n:,} imgs)")
                print(f"    1x: {fmt_h(h1)}   2x: {fmt_h(h2)}   3x: {fmt_h(h3)}")
                print(f"    Clases: {', '.join(s['name'] for s in stats[:n_cls])}")
                print(f"\n  Configurar en tu base config:")
                print(f"    num_classes = {n_cls}   (cambiar my_classes en _base_*.py)")
                print(f"    subset_frac  — filtrar JSON o usar {frac:.0%} de las imágenes")
                return
    print("  No se encontró config en 6-30h para 2x. Revisa hardware o batch size.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Explorar subsets COCO para detección",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--ann-file", "-a", required=True, type=Path)
    p.add_argument("--batch-size", "-b", type=int, default=8)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--arch", default="gated_cnn",
                   choices=["gated_cnn", "gated_cnn_mamba", "gated_cnn_dat"])
    p.add_argument("--ref-its", type=float, default=0.0,
                   help="Velocidad manual en it/s si no usas --benchmark")
    p.add_argument("--top", type=int, default=25,
                   help="Nº clases a mostrar en el ranking (default 25)")
    args = p.parse_args()

    print("\n" + "═"*70)
    print("  COCO Subset Explorer — MetaFormer Detection")
    print("═"*70)

    coco = load_coco(args.ann_file)
    print(f"  Imágenes: {len(coco['images']):,}  |  "
          f"Instancias: {sum(1 for a in coco['annotations'] if not a.get('iscrowd')):,}  |  "
          f"Clases: {len(coco['categories'])}")

    stats = class_stats(coco)
    print_ranking(stats, args.top)
    print_candidate_classes(stats)

    its = args.ref_its
    if args.benchmark:
        print(f"\n  BENCHMARK  arch={args.arch}  bs={args.batch_size}")
        print(f"{'─'*50}")
        its = benchmark(args.arch, args.batch_size)

    print_time_table(coco, stats, args.batch_size, its)
    recommend(coco, stats, args.batch_size, its)
    print("\n" + "═"*70 + "\n")


if __name__ == "__main__":
    main()
