"""
check_convergence.py — Analiza si el entrenamiento ha convergido.

Lee los logs de MMEngine y genera curvas de loss/mAP para decidir
si necesitas más épocas.

Uso:
    python src/detection/tools/check_convergence.py \\
        --work-dir outputs/detection/faster_rcnn_gated_cnn_fpn_1x

    # Comparar varios experimentos:
    python src/detection/tools/check_convergence.py \\
        --work-dir outputs/detection/exp_gcnn outputs/detection/exp_mamba \\
        --out outputs/convergence.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ─── Lectura de logs MMEngine ─────────────────────────────────────────────────

def find_scalars_files(work_dir: Path) -> list[Path]:
    """Busca archivos scalars.json de MMEngine (formato JSONL)."""
    candidates = sorted(work_dir.glob("*/vis_data/scalars.json"))
    if not candidates:
        # Alternativa: buscar .log y parsearlo
        candidates = sorted(work_dir.glob("*.log"))
    return candidates


def load_scalars_jsonl(path: Path) -> list[dict]:
    """Carga un archivo JSONL de MMEngine (una línea = un dict)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_mmengine_log(log_path: Path) -> list[dict]:
    """Parsea un archivo .log de MMEngine buscando métricas clave."""
    import re
    records = []
    # Ejemplo línea train: "Epoch(train) [1][50/14786]  loss: 1.234 ..."
    # Ejemplo línea val:   "Epoch(val)   [1]  coco/bbox_mAP: 0.123 ..."
    train_re = re.compile(
        r"Epoch\(train\)\s+\[(\d+)\]\[(\d+)/(\d+)\].*?loss:\s*([\d.]+)"
    )
    val_re = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\].*?bbox_mAP:\s*([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = train_re.search(line)
            if m:
                epoch, it, total, loss = int(m[1]), int(m[2]), int(m[3]), float(m[4])
                step = (epoch - 1) * total + it
                records.append({"mode": "train", "epoch": epoch, "step": step, "loss": loss})
                continue
            m = val_re.search(line)
            if m:
                records.append({"mode": "val", "epoch": int(m[1]),
                                 "coco/bbox_mAP": float(m[2])})
    return records


def load_experiment(work_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Devuelve (train_records, val_records) desde work_dir.
    train_records: [{"step": int, "epoch": int, "loss": float, ...}]
    val_records:   [{"epoch": int, "coco/bbox_mAP": float, ...}]
    """
    scalars = find_scalars_files(work_dir)
    if not scalars:
        print(f"  [AVISO] No se encontraron logs en {work_dir}")
        return [], []

    src = scalars[-1]  # usar el más reciente
    print(f"  Leyendo: {src}")

    if src.suffix == ".json" and "scalars" in src.name:
        records = load_scalars_jsonl(src)
    else:
        records = parse_mmengine_log(src)

    train = [r for r in records if r.get("loss") is not None]
    val = [r for r in records
           if r.get("coco/bbox_mAP") is not None or r.get("bbox_mAP") is not None]

    # Normalizar clave mAP
    for r in val:
        if "bbox_mAP" in r and "coco/bbox_mAP" not in r:
            r["coco/bbox_mAP"] = r["bbox_mAP"]

    return train, val


# ─── Análisis de convergencia ─────────────────────────────────────────────────

def convergence_report(val_records: list[dict], window: int = 3) -> dict:
    """
    Determina si el entrenamiento ha convergido mirando las últimas `window` épocas.
    """
    if len(val_records) < 2:
        return {"converged": None, "reason": "datos insuficientes"}

    maps = [r["coco/bbox_mAP"] for r in val_records]
    best_map = max(maps)
    best_ep = val_records[maps.index(best_map)]["epoch"]
    last_map = maps[-1]
    last_ep = val_records[-1]["epoch"]

    if len(maps) >= window + 1:
        recent = maps[-(window + 1):]
        delta = recent[-1] - recent[0]
        converged = abs(delta) < 0.002
    else:
        delta = maps[-1] - maps[0]
        converged = None  # pocas épocas, no concluyente

    return {
        "best_mAP": best_map,
        "best_epoch": best_ep,
        "last_mAP": last_map,
        "last_epoch": last_ep,
        "delta_last_window": delta if len(maps) >= window + 1 else None,
        "converged": converged,
        "total_epochs": last_ep,
        "recommendation": _recommend(converged, delta if len(maps) >= window+1 else None,
                                     best_ep, last_ep),
    }


def _recommend(converged, delta, best_ep, last_ep) -> str:
    if converged is None:
        return "Pocas épocas registradas. Continúa entrenando."
    if converged:
        return (f"✓ Convergido. mAP estable en las últimas épocas. "
                f"Mejor época: {best_ep}. No necesitas más épocas.")
    if delta is not None and delta > 0.002:
        return (f"⚠ mAP sigue subiendo (Δ={delta:.4f} en ventana reciente). "
                f"Considera añadir más épocas (2x o 3x schedule).")
    return f"~ mAP oscilando. Revisa las curvas manualmente."


def print_report(name: str, train: list[dict], val: list[dict]) -> None:
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    if not train and not val:
        print("  Sin datos.")
        return

    if train:
        losses = [r["loss"] for r in train if "loss" in r]
        if losses:
            print(f"  Loss entrenamiento:")
            print(f"    Inicial : {losses[0]:.4f}")
            print(f"    Final   : {losses[-1]:.4f}")
            print(f"    Mínimo  : {min(losses):.4f}")

    if val:
        rep = convergence_report(val)
        print(f"\n  mAP validación por época:")
        for r in val:
            ep = r.get("epoch", "?")
            mp = r.get("coco/bbox_mAP", 0)
            bar = "█" * int(mp * 100)
            mark = " ← BEST" if mp == rep["best_mAP"] else ""
            print(f"    Época {ep:>3}: {mp:.4f}  {bar}{mark}")
        print(f"\n  Mejor mAP : {rep['best_mAP']:.4f} (época {rep['best_epoch']})")
        print(f"  Última mAP: {rep['last_mAP']:.4f} (época {rep['last_epoch']})")
        if rep["delta_last_window"] is not None:
            print(f"  Δ (ventana): {rep['delta_last_window']:+.4f}")
        print(f"\n  → {rep['recommendation']}")


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_curves(
    experiments: dict[str, tuple[list, list]],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [AVISO] matplotlib no disponible. Instala con: pip install matplotlib")
        return

    n = len(experiments)
    fig = plt.figure(figsize=(14, 5 * max(1, (n + 1) // 2)))
    gs = gridspec.GridSpec(max(1, (n + 1) // 2), 2, figure=fig)
    fig.suptitle("Convergencia del entrenamiento de detección", fontsize=14, fontweight="bold")

    for idx, (name, (train, val)) in enumerate(experiments.items()):
        ax_loss = fig.add_subplot(gs[idx // 2, (idx % 2) * 1])

        if train:
            steps = [r.get("step", i) for i, r in enumerate(train)]
            losses = [r["loss"] for r in train]
            ax_loss.plot(steps, losses, alpha=0.4, color="steelblue", linewidth=0.8)
            # Suavizado (media móvil)
            w = max(1, len(losses) // 50)
            smooth = [sum(losses[max(0, i-w):i+1]) / len(losses[max(0, i-w):i+1])
                      for i in range(len(losses))]
            ax_loss.plot(steps, smooth, color="steelblue", linewidth=2, label="loss (suavizado)")

        ax_loss.set_xlabel("Iteración")
        ax_loss.set_ylabel("Loss", color="steelblue")
        ax_loss.tick_params(axis="y", labelcolor="steelblue")
        ax_loss.set_title(name)

        if val:
            ax_map = ax_loss.twinx()
            epochs = [r["epoch"] for r in val]
            maps = [r["coco/bbox_mAP"] for r in val]
            ax_map.plot(
                [steps[min(int(e / max(epochs) * len(steps)) - 1, len(steps)-1)]
                 if train else e for e in epochs],
                maps, "o-", color="tomato", linewidth=2, markersize=5, label="mAP@0.5:0.95"
            )
            ax_map.set_ylabel("mAP", color="tomato")
            ax_map.tick_params(axis="y", labelcolor="tomato")

            # Líneas de convergencia
            rep = convergence_report(val)
            best_ep = rep["best_epoch"]
            best_map = rep["best_mAP"]
            ax_map.axhline(best_map, color="tomato", linestyle="--", alpha=0.4)
            ax_map.annotate(
                f"best: {best_map:.3f} (ep{best_ep})",
                xy=(0.98, best_map),
                xycoords=("axes fraction", "data"),
                ha="right", color="tomato", fontsize=8,
            )

        lines1, labels1 = ax_loss.get_legend_handles_labels()
        if val:
            lines2, labels2 = ax_map.get_legend_handles_labels()
            ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Curvas guardadas en: {out_path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Analiza convergencia del entrenamiento de detección",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--work-dir", "-w", nargs="+", required=True, type=Path,
                   help="Uno o más work_dir de MMDetection")
    p.add_argument("--out", "-o", type=Path,
                   default=Path("outputs/detection/convergence.png"),
                   help="Path de salida para el plot (default: outputs/detection/convergence.png)")
    p.add_argument("--window", type=int, default=3,
                   help="Ventana de épocas para detectar plateau (default: 3)")
    args = p.parse_args()

    print("\n" + "═"*60)
    print("  Check Convergence — MetaFormer Detection")
    print("═"*60)

    experiments: dict[str, tuple[list, list]] = {}
    for wd in args.work_dir:
        name = wd.name
        print(f"\n  Experimento: {name}")
        train, val = load_experiment(wd)
        print(f"  Train records: {len(train)}  |  Val records: {len(val)}")
        experiments[name] = (train, val)
        print_report(name, train, val)

    if any(t or v for t, v in experiments.values()):
        plot_curves(experiments, args.out)

    print("\n" + "═"*60 + "\n")


if __name__ == "__main__":
    main()
