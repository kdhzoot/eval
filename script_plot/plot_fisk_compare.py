#!/usr/bin/env python3
"""Plot and compare two Fisk (log-logistic) PDFs."""

import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisk


def parse_params(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 'shape,loc,scale', got: {text}"
        )
    try:
        shape, loc, scale = map(float, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"All shape/loc/scale must be numeric: {text}"
        ) from exc
    if shape <= 0 or scale <= 0:
        raise argparse.ArgumentTypeError(
            f"shape and scale must be > 0: {text}"
        )
    return shape, loc, scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two Fisk distributions with PDF plot."
    )
    parser.add_argument(
        "--a",
        type=parse_params,
        default="5.50,1.42,0.50",
        help="Params for A as 'shape,loc,scale' (default: 5.50,1.42,0.50)",
    )
    parser.add_argument(
        "--b",
        type=parse_params,
        default="1.38,1.44,0.40",
        help="Params for B as 'shape,loc,scale' (default: 1.38,1.44,0.40)",
    )
    parser.add_argument("--label-a", default="Fisk A", help="Legend label for A")
    parser.add_argument("--label-b", default="Fisk B", help="Legend label for B")
    parser.add_argument("--focus-min", type=float, default=1.60, help="Focused x-range min (default: 1.60)")
    parser.add_argument("--focus-max", type=float, default=2.70, help="Focused x-range max (default: 2.70)")
    parser.add_argument(
        "--points",
        type=int,
        default=2000,
        help="Number of x points (default: 2000)",
    )
    parser.add_argument(
        "--out",
        default="fisk_compare.png",
        help="Output image path (default: fisk_compare.png)",
    )
    args = parser.parse_args()

    a_shape, a_loc, a_scale = args.a
    b_shape, b_loc, b_scale = args.b

    x_min = args.focus_min
    x_max = args.focus_max
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("Invalid focus range. Ensure --focus-max > --focus-min.")

    x = np.linspace(x_min, x_max, args.points)
    pdf_a = fisk.pdf(x, a_shape, loc=a_loc, scale=a_scale)
    pdf_b = fisk.pdf(x, b_shape, loc=b_loc, scale=b_scale)

    mean_a = fisk.mean(a_shape, loc=a_loc, scale=a_scale)
    mean_b = fisk.mean(b_shape, loc=b_loc, scale=b_scale)

    fig, ax_pdf = plt.subplots(1, 1, figsize=(8, 5))

    ax_pdf.plot(
        x, pdf_a, lw=2.2,
        label=f"{args.label_a} (c={a_shape:.2f}, loc={a_loc:.2f}, scale={a_scale:.2f})"
    )
    ax_pdf.plot(
        x, pdf_b, lw=2.2,
        label=f"{args.label_b} (c={b_shape:.2f}, loc={b_loc:.2f}, scale={b_scale:.2f})"
    )
    if np.isfinite(mean_a):
        ax_pdf.axvline(mean_a, color="C0", ls="--", alpha=0.8, label=f"{args.label_a} mean={mean_a:.3f}")
    if np.isfinite(mean_b):
        ax_pdf.axvline(mean_b, color="C1", ls="--", alpha=0.8, label=f"{args.label_b} mean={mean_b:.3f}")
    ax_pdf.set_title(f"Fisk PDF (focused: {x_min:.2f} to {x_max:.2f})")
    ax_pdf.set_xlabel("x")
    ax_pdf.set_ylabel("density")
    ax_pdf.grid(alpha=0.25)
    ax_pdf.legend(fontsize=9)
    ax_pdf.set_xlim(x_min, x_max)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"[DONE] wrote: {args.out}")


if __name__ == "__main__":
    main()
